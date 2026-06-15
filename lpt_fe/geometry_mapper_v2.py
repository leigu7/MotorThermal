"""
Maps a MotorGeometry to a Lumped Parameter Thermal Network (LPTN).
Version 2 - redesigned according to modelingPrinciple.md.

Key design principles:
  1. Slot liner is always present between winding and its surroundings
  2. For slotted: stator_winding → liner → stator_tooth (tangential) AND
                   stator_winding → liner → stator_yoke (radial through slot bottom)
                   stator_tooth → stator_yoke (radial directly, no liner)
                   stator_tooth → airgap (convection)
     Slot opening is small → winding→airgap path neglected
  3. For slotless: stator_winding → liner → stator_yoke (one side)
                   stator_winding → airgap (other side, no tooth)
  4. Rotor: 2D/1-slice (axial symmetry)
  5. Stator: 3 slices for end-winding effects (center + two ends)
"""

import math
import logging
from typing import List, Optional, Dict, Tuple

from lpt_fe.node import ThermalNode, ThermalResistance, ThermalNetwork
from lpt_fe.correlations import (
    AIR_K,
    h_airgap_natural, h_housing_natural, h_housing_forced, h_housing_water_jacket,
    k_slot_equivalent, r_cylindrical_radial, r_cylindrical_axial,
    r_interface_slot_liner, r_convective,
)

logger = logging.getLogger(__name__)


class StatorLPTNBuilder:
    """
    Builds the stator-side thermal network according to modelingPrinciple.md.
    
    For slotted motors:
        Nodes: housing, stator_yoke, stator_tooth, slot_winding, airgap
        Resistances:
            R_yoke_housing: stator_yoke → housing (radial conduction)
            R_tooth_yoke: stator_tooth → stator_yoke (radial conduction)
            R_winding_tooth_wall: slot_winding → stator_tooth (tangential via liner, 2 walls)
            R_winding_yoke_bottom: slot_winding → stator_yoke (radial via liner, slot bottom)
            R_tooth_airgap: stator_tooth → airgap (convection, tooth bore face)
            R_winding_airgap: slot_winding → airgap (convection, slot opening) -- WEAK path
        
        The liner is a thermal resistance layer, not a separate node.
        Winding-tooth contact via slot liner on left+right walls.
        Winding-yoke contact via slot liner on bottom.
    
    For slotless motors:
        Nodes: housing, stator_yoke, slot_winding, airgap
        Resistances:
            R_yoke_housing: stator_yoke → housing (radial conduction)
            R_winding_yoke: slot_winding → stator_yoke (radial via liner)
            R_winding_airgap: slot_winding → airgap (convection, full bore area)
    """
    
    def __init__(self, geo, config, mat, sector_fraction):
        self.geo = geo
        self.config = config
        self.mat = mat
        self.sf = sector_fraction
        self.m = lambda val_mm: val_mm / 1000.0
        
        # Material property helpers
        def k_radial(comp):
            m = self.mat.get(comp)
            return m.k_radial if m else 1.0
        
        def k_axial(comp):
            m = self.mat.get(comp)
            return m.k_axial if m else 1.0
        
        def rho(comp):
            m = self.mat.get(comp)
            return m.rho if m else 1000.0
        
        def cp_val(comp):
            m = self.mat.get(comp)
            return m.cp if m else 500.0
        
        self.k_radial = k_radial
        self.k_axial = k_axial
        self.rho = rho
        self.cp = cp_val
    
    def build_stator_network(self, nodes, resistances, node_idx, 
                              idx_housing, idx_yoke, idx_tooth, idx_winding, idx_airgap):
        """
        Build all stator-side resistances.
        Returns the next available node index.
        
        Parameters:
            nodes: list of ThermalNode (appended to)
            resistances: list of ThermalResistance (appended to)
            node_idx: next available index
            idx_housing: index of housing node
            idx_yoke: index of stator_yoke node
            idx_tooth: index of stator_tooth node (None for slotless)
            idx_winding: index of slot_winding node
            idx_airgap: index of airgap node
        """
        cfg = self.config
        geo = self.geo
        sf = self.sf
        m = self.m
        
        # Convert geometry
        Rso_m = m(geo.Rso)
        Rsi_m = m(geo.Rsi)
        Rro_m = m(geo.Rro)
        Lstk_m = m(geo.stack_length)
        hw_m = m(geo.housing_wall_thickness)
        gap_m = m(geo.airgap_length)
        t_liner_m = m(cfg.slot_liner_thickness)
        
        s = geo.slot
        is_slotted = geo.structure_type.lower() == "slotted"
        
        if is_slotted:
            Hs0_m = m(s.Hs0)
            Hs1_m = m(s.Hs1)
            Hs2_m = m(s.Hs2)
            Bs0_m = m(s.Bs0)
            Bs1_m = m(s.Bs1)
            Bs2_m = m(s.Bs2)
            N_slots = geo.num_slots
            slot_pitch_angle = 2 * math.pi / N_slots
            Rslot_bottom_m = Rsi_m + Hs0_m + Hs1_m + Hs2_m
        else:
            Hs0_m = Hs1_m = Hs2_m = 0
            Bs0_m = Bs1_m = Bs2_m = 0
            N_slots = 0
            slot_pitch_angle = 0
            Rslot_bottom_m = m(geo.winding_outer_radius) if hasattr(geo, 'winding_outer_radius') else Rsi_m
        
        def add_R(name, fi, to, R, rtype="conduction", **kw):
            override = cfg.resistance_overrides.get(name)
            h = kw.pop('h_coefficient', None)
            if h is None and rtype == "convection" and R > 0:
                area = kw.get('effective_area', 0)
                if area > 0:
                    h = 1.0 / (R * area)
            if h is not None:
                kw['h_coefficient'] = h
            resistances.append(ThermalResistance(
                name=name, node_from=fi, node_to=to,
                resistance=R, resistance_type=rtype,
                user_override=override,
                **kw
            ))
        
        # ---- R1: Housing → Ambient (convection) ----
        R_housing_outer_m = Rso_m + hw_m
        A_housing = 2 * math.pi * R_housing_outer_m * Lstk_m * sf
        A_endcaps = 2 * math.pi * R_housing_outer_m ** 2 * sf
        A_housing_total = A_housing + A_endcaps
        
        if cfg.cooling_mode == "TEFC":
            h_hsg = h_housing_forced(R_housing_outer_m, cfg.housing_air_speed or 3.0)
        elif cfg.cooling_mode == "Water Jacket":
            h_hsg = h_housing_water_jacket(
                cfg.coolant_flow_rate, 0.008, cfg.coolant_temperature
            )
        else:  # TENV
            h_hsg = h_housing_natural(R_housing_outer_m, Lstk_m, 70, cfg.ambient_temperature)
        
        R_housing_ambient = r_convective(A_housing_total, h_hsg)
        add_R("R_housing_ambient", idx_housing, 0, R_housing_ambient, "convection",
              effective_area=A_housing_total, h_coefficient=h_hsg)
        
        # ---- R2: Stator Yoke → Housing (radial conduction) ----
        R_yoke_housing = r_cylindrical_radial(
            Rso_m, Rso_m + hw_m, Lstk_m, self.k_radial("housing"), sf
        )
        add_R("R_yoke_housing", idx_yoke, idx_housing, R_yoke_housing,
              effective_length=hw_m, effective_area=2*math.pi*Rso_m*Lstk_m*sf,
              conductivity=self.k_radial("housing"))
        
        if is_slotted:
            # ---- R3: Stator Tooth → Stator Yoke (radial conduction, direct) ----
            # Tooth cross-section at slot bottom
            tooth_arc_at_bottom = slot_pitch_angle * Rslot_bottom_m - Bs2_m
            contact_area = tooth_arc_at_bottom * Lstk_m * N_slots * sf
            
            R_tooth_yoke = r_cylindrical_radial(
                Rslot_bottom_m, Rso_m, Lstk_m, self.k_radial("stator_core"),
                sector_fraction=sf * N_slots * (tooth_arc_at_bottom / (slot_pitch_angle * Rslot_bottom_m))
            )
            add_R("R_tooth_yoke", idx_tooth, idx_yoke, R_tooth_yoke,
                  effective_length=Rso_m - Rslot_bottom_m,
                  effective_area=contact_area,
                  conductivity=self.k_radial("stator_core"))
            
            # ---- R4: Slot Winding → Stator Tooth (tangential, through liner on 2 walls) ----
            # The winding contacts the left and right tooth walls via slot liner.
            # Slot width at mid-height
            R_mid_slot_m = Rsi_m + Hs0_m + (Hs1_m + Hs2_m) / 2
            w_slot_mid = (Bs1_m + Bs2_m) / 2  # average tangential width of slot
            
            fill = geo.fill_factor if hasattr(geo, 'fill_factor') else 0.45
            k_eq = k_slot_equivalent(fill)
            k_winding_radial = k_eq["k_radial"]
            
            # Effective conduction length from winding center to wall
            # For a slab with two-side cooling: L_eff = half-width / 3
            L_eff_wall = w_slot_mid / 6.0
            
            # Contact area per wall: slot body height × stack length
            A_wall = Lstk_m * (Hs1_m + Hs2_m)
            
            # Winding body resistance to one wall
            R_body_one_wall = L_eff_wall / (k_winding_radial * A_wall) if A_wall > 0 else 1e6
            
            # Slot liner resistance (Nomex paper, thickness = t_liner_m)
            k_liner = cfg.liner_conductivity
            R_liner_wall = t_liner_m / (k_liner * A_wall) if A_wall > 0 else 1e6
            
            # Total per wall: body + liner in series
            R_per_wall = R_body_one_wall + R_liner_wall
            
            # Two walls in parallel
            R_winding_tooth = R_per_wall / 2.0 if R_per_wall > 0 else 1e6
            
            # Scale for sector/N_slots
            if sf < 1.0:
                R_winding_tooth *= (1.0 / (sf * N_slots))
            else:
                R_winding_tooth /= N_slots
            
            add_R("R_winding_tooth", idx_winding, idx_tooth, R_winding_tooth,
                  "interface",
                  effective_length=L_eff_wall,
                  effective_area=2 * A_wall * N_slots * sf,
                  conductivity=k_winding_radial,
                  liner_thickness=t_liner_m,
                  liner_conductivity=k_liner)
            
            # ---- R5: Slot Winding → Stator Yoke (radial, through liner at slot bottom) ----
            # Heat flows from winding body through slot bottom liner to yoke
            A_bottom_per_slot = Lstk_m * Bs2_m
            A_bottom_total = A_bottom_per_slot * N_slots * sf
            
            if A_bottom_total > 0 and Hs2_m > 0:
                # Winding body resistance (half slot height)
                R_body_bottom = (Hs2_m / 2) / (k_winding_radial * A_bottom_total)
                # Slot liner at bottom
                R_liner_bottom = t_liner_m / (k_liner * A_bottom_total)
                R_winding_yoke_bottom = R_body_bottom + R_liner_bottom
            else:
                R_winding_yoke_bottom = 1e6
            
            add_R("R_winding_yoke_bottom", idx_winding, idx_yoke, R_winding_yoke_bottom,
                  "interface",
                  effective_length=Hs2_m / 2 if Hs2_m > 0 else 0,
                  effective_area=A_bottom_total,
                  conductivity=k_winding_radial,
                  liner_thickness=t_liner_m,
                  liner_conductivity=k_liner)
            
            # ---- R6: Stator Tooth → Airgap (convection, tooth face at bore) ----
            tooth_arc_at_bore = slot_pitch_angle * Rsi_m - Bs0_m
            A_tooth_bore = tooth_arc_at_bore * Lstk_m * N_slots * sf
            
            if A_tooth_bore > 0:
                h_ag_tooth = h_airgap_natural(Rsi_m, gap_m, Lstk_m, 0)  # stationary side
                R_tooth_airgap = r_convective(A_tooth_bore, h_ag_tooth)
                add_R("R_tooth_airgap", idx_tooth, idx_airgap, R_tooth_airgap, "convection",
                      effective_area=A_tooth_bore, h_coefficient=h_ag_tooth)
            
            # ---- R7: Slot Winding → Airgap (convection through slot opening) ----
            # According to modelingPrinciple.md: slot opening is small so this can be neglected.
            # But we keep it as a weak coupling path (high R).
            A_winding_bore = Bs0_m * Lstk_m * N_slots * sf
            if A_winding_bore > 0:
                h_ag_winding = h_airgap_natural(Rsi_m, gap_m, Lstk_m, 0)
                R_winding_airgap = r_convective(A_winding_bore, h_ag_winding)
                add_R("R_winding_airgap", idx_winding, idx_airgap, R_winding_airgap, "convection",
                      effective_area=A_winding_bore, h_coefficient=h_ag_winding)
        
        else:
            # ---- Slotless motor ----
            # No tooth. Winding contacts yoke (via liner) and airgap directly.
            
            Rwi_m = m(geo.winding_inner_radius) if hasattr(geo, 'winding_inner_radius') else Rsi_m
            Rwo_m = m(geo.winding_outer_radius) if hasattr(geo, 'winding_outer_radius') else Rslot_bottom_m
            
            # ---- R3: Slot Winding → Stator Yoke (radial, via liner) ----
            # Winding outer radius to yoke inner radius
            # If winding directly contacts yoke at Rwo, the gap is just the liner
            if Rwo_m < Rso_m:
                # Liner at winding-to-yoke interface
                A_winding_yoke = 2 * math.pi * Rwo_m * Lstk_m * sf
                k_liner = cfg.liner_conductivity
                R_liner_wall = t_liner_m / (k_liner * A_winding_yoke)
                
                # Winding body resistance (radial half-thickness)
                t_winding_radial = Rwo_m - Rwi_m
                k_winding_radial = self.k_radial("winding")
                R_body = (t_winding_radial / 2) / (k_winding_radial * A_winding_yoke)
                
                R_winding_yoke = R_body + R_liner_wall
            else:
                R_winding_yoke = 0.01
            
            add_R("R_winding_yoke", idx_winding, idx_yoke, R_winding_yoke,
                  "interface",
                  effective_length=(Rwo_m - Rwi_m) / 2 if Rwo_m > Rwi_m else 0,
                  effective_area=2 * math.pi * Rwo_m * Lstk_m * sf,
                  conductivity=self.k_radial("winding"),
                  liner_thickness=t_liner_m)
            
            # ---- R4: Slot Winding → Airgap (convection, full bore area) ----
            A_winding_airgap = 2 * math.pi * Rwi_m * Lstk_m * sf
            h_ag_winding = h_airgap_natural(Rwi_m, gap_m, Lstk_m, 0)
            R_winding_airgap = r_convective(A_winding_airgap, h_ag_winding)
            add_R("R_winding_airgap", idx_winding, idx_airgap, R_winding_airgap, "convection",
                  effective_area=A_winding_airgap, h_coefficient=h_ag_winding)
        
        return node_idx


class GeometryMapperV2:
    """
    Builds a ThermalNetwork from a MotorGeometry.
    Version 2 implementing modelingPrinciple.md topology.
    """
    
    def __init__(self, geo, config=None):
        from geometry.motor_geometry_v2 import MotorGeometry
        self.geo = geo
        if config is None:
            from lpt_fe.geometry_mapper import NetworkBuilderConfig
            config = NetworkBuilderConfig()
        self.config = config
        self._load_materials()
    
    def _load_materials(self):
        from materials.material_db import get_material
        self.mat = {}
        for comp, mat_name in self.config.materials.items():
            m = get_material(mat_name)
            if m is None:
                logger.warning(f"Material '{mat_name}' not found for {comp}")
                self.mat[comp] = None
            else:
                self.mat[comp] = m
    
    def _sector_fraction(self) -> float:
        if not self.config.use_sector:
            return 1.0
        return 1.0 / self.geo.num_poles * self.config.sector_n_poles
    
    def build(self) -> ThermalNetwork:
        """
        Build the full thermal network according to modelingPrinciple.md.
        
        Node numbering:
          0: Ambient (BC)
          1: Housing
          2: Stator Yoke
          3: Stator Tooth (only for slotted)
          4: Slot Winding
          5: Airgap
          6: Magnet
          7: Rotor Core
          8: Shaft
          9: Shaft End (exposed shaft ends, convects to ambient)
          10: End Cap (housing end plates, connects housing to bearing/shaft)
        """
        geo = self.geo
        cfg = self.config
        sf = self._sector_fraction()
        m = lambda val_mm: val_mm / 1000.0
        
        is_slotted = geo.structure_type.lower() == "slotted"
        
        # ---- Geometry helpers ----
        def k_radial(comp):
            mat = self.mat.get(comp)
            return mat.k_radial if mat else 1.0
        
        def k_axial(comp):
            mat = self.mat.get(comp)
            return mat.k_axial if mat else 1.0
        
        def rho(comp):
            mat = self.mat.get(comp)
            return mat.rho if mat else 1000.0
        
        def cp_val(comp):
            mat = self.mat.get(comp)
            return mat.cp if mat else 500.0
        
        # ---- Geometry with zero-radius guards ----
        Rso_m = m(geo.Rso)
        Rsi_m = m(geo.Rsi)
        Rro_m = m(geo.Rro)
        Rri_m = max(m(geo.Rri), 1e-4)  # prevent zero radius
        Rshaft_m = max(m(geo.shaft_radius), 1e-4)  # prevent zero radius
        Lstk_m = m(geo.stack_length)
        hw_m = m(geo.housing_wall_thickness)
        t_mag_m = m(geo.magnet_thickness)
        gap_m = m(geo.airgap_length)
        
        # Clamp radii to maintain physical consistency
        Rshaft_m = min(Rshaft_m, Rri_m - 1e-6) if Rri_m > Rshaft_m + 1e-6 else Rshaft_m
        Rri_m = max(Rri_m, Rshaft_m + 1e-6)
        
        s = geo.slot
        if is_slotted:
            Hs0_m = m(s.Hs0)
            Hs1_m = m(s.Hs1)
            Hs2_m = m(s.Hs2)
            Bs0_m = m(s.Bs0)
            Bs1_m = m(s.Bs1)
            Bs2_m = m(s.Bs2)
            N_slots = geo.num_slots
            slot_pitch_angle = 2 * math.pi / N_slots
            Rslot_bottom_m = Rsi_m + Hs0_m + Hs1_m + Hs2_m
        else:
            Hs0_m = Hs1_m = Hs2_m = 0
            Bs0_m = Bs1_m = Bs2_m = 0
            N_slots = 0
            slot_pitch_angle = 0
            Rslot_bottom_m = m(geo.winding_outer_radius) if hasattr(geo, 'winding_outer_radius') else Rsi_m
        
        # ---- Create nodes ----
        nodes = []
        idx = 0
        
        # Node 0: Ambient
        nodes.append(ThermalNode(
            name="ambient", index=idx, volume=0, density=1, cp=1,
            fixed_temperature=cfg.ambient_temperature
        ))
        idx += 1
        
        # Node 1: Housing
        V_housing = math.pi * ((Rso_m + hw_m) ** 2 - Rso_m ** 2) * Lstk_m * sf
        nodes.append(ThermalNode(
            name="housing", index=idx, volume=V_housing,
            density=rho("housing"), cp=cp_val("housing"), loss=0,
        ))
        idx_housing = idx
        idx += 1
        
        # Node 2: Stator Yoke
        if is_slotted:
            V_yoke = math.pi * (Rso_m ** 2 - Rslot_bottom_m ** 2) * Lstk_m * sf
        else:
            Rwo_m = m(geo.winding_outer_radius) if hasattr(geo, 'winding_outer_radius') else Rslot_bottom_m
            V_yoke = math.pi * (Rso_m ** 2 - Rwo_m ** 2) * Lstk_m * sf
        
        nodes.append(ThermalNode(
            name="stator_yoke", index=idx, volume=V_yoke,
            density=rho("stator_core"), cp=cp_val("stator_core"),
            loss=cfg.loss_iron_yoke,
            loss_user_override=cfg.loss_overrides.get("stator_yoke"),
        ))
        idx_yoke = idx
        idx += 1
        
        # Node 3: Stator Tooth (slotted only)
        idx_tooth = None
        if is_slotted:
            # Tooth volume including tip (Hs0)
            tooth_bottom_arc = slot_pitch_angle * Rslot_bottom_m - Bs2_m
            R_shoulder_m = Rsi_m + Hs0_m + Hs1_m
            tooth_top_arc = slot_pitch_angle * R_shoulder_m - Bs1_m
            avg_tooth_width = (tooth_top_arc + tooth_bottom_arc) / 2
            
            V_teeth = N_slots * avg_tooth_width * (Hs1_m + Hs2_m) * Lstk_m * sf
            # Add tip volume
            tip_arc = slot_pitch_angle * Rsi_m - Bs0_m
            V_teeth += N_slots * tip_arc * Hs0_m * Lstk_m * sf
            
            nodes.append(ThermalNode(
                name="stator_tooth", index=idx, volume=max(V_teeth, 1e-12),
                density=rho("stator_core"), cp=cp_val("stator_core"),
                loss=cfg.loss_iron_teeth,
                loss_user_override=cfg.loss_overrides.get("stator_teeth"),
            ))
            idx_tooth = idx
            idx += 1
        
        # Node 4: Slot Winding
        if is_slotted:
            avg_slot_width = (Bs0_m + Bs2_m) / 2
            slot_area = avg_slot_width * (Hs1_m + Hs2_m)
            V_winding_slot = N_slots * slot_area * Lstk_m * geo.fill_factor * sf
        else:
            Rwi_m = m(geo.winding_inner_radius) if hasattr(geo, 'winding_inner_radius') else Rsi_m
            Rwo_m = m(geo.winding_outer_radius) if hasattr(geo, 'winding_outer_radius') else Rslot_bottom_m
            V_winding_slot = math.pi * (Rwo_m ** 2 - Rwi_m ** 2) * Lstk_m * sf
        
        nodes.append(ThermalNode(
            name="slot_winding", index=idx, volume=max(V_winding_slot, 1e-12),
            density=rho("winding"), cp=cp_val("winding"),
            loss=cfg.loss_copper_slot,
            loss_temperature_dependent=True,
            loss_ref_temp=20.0,
            loss_alpha=0.00393,
            loss_user_override=cfg.loss_overrides.get("slot_winding"),
        ))
        idx_winding = idx
        idx += 1
        
        # Node 5: Airgap
        V_airgap = math.pi * (Rsi_m ** 2 - Rro_m ** 2) * Lstk_m * sf
        rho_air = 1.05
        cp_air = 1005
        nodes.append(ThermalNode(
            name="airgap", index=idx, volume=max(V_airgap, 1e-12),
            density=rho_air, cp=cp_air, loss=0,
        ))
        idx_airgap = idx
        idx += 1
        
        # Node 6: Magnet
        pole_angle = 2 * math.pi / geo.num_poles
        magnet_arc = pole_angle * geo.magnet_span_ratio
        Rmag_outer_m = Rro_m
        Rmag_inner_m = Rro_m - t_mag_m
        mag_area_per_pole = 0.5 * magnet_arc * (Rmag_outer_m ** 2 - Rmag_inner_m ** 2)
        V_mag = geo.num_poles * mag_area_per_pole * Lstk_m * sf
        
        nodes.append(ThermalNode(
            name="magnet", index=idx, volume=max(V_mag, 1e-12),
            density=rho("magnet"), cp=cp_val("magnet"),
            loss=cfg.loss_magnet,
            loss_user_override=cfg.loss_overrides.get("magnet"),
        ))
        idx_magnet = idx
        idx += 1
        
        # Node 7: Rotor Core
        rotor_core_outer_m = Rro_m - t_mag_m
        V_rotor = math.pi * (rotor_core_outer_m ** 2 - Rri_m ** 2) * Lstk_m * sf
        nodes.append(ThermalNode(
            name="rotor_core", index=idx, volume=max(V_rotor, 1e-12),
            density=rho("rotor_core"), cp=cp_val("rotor_core"), loss=0,
        ))
        idx_rotor = idx
        idx += 1
        
        # Node 8: Shaft (in-stack portion)
        V_shaft = math.pi * Rri_m ** 2 * Lstk_m * sf
        nodes.append(ThermalNode(
            name="shaft", index=idx, volume=max(V_shaft, 1e-12),
            density=rho("shaft"), cp=cp_val("shaft"), loss=0,
        ))
        idx_shaft = idx
        idx += 1
        
        # Node 9: Shaft End (exposed shaft ends beyond stator, both ends)
        L_shaft_end = 0.030  # [m] ~30mm shaft extension each side
        V_shaft_end = math.pi * Rshaft_m ** 2 * (2 * L_shaft_end) * sf
        nodes.append(ThermalNode(
            name="shaft_end", index=idx, volume=max(V_shaft_end, 1e-12),
            density=rho("shaft"), cp=cp_val("shaft"), loss=0,
        ))
        idx_shaft_end = idx
        idx += 1
        
        # Node 10: End Cap (housing end plate on each side)
        # Annular disc: outer = housing outer radius, inner = shaft clearance
        t_endcap = 0.005  # [m] ~5mm end cap thickness
        R_endcap_outer = Rso_m + hw_m
        R_endcap_inner = min(Rshaft_m, R_endcap_outer - 1e-6)  # prevent zero-thickness annulus
        V_endcap = max(math.pi * (R_endcap_outer ** 2 - R_endcap_inner ** 2) * (2 * t_endcap) * sf, 1e-12)
        nodes.append(ThermalNode(
            name="end_cap", index=idx, volume=max(V_endcap, 1e-12),
            density=rho("housing"), cp=cp_val("housing"), loss=0,
        ))
        idx_endcap = idx
        idx += 1
        
        # ---- Build resistances ----
        resistances = []
        
        def add_R(name, fi, to, R, rtype="conduction", **kw):
            override = cfg.resistance_overrides.get(name)
            h = kw.pop('h_coefficient', None)
            if h is None and rtype == "convection" and R > 0:
                area = kw.get('effective_area', 0)
                if area > 0:
                    h = 1.0 / (R * area)
            if h is not None:
                kw['h_coefficient'] = h
            resistances.append(ThermalResistance(
                name=name, node_from=fi, node_to=to,
                resistance=R, resistance_type=rtype,
                user_override=override,
                **kw
            ))
        
        # ---- Use StatorLPTNBuilder for stator side ----
        stator_builder = StatorLPTNBuilder(geo, cfg, self.mat, sf)
        stator_builder.build_stator_network(
            nodes, resistances, idx,
            idx_housing, idx_yoke, idx_tooth, idx_winding, idx_airgap
        )
        
        # ---- Rotor side resistances ----
        
        # R8: Airgap → Magnet (rotating convection)
        h_ag_rotor = h_airgap_natural(Rro_m, gap_m, Lstk_m, cfg.speed_rpm)
        A_airgap_rotor = 2 * math.pi * Rro_m * Lstk_m * sf
        R_airgap_magnet = r_convective(A_airgap_rotor, h_ag_rotor)
        add_R("R_airgap_magnet", idx_airgap, idx_magnet, R_airgap_magnet, "convection",
              effective_area=A_airgap_rotor, h_coefficient=h_ag_rotor)
        
        # R9: Rotor Core → Magnet (radial conduction)
        R_rotor_magnet = r_cylindrical_radial(
            Rmag_inner_m, Rmag_outer_m, Lstk_m, k_radial("magnet"), sf
        )
        add_R("R_rotor_magnet", idx_rotor, idx_magnet, R_rotor_magnet,
              effective_length=t_mag_m,
              effective_area=2 * math.pi * Rmag_inner_m * Lstk_m * sf,
              conductivity=k_radial("magnet"))
        
        # R10: Shaft → Rotor Core (radial conduction)
        if Rshaft_m > 0 and Rri_m > Rshaft_m and Lstk_m > 0:
            R_shaft_rotor = r_cylindrical_radial(
                Rshaft_m, Rri_m, Lstk_m, k_radial("shaft"), sf
            )
            area_shaft = 2 * math.pi * Rshaft_m * Lstk_m * sf
            add_R("R_shaft_rotor", idx_shaft, idx_rotor, R_shaft_rotor,
                  effective_length=Rri_m - Rshaft_m,
                  effective_area=area_shaft,
                  conductivity=k_radial("shaft"))
        else:
            # Zero-radius shaft: treat as merged with rotor core
            add_R("R_shaft_rotor", idx_shaft, idx_rotor, 0.001,
                  effective_length=0, effective_area=0, conductivity=0)
        
        # R11: Rotor Core → Shaft (alternate, inner path)
        if Rri_m > Rshaft_m:
            R_rotor_shaft_inner = r_cylindrical_radial(
                Rshaft_m, Rri_m, Lstk_m, k_radial("rotor_core"), sf
            )
        else:
            R_rotor_shaft_inner = 0.001
        add_R("R_rotor_shaft_inner", idx_rotor, idx_shaft, R_rotor_shaft_inner)
        
        # ---- Rotor cooling paths (to ambient) ----
        
        # R12: Shaft → Shaft End (axial conduction through shaft)
        L_axial_shaft = L_shaft_end  # half the exposed length
        A_shaft_axial = math.pi * Rshaft_m ** 2
        if A_shaft_axial > 1e-12:
            R_shaft_shaftend = L_axial_shaft / (k_axial("shaft") * A_shaft_axial)
        else:
            R_shaft_shaftend = 0.001
        add_R("R_shaft_shaftend", idx_shaft, idx_shaft_end, R_shaft_shaftend,
              effective_length=L_axial_shaft,
              effective_area=A_shaft_axial,
              conductivity=k_axial("shaft"))
        
        # R13: Shaft End → Ambient (convection from exposed shaft surface)
        A_shaft_end_surf = 2 * math.pi * Rshaft_m * (2 * L_shaft_end) * sf  # cylindrical surface
        A_shaft_end_face = 2 * math.pi * Rshaft_m ** 2 * sf  # end faces
        A_shaft_end_total = max(A_shaft_end_surf + A_shaft_end_face, 1e-12)
        # Natural convection from shaft (h ~ 10-15 W/m2K for still air)
        h_shaft_end = 12.0  # [W/m2K] moderate natural convection
        R_shaft_end_ambient = 1.0 / (h_shaft_end * A_shaft_end_total)
        add_R("R_shaft_end_ambient", idx_shaft_end, 0, R_shaft_end_ambient, "convection",
              effective_area=A_shaft_end_total, h_coefficient=h_shaft_end)
        
        # R14: Housing → End Cap (axial conduction through housing walls)
        # Both ends: housing end face to end cap
        A_housing_to_endcap = max(math.pi * ((Rso_m + hw_m) ** 2 - Rso_m ** 2) * sf, 1e-12)
        R_housing_endcap = t_endcap / (k_axial("housing") * A_housing_to_endcap)
        add_R("R_housing_endcap", idx_housing, idx_endcap, R_housing_endcap,
              effective_length=t_endcap,
              effective_area=A_housing_to_endcap,
              conductivity=k_axial("housing"))
        
        # R15: End Cap → Ambient (convection from end cap faces)
        # Both end caps: outer face exposed to ambient
        A_endcap_outer = math.pi * (R_endcap_outer ** 2 - R_endcap_inner ** 2) * sf  # one side
        A_endcap_total = 2 * A_endcap_outer * sf  # both caps, both sides? no, one side each, outer face
        # Actually: each cap has inner face (housing side) and outer face (ambient side)
        # Outer face convects; inner face conducts to housing
        A_endcap_ambient = 2 * (math.pi * R_endcap_outer ** 2 * sf)  # both end caps, outer surfaces
        h_endcap = 8.0  # [W/m2K] natural convection from end cap (slightly less than housing)
        R_endcap_ambient = 1.0 / (h_endcap * A_endcap_ambient)
        add_R("R_endcap_ambient", idx_endcap, 0, R_endcap_ambient, "convection",
              effective_area=A_endcap_ambient, h_coefficient=h_endcap)
        
        # R16: Shaft → End Cap (through bearing)
        # Bearing thermal resistance: typical 0.1-0.5 K/W for small bearings
        # Scale by sector fraction
        R_bearing = cfg.shaft_to_endcap_resistance if hasattr(cfg, 'shaft_to_endcap_resistance') else 0.3
        # For sector model: bearings are full-circle, but we distribute
        R_bearing_scaled = R_bearing / sf if sf > 0 else R_bearing
        add_R("R_shaft_endcap_bearing", idx_shaft, idx_endcap, R_bearing_scaled, "interface",
              effective_length=0, effective_area=0, conductivity=0)
        
        # ---- Create network ----
        net = ThermalNetwork(nodes, resistances, f"2D LPTN v2 ({cfg.cooling_mode})")
        return net


def build_thermal_network_v2(geo, config=None):
    """
    Convenience function: build a ThermalNetwork using the v2 topology.
    
    Parameters:
        geo: MotorGeometry instance
        config: NetworkBuilderConfig (or None for defaults)
    
    Returns:
        ThermalNetwork ready for solving
    """
    mapper = GeometryMapperV2(geo, config)
    return mapper.build()
