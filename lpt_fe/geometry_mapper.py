"""
Maps a MotorGeometry to a Lumped Parameter Thermal Network (LPTN).
Supports 2D (radial only) and 3D (radial + axial slices) modes.
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


class NetworkBuilderConfig:
    """
    Configuration for the thermal network builder.
    
    All fields have defaults; user can override any.
    """
    def __init__(self):
        # Dimensionality
        self.dimensionality: str = "2D"         # "2D" or "3D"
        
        # Sector model
        self.use_sector: bool = False           # True = 1/N pole model
        self.sector_n_poles: int = 1            # N in 1/N
        
        # Cooling mode
        self.cooling_mode: str = "TENV"         # "TENV", "TEFC", "Water Jacket"
        
        # External conditions
        self.ambient_temperature: float = 40.0   # [°C]
        self.housing_air_speed: float = 0.0      # [m/s] for forced convection
        self.coolant_flow_rate: float = 10.0     # [L/min] for water jacket
        self.coolant_temperature: float = 50.0   # [°C] for water jacket
        
        # Speed
        self.speed_rpm: float = 3000.0
        
        # Slot liner properties (for winding insulation)
        self.slot_liner_thickness: float = 0.3     # [mm] slot liner paper thickness
        self.liner_conductivity: float = 0.25      # [W/mK] liner material k (Nomex ~0.25)
        
        # 3D settings
        self.n_axial_slices: int = 3
        self.end_winding_length: Optional[float] = None  # [m] auto-calculated if None
        
        # Material assignments (maps component name to material name)
        self.materials: Dict[str, str] = {
            "stator_core": "M19_24Ga",
            "rotor_core": "M19_24Ga",
            "magnet": "N42UH",
            "winding": "Winding_Eq",
            "housing": "Al6061",
            "shaft": "Shaft_Steel",
            "slot_liner": "Slot_Liner",
        }
        
        # Losses [W]
        self.loss_copper_slot: float = 60.0
        self.loss_copper_end: float = 20.0
        self.loss_iron_yoke: float = 15.0
        self.loss_iron_teeth: float = 10.0
        self.loss_magnet: float = 5.0
        self.loss_mechanical: float = 5.0
        
        # User resistance overrides (None = use calculated)
        # Key format: "R_shaft_rotor", "R_rotor_magnet", etc.
        self.resistance_overrides: Dict[str, Optional[float]] = {}
        
        # User loss overrides (None = use calculated/auto)
        self.loss_overrides: Dict[str, Optional[float]] = {}

    def to_dict(self) -> Dict:
        """Return config as a JSON-serializable dict."""
        return {
            "dimensionality": self.dimensionality,
            "use_sector": self.use_sector,
            "sector_n_poles": self.sector_n_poles,
            "cooling_mode": self.cooling_mode,
            "ambient_temperature": self.ambient_temperature,
            "housing_air_speed": self.housing_air_speed,
            "coolant_flow_rate": self.coolant_flow_rate,
            "coolant_temperature": self.coolant_temperature,
            "speed_rpm": self.speed_rpm,
            "loss_copper_slot": self.loss_copper_slot,
            "loss_copper_end": self.loss_copper_end,
            "loss_iron_yoke": self.loss_iron_yoke,
            "loss_iron_teeth": self.loss_iron_teeth,
            "loss_magnet": self.loss_magnet,
            "loss_mechanical": self.loss_mechanical,
            "materials": dict(self.materials),
        }

    @staticmethod
    def from_dict(data: Dict) -> "NetworkBuilderConfig":
        """Restore from a dict."""
        cfg = NetworkBuilderConfig()
        for k, v in data.items():
            if hasattr(cfg, k):
                if isinstance(getattr(cfg, k), dict) and isinstance(v, dict):
                    getattr(cfg, k).update(v)
                else:
                    setattr(cfg, k, v)
        return cfg


class GeometryMapper:
    """
    Builds a ThermalNetwork from a MotorGeometry based on configuration.
    """
    
    def __init__(self, geo, config: NetworkBuilderConfig = None):
        """
        Parameters:
            geo: MotorGeometry instance
            config: NetworkBuilderConfig (or None for defaults)
        """
        from geometry.motor_geometry_v2 import MotorGeometry
        self.geo = geo
        self.config = config or NetworkBuilderConfig()
        
        # Get materials
        self._load_materials()
        
    def _load_materials(self):
        """Load material properties from the database."""
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
        """Fraction of full circle for sector model."""
        if not self.config.use_sector:
            return 1.0
        return 1.0 / self.geo.num_poles * self.config.sector_n_poles
    
    def _mm_to_m(self, val_mm: float) -> float:
        """Convert mm to m."""
        return val_mm / 1000.0
    
    def build_2d(self) -> ThermalNetwork:
        """
        Build a 2D radial thermal network.
        
        Nodes (in radial order):
          0: Ambient (BC, fixed T)
          1: Housing
          2: Stator Yoke
          3: Stator Tooth Body (including tip region Hs0)
          4: Slot Winding (homogenized)
          5: Airgap (separate thermal node)
          6: Magnet
          7: Rotor Core
          8: Shaft
        
        The airgap is a separate thermal node connected to:
          - Stator Tooth and Slot Winding (stationary side, in parallel)
          - Magnet (rotating side, Taylor number correlation)
        """
        geo = self.geo
        cfg = self.config
        sf = self._sector_fraction()  # sector fraction (1.0 = full motor)
        
        m = self._mm_to_m
        
        # ---- Convert geometry to meters ----
        Rso_m = m(geo.Rso)
        Rsi_m = m(geo.Rsi)
        Rro_m = m(geo.Rro)
        Rri_m = m(geo.Rri)
        Rshaft_m = m(geo.shaft_radius)
        Lstk_m = m(geo.stack_length)
        hw_m = m(geo.housing_wall_thickness)
        t_mag_m = m(geo.magnet_thickness)
        gap_m = m(geo.airgap_length)
        
        s = geo.slot
        is_slotted = geo.structure_type.lower() == "slotted"
        
        # Slot geometry in meters
        if is_slotted:
            Hs0_m = m(s.Hs0)
            Hs1_m = m(s.Hs1)
            Hs2_m = m(s.Hs2)
            Bs0_m = m(s.Bs0)
            Bs1_m = m(s.Bs1)
            Bs2_m = m(s.Bs2)
            Rs_fillet_m = m(s.Rs_fillet)
            N_slots = geo.num_slots
            slot_pitch_angle = 2 * math.pi / N_slots
        else:
            Hs0_m = Hs1_m = Hs2_m = 0
            Bs0_m = Bs1_m = Bs2_m = 0
            N_slots = 0
            slot_pitch_angle = 0
        
        # Slot bottom radius (where tooth meets yoke)
        if is_slotted:
            Rslot_bottom_m = Rsi_m + Hs0_m + Hs1_m + Hs2_m
        else:
            Rslot_bottom_m = m(geo.winding_outer_radius) if hasattr(geo, 'winding_outer_radius') else Rsi_m
        
        # ---- Create nodes ----
        nodes = []
        
        # Helper to get material property safely
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
        
        idx = 0
        
        # Node 0: Ambient (boundary condition)
        nodes.append(ThermalNode(
            name="ambient", index=idx, volume=0, density=1, cp=1,
            fixed_temperature=cfg.ambient_temperature
        ))
        idx += 1
        
        # ---- Volumes (all in m³, scaled by sector fraction) ----
        
        # 1. Housing
        V_housing = math.pi * ((Rso_m + hw_m) ** 2 - Rso_m ** 2) * Lstk_m * sf
        nodes.append(ThermalNode(
            name="housing", index=idx, volume=V_housing,
            density=rho("housing"), cp=cp_val("housing"),
            loss=0,  # housing generates no heat
        ))
        idx_housing = idx
        idx += 1
        
        # 2. Stator Yoke (outer ring, from Rslot_bottom to Rso)
        if is_slotted:
            V_yoke = math.pi * (Rso_m ** 2 - Rslot_bottom_m ** 2) * Lstk_m * sf
        else:
            # For slotless: yoke is from winding_outer_radius to Rso
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
        
        # 3. Stator Tooth Body (between slots)
        if is_slotted:
            # Tooth volume per slot: tooth width × slot height × Lstk
            # Tooth bottom width = slot_pitch_at_slot_bottom - Bs2
            tooth_bottom_arc = slot_pitch_angle * Rslot_bottom_m - Bs2_m
            # Tooth top width (at shoulder) = slot_pitch_at_shoulder - Bs1
            R_shoulder_m = Rsi_m + Hs0_m + Hs1_m
            tooth_top_arc = slot_pitch_angle * R_shoulder_m - Bs1_m
            
            # Average tooth width for body region (Hs1 + Hs2)
            avg_tooth_width_body = (tooth_top_arc + tooth_bottom_arc) / 2
            V_teeth = N_slots * avg_tooth_width_body * (Hs1_m + Hs2_m) * Lstk_m * sf
            # Include tooth tip (Hs0) volume in tooth body
            tip_arc = slot_pitch_angle * Rsi_m - Bs0_m
            V_tip = N_slots * tip_arc * Hs0_m * Lstk_m * sf
            V_teeth += V_tip
            
            # If sector model, scale by sector fraction
            if sf < 1.0:
                V_teeth *= sf / (1.0 if sf == 1.0 else 1.0)
        else:
            V_teeth = 0
        
        nodes.append(ThermalNode(
            name="stator_tooth", index=idx, volume=max(V_teeth, 1e-12),
            density=rho("stator_core"), cp=cp_val("stator_core"),
            loss=cfg.loss_iron_teeth,
            loss_user_override=cfg.loss_overrides.get("stator_teeth"),
        ))
        idx_tooth = idx
        idx += 1
        
        # 4. Slot Winding (winding inside slots)
        if is_slotted:
            # Slot area (trapezoidal approximation)
            avg_slot_width = (Bs0_m + Bs2_m) / 2
            slot_area = avg_slot_width * (Hs1_m + Hs2_m)
            V_winding_slot = N_slots * slot_area * Lstk_m * geo.fill_factor * sf
        else:
            # Slotless: winding is annular band
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
        
        # 5. Airgap (separate thermal node)
        # Airgap volume = annular volume between Rsi and Rro
        V_airgap = math.pi * (Rsi_m**2 - Rro_m**2) * Lstk_m * sf
        # Air density at ~80C, cp ~ 1005 J/kgK
        rho_air = 1.05  # kg/m3 at ~80C
        cp_air = 1005   # J/kgK
        nodes.append(ThermalNode(
            name="airgap", index=idx, volume=max(V_airgap, 1e-12),
            density=rho_air, cp=cp_air,
            loss=0,
        ))
        idx_airgap = idx
        idx += 1
        
        # 7. Magnet
        pole_angle = 2 * math.pi / geo.num_poles
        magnet_arc = pole_angle * geo.magnet_span_ratio
        Rmag_outer_m = Rro_m
        Rmag_inner_m = Rro_m - t_mag_m
        
        # Magnet cross-sectional area per pole
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
        
        # 7. Rotor Core (inner to magnet inner radius)
        rotor_core_outer_m = Rro_m - t_mag_m
        V_rotor = math.pi * (rotor_core_outer_m ** 2 - Rri_m ** 2) * Lstk_m * sf
        
        nodes.append(ThermalNode(
            name="rotor_core", index=idx, volume=max(V_rotor, 1e-12),
            density=rho("rotor_core"), cp=cp_val("rotor_core"),
            loss=0,
        ))
        idx_rotor = idx
        idx += 1
        
        # 8. Shaft
        V_shaft = math.pi * Rri_m ** 2 * Lstk_m * sf
        nodes.append(ThermalNode(
            name="shaft", index=idx, volume=max(V_shaft, 1e-12),
            density=rho("shaft"), cp=cp_val("shaft"),
            loss=0,
        ))
        idx_shaft = idx
        idx += 1
        
        # ---- Build resistances ----
        resistances = []
        
        def add_R(name, fi, to, R, rtype="conduction", **kw):
            override = cfg.resistance_overrides.get(name)
            # Compute h coefficient for convection resistances
            h = kw.pop('h_coefficient', None)  # may be explicitly provided
            if h is None and rtype == "convection" and R > 0:
                area = kw.get('effective_area', 0)
                if area > 0:
                    h = 1.0 / (R * area)  # h = 1/(R*A)
            if h is not None:
                kw['h_coefficient'] = h
            resistances.append(ThermalResistance(
                name=name, node_from=fi, node_to=to,
                resistance=R, resistance_type=rtype,
                user_override=override,
                **kw
            ))
        
        # R1: Housing → Ambient
        R_housing_outer = (Rso_m + hw_m)
        A_housing = 2 * math.pi * R_housing_outer * Lstk_m * sf
        # Add end-cap area (both ends)
        A_endcaps = 2 * math.pi * R_housing_outer ** 2 * sf
        A_housing_total = A_housing + A_endcaps
        
        if cfg.cooling_mode == "TEFC":
            h_hsg = h_housing_forced(R_housing_outer, cfg.housing_air_speed or 3.0)
        elif cfg.cooling_mode == "Water Jacket":
            h_hsg = h_housing_water_jacket(
                cfg.coolant_flow_rate, 0.008, cfg.coolant_temperature
            )
        else:  # TENV
            h_hsg = h_housing_natural(R_housing_outer, Lstk_m, 70, cfg.ambient_temperature)
        
        R_housing_ambient = r_convective(A_housing_total, h_hsg)
        add_R("R_housing_ambient", idx_housing, 0, R_housing_ambient, "convection",
              effective_area=A_housing_total, h_coefficient=h_hsg)
        
        # R2: Stator Yoke → Housing
        R_yoke_housing = r_cylindrical_radial(
            Rso_m, Rso_m + hw_m, Lstk_m, k_radial("housing"), sf
        )
        add_R("R_yoke_housing", idx_yoke, idx_housing, R_yoke_housing,
              effective_length=hw_m, effective_area=2*math.pi*Rso_m*Lstk_m*sf,
              conductivity=k_radial("housing"))
        
        # R3: Stator Tooth → Stator Yoke
        if is_slotted:
            # Area-weighted: tooth-to-yoke contact at slot bottom
            # Tooth contact fraction = (slot_pitch - Bs2/Rslot_bottom) / slot_pitch
            tooth_angle = slot_pitch_angle - Bs2_m / Rslot_bottom_m
            contact_fraction = tooth_angle / slot_pitch_angle if slot_pitch_angle > 0 else 0
            sector_for_contact = sf * contact_fraction
            
            R_tooth_yoke = r_cylindrical_radial(
                Rslot_bottom_m, Rso_m, Lstk_m, k_radial("stator_core"),
                sector_fraction=sf * N_slots * contact_fraction
            )
        else:
            R_tooth_yoke = 0.01  # very low (slotless yoke is direct)
        
        add_R("R_tooth_yoke", idx_tooth, idx_yoke, R_tooth_yoke,
              effective_length=Rso_m - Rslot_bottom_m)
        
        
        # R5: Slot Winding → Stator Tooth (parallel paths through slot liner)
        if is_slotted and N_slots > 0:
            # Two tooth walls + slot bottom
            # Winding contacts left tooth wall, right tooth wall, and slot bottom
            
            # Slot width at mid-height
            R_mid_slot_m = Rsi_m + Hs0_m + (Hs1_m + Hs2_m) / 2
            w_slot_mid = (Bs1_m + Bs2_m) / 2  # average tangential width
            
            # ---- Wall path ----
            # Each slot's winding contacts two tooth walls.
            # Effective length for slab with 2-side cooling = Ws/6
            # (derived from parabolic temp profile, average vs surface temp)
            L_eff_wall = w_slot_mid / 6.0
            
            # Area per wall = Lstk × slot_body_height (Hs1+Hs2, winding contacts this)
            A_wall = Lstk_m * (Hs1_m + Hs2_m)
            
            # Get slot equivalent k
            fill = geo.fill_factor if hasattr(geo, 'fill_factor') else 0.45
            k_eq = k_slot_equivalent(fill)
            k_winding_radial = k_eq["k_radial"]
            
            # Resistance from winding average to one tooth wall
            R_one_wall = L_eff_wall / (k_winding_radial * A_wall) if A_wall > 0 else 1e6
            
            # Add slot liner resistance in series (0.3mm Nomex)
            R_liner = r_interface_slot_liner(Lstk_m, w_slot_mid, 0.0003, 0.25)
            R_per_wall = R_one_wall + R_liner
            
            # ---- Bottom path (through slot bottom to yoke) ----
            A_bottom_per_slot = Lstk_m * Bs2_m
            if A_bottom_per_slot > 0:
                # Heat flows from winding body through the slot bottom liner
                # The effective length is just the liner thickness (not slot depth!)
                # because the winding directly contacts the slot bottom
                R_liner_bottom = r_interface_slot_liner(Lstk_m, Bs2_m, 0.0003, 0.25)
                # Plus some path through the lower portion of winding body
                R_body_bottom = (Hs2_m / 2) / (k_winding_radial * A_bottom_per_slot) if Hs2_m > 0 else 0
                R_bottom_total = R_body_bottom + R_liner_bottom
            else:
                R_bottom_total = 1e6
            
            # Total winding-to-stator: two walls + bottom in parallel
            if R_per_wall > 0 and R_bottom_total > 0:
                R_winding_tooth = 1.0 / (2.0 / R_per_wall + 1.0 / R_bottom_total)
            else:
                R_winding_tooth = 1e6
            
            # Scale for sector model (each slot's R is for that slot,
            # N_slots are in parallel, and sector fraction reduces count)
            if sf < 1.0:
                R_winding_tooth *= (1.0 / (sf * N_slots))
            else:
                R_winding_tooth /= N_slots  # N_slots in parallel
        else:
            R_winding_tooth = 0.1  # slotless: winding directly contacts stator
        
        add_R("R_winding_tooth", idx_winding, idx_tooth, R_winding_tooth,
              "interface", effective_length=w_slot_mid/2 if is_slotted else 0,
              effective_area=2*A_wall if is_slotted else 0,
              conductivity=k_winding_radial if is_slotted else 1.0)
        
        # R6: Slot Winding → Stator Yoke (through slot bottom to yoke)
        if is_slotted and Bs2_m > 0 and N_slots > 0:
            A_bottom_total = Lstk_m * Bs2_m * N_slots * sf
            if A_bottom_total > 0:
                # Direct path from winding to yoke through slot bottom
                # Heat crosses through: winding body (half depth) + slot liner + yoke
                R_body = (Hs2_m / 2) / (k_winding_radial * A_bottom_total) if Hs2_m > 0 else 0
                R_liner = r_interface_slot_liner(Lstk_m, Bs2_m, 0.0003, 0.25)
                # Convert per-slot liner to total (N_slots × sf in parallel)
                if N_slots * sf > 0:
                    R_liner_total = R_liner / (N_slots * sf)
                else:
                    R_liner_total = 1e6
                R_winding_yoke_total = R_body + R_liner_total
            else:
                R_winding_yoke_total = 1e6
        else:
            R_winding_yoke_total = 0.1
        
        add_R("R_winding_yoke", idx_winding, idx_yoke, R_winding_yoke_total,
              effective_length=Hs0_m+Hs1_m+Hs2_m if is_slotted else 0)
        
        
                        # R8: Stator Tooth -> Airgap (stationary side convection, tooth face at bore)
        if is_slotted:
            # Tooth face area at bore = (slot pitch - slot opening) * stack length * N_slots * sector
            tooth_arc_at_bore = slot_pitch_angle * Rsi_m - Bs0_m
            A_tooth_bore = tooth_arc_at_bore * Lstk_m * N_slots * sf
        else:
            A_tooth_bore = 0  # slotless: no tooth
        if A_tooth_bore > 0:
            h_ag_tooth = h_airgap_natural(Rsi_m, gap_m, Lstk_m, 0)  # stationary side
            R_tooth_airgap = r_convective(A_tooth_bore, h_ag_tooth)
            add_R("R_tooth_airgap", idx_tooth, idx_airgap, R_tooth_airgap, "convection",
                  effective_area=A_tooth_bore, h_coefficient=h_ag_tooth)
        
        # R9: Slot Winding -> Airgap (through slot opening, stationary side)
        if is_slotted:
            # Slot opening area at bore
            A_winding_bore = Bs0_m * Lstk_m * N_slots * sf
        else:
            # Slotless: winding surface is the entire bore area
            A_winding_bore = 2 * math.pi * Rsi_m * Lstk_m * sf
        if A_winding_bore > 0:
            h_ag_winding = h_airgap_natural(Rsi_m, gap_m, Lstk_m, 0)  # stationary side
            R_winding_airgap = r_convective(A_winding_bore, h_ag_winding)
            add_R("R_winding_airgap", idx_winding, idx_airgap, R_winding_airgap, "convection",
                  effective_area=A_winding_bore, h_coefficient=h_ag_winding)
# R9: Airgap -> Magnet (rotor side, rotating convection)
        # Rotating surface uses Taylor number correlation at full speed
        h_ag_rotor = h_airgap_natural(Rro_m, gap_m, Lstk_m, cfg.speed_rpm)
        A_airgap_rotor = 2 * math.pi * Rro_m * Lstk_m * sf
        R_airgap_magnet = r_convective(A_airgap_rotor, h_ag_rotor)
        add_R("R_airgap_magnet", idx_airgap, idx_magnet, R_airgap_magnet, "convection",
              effective_area=A_airgap_rotor, h_coefficient=h_ag_rotor)
        
# R9: Rotor Core → Magnet
        R_rotor_magnet = r_cylindrical_radial(
            Rmag_inner_m, Rmag_outer_m, Lstk_m, k_radial("magnet"), sf
        )
        add_R("R_rotor_magnet", idx_rotor, idx_magnet, R_rotor_magnet,
              effective_length=t_mag_m,
              effective_area=2*math.pi*Rmag_inner_m*Lstk_m*sf,
              conductivity=k_radial("magnet"))
        
        # R12: Shaft → Rotor Core
        R_shaft_rotor = r_cylindrical_radial(
            Rshaft_m, Rri_m, Lstk_m, k_radial("shaft"), sf
        )
        add_R("R_shaft_rotor", idx_shaft, idx_rotor, R_shaft_rotor,
              effective_length=Rri_m - Rshaft_m,
              effective_area=2*math.pi*Rshaft_m*Lstk_m*sf,
              conductivity=k_radial("shaft"))
        
        # R12: Rotor Core -> Shaft (alternate path)
        if Rri_m > Rshaft_m:
            R_rotor_shaft_inner = r_cylindrical_radial(
                Rshaft_m, Rri_m, Lstk_m, k_radial("rotor_core"), sf
            )
        else:
            R_rotor_shaft_inner = 0.001
        add_R("R_rotor_shaft_inner", idx_rotor, idx_shaft, R_rotor_shaft_inner)
        
        # ---- Create network ----
        net = ThermalNetwork(nodes, resistances, f"2D LPTN ({cfg.cooling_mode})")
        return net
    
    def build_3d(self) -> ThermalNetwork:
        """Build a 3D network with end-winding node (full multi-slice coming)."""
        net = self.build_2d()
        cfg = self.config
        geo = self.geo
        m = self._mm_to_m
        Lstk_m = m(geo.stack_length)

        winding_node = net.get_node_by_name("slot_winding")
        V_end = winding_node.volume * 0.3 if winding_node else 1e-6
        
        # Find max existing index
        max_idx = max(n.index for n in net.nodes)
        
        from lpt_fe.node import ThermalNode
        end_winding = ThermalNode(
            name="end_winding",
            index=max_idx + 1,
            volume=V_end,
            density=8900, cp=385,
            loss=cfg.loss_copper_end,
            loss_temperature_dependent=True,
            loss_ref_temp=20.0,
            loss_alpha=0.00393,
            loss_user_override=cfg.loss_overrides.get("end_winding"),
        )
        net.nodes.append(end_winding)
        
        # Connect end winding to slot winding (axial conduction through copper)
        from lpt_fe.correlations import r_cylindrical_axial
        winding_node = net.get_node_by_name("slot_winding")
        if winding_node:
            # Cross-sectional area of copper in slots
            A_cu = winding_node.volume / Lstk_m if hasattr(self, '_mm_to_m') else 0
            L_end = 0.03  # 30mm estimated end winding length
            A_cu = max(A_cu, 1e-6)
            R_end_axial = r_cylindrical_axial(L_end, A_cu, 385)
            
            net.resistances.append(ThermalResistance(
                name="R_endwinding_slot",
                node_from=end_winding.index,
                node_to=winding_node.index,
                resistance=R_end_axial,
                resistance_type="conduction",
                effective_length=L_end,
                effective_area=A_cu,
                conductivity=385,
            ))
        
        net.name = f"3D LPTN ({cfg.cooling_mode}, {cfg.n_axial_slices} slices)"
        return net


def build_thermal_network(geo, config: NetworkBuilderConfig = None) -> ThermalNetwork:
    """
    Convenience function: build a ThermalNetwork from a MotorGeometry.
    
    Parameters:
        geo: MotorGeometry instance
        config: NetworkBuilderConfig (or None for defaults)
    
    Returns:
        ThermalNetwork ready for solving
    """
    mapper = GeometryMapper(geo, config)
    if config and config.dimensionality == "3D":
        return mapper.build_3d()
    else:
        return mapper.build_2d()
