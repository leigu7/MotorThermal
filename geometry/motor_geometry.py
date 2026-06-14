"""
Parametric 2D radial cross-section geometry for radial-flux motor.
Supports:
  - Slotless (concentric rings)
  - Slotted (distributed/concentrated windings)

All dimensions in mm.
Angle convention: 0 = 3 o'clock, positive CCW.
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Optional
import math
import numpy as np


@dataclass
class MotorGeometryParams:
    """All geometric parameters for a radial-flux motor cross-section."""

    # === Structure ===
    structure_type: str = "Slotted (distributed)"  # "Slotless", "Slotted (distributed)", "Slotted (concentrated)"

    # === Stator ===
    Rso: float = 50.0       # Stator outer radius [mm]
    Rsi: float = 30.0       # Stator inner radius (bore) [mm]
    stack_length: float = 60.0  # Axial stack length [mm]

    # === Slots (only for slotted types) ===
    num_slots: int = 24
    slot_depth: float = 10.0   # Radial depth of slot [mm]
    slot_opening: float = 2.0  # Slot opening width at airgap [mm]
    tooth_width_min: float = 3.0  # Minimum tooth width (at airgap side) [mm]
    slot_wedge_height: float = 0.5  # Wedge / tooth-tip height [mm]

    # === Winding ===
    winding_layers: int = 2  # 1 or 2 layers per slot
    conductor_diameter: float = 1.0  # Bare wire diameter [mm]
    turns_per_slot: int = 20
    fill_factor: float = 0.45  # Overall slot fill factor
    insulation_class: str = "Class H (180°C)"

    # === Rotor ===
    Rro: float = 29.0       # Rotor outer radius [mm]
    Rri: float = 10.0       # Rotor inner radius (shaft) [mm]

    # === Magnet ===
    num_poles: int = 8
    magnet_thickness: float = 4.0  # Radial thickness [mm]
    magnet_span_ratio: float = 0.82  # Pole arc / pole pitch ratio (0-1)
    magnet_grade: str = "N35SH"  # NdFeB grade
    magnet_max_temp: float = 150.0  # Max operating temp [°C]

    # === Airgap ===
    airgap_length: float = 1.0  # Mechanical airgap [mm]

    # === Housing ===
    housing_wall_thickness: float = 5.0  # Housing wall thickness [mm]
    housing_finned: bool = False

    # === Shaft ===
    shaft_radius: float = 10.0  # Shaft radius (=Rri) [mm]

    # === Derived quantities (computed) ===
    _computed: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Validate and compute derived quantities."""
        self._validate()
        self._compute_derived()

    def _validate(self):
        """Basic consistency checks."""
        if self.Rsi >= self.Rso:
            raise ValueError(f"Stator inner radius Rsi={self.Rsi} must be < Rso={self.Rso}")
        if self.Rro >= self.Rsi:
            raise ValueError(f"Rotor outer radius Rro={self.Rro} must be < stator inner radius Rsi={self.Rsi}")
        if self.Rri >= self.Rro:
            raise ValueError(f"Rotor inner radius Rri={self.Rri} must be < Rro={self.Rro}")
        if self.airgap_length <= 0:
            raise ValueError("Airgap must be positive")
        expected_airgap = self.Rsi - self.Rro
        if abs(expected_airgap - self.airgap_length) > 0.01:




            # Auto-correct: airgap must equal Rsi - Rro
            self.airgap_length = round(expected_airgap, 3)
        if self.magnet_thickness > self.Rro - self.Rri:
            raise ValueError("Magnet thickness exceeds rotor back-iron depth")
        if self.num_poles <= 0:
            raise ValueError("Pole count must be positive")
        is_slotted = "Slotted" in self.structure_type
        if is_slotted and self.num_slots <= 0:
            raise ValueError("Pole and slot counts must be positive")
        if self.structure_type not in ["Slotless", "Slotted (distributed)", "Slotted (concentrated)"]:
            raise ValueError(f"Unknown structure type: {self.structure_type}")

    def _compute_derived(self):
        """Compute derived geometric quantities."""
        # Stator back-iron thickness
        stator_yoke_thickness = self.Rso - self.Rsi

        # Rotor back-iron thickness
        rotor_back_iron = self.Rro - self.magnet_thickness if self.magnet_thickness > 0 else self.Rro - self.Rri

        # Pole pitch at airgap (mechanical angle)
        pole_pitch_angle = 360.0 / self.num_poles
        magnet_arc_angle = pole_pitch_angle * self.magnet_span_ratio

        # Slot pitch angle
        is_slotted = "Slotted" in self.structure_type
        slot_pitch_angle = 360.0 / self.num_slots if is_slotted and self.num_slots > 0 else 0

        # Tooth geometry (at airgap side)
        Rtooth_tip = self.Rsi  # tooth tip at stator bore
        Rslot_bottom = self.Rsi + self.slot_depth  # slot bottom radius

        # Slot area estimation (trapezoidal approximation)
        tooth_tip_width = self.tooth_width_min
        # Width at slot bottom
        theta_tooth_tip = tooth_tip_width / self.Rsi  # rad
        tooth_bottom_width = theta_tooth_tip * Rslot_bottom
        # Slot area (trapezoid approximation)
        slot_avg_width = (slot_pitch_angle * math.pi / 180) * (self.Rsi + Rslot_bottom) / 2 - (tooth_tip_width + tooth_bottom_width) / 2
        slot_area = slot_avg_width * self.slot_depth * 0.9  # 0.9 for corner fillets approx

        # Conductor area
        conductor_area = math.pi * (self.conductor_diameter / 2) ** 2
        total_conductor_area = self.turns_per_slot * conductor_area

        self._computed = {
            "pole_pitch_angle_deg": pole_pitch_angle,
            "magnet_arc_angle_deg": magnet_arc_angle,
            "slot_pitch_angle_deg": slot_pitch_angle,
            "stator_yoke_thickness": stator_yoke_thickness,
            "rotor_back_iron": rotor_back_iron,
            "slot_area_mm2": slot_area,
            "conductor_area_mm2": conductor_area,
            "total_conductor_area_mm2": total_conductor_area,
            "effective_slot_fill": total_conductor_area / slot_area if slot_area > 0 else 0,
            "tooth_bottom_width": tooth_bottom_width,
            "Rslot_bottom": Rslot_bottom,
            "magnet_arc_angle_rad": magnet_arc_angle * math.pi / 180,
        }

    @property
    def computed(self):
        return self._computed

    def update(self, **kwargs):
        """Update parameters and recompute."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.__post_init__()

    def as_dict(self):
        """Return a flat dict of all parameters (excluding _computed)."""
        d = asdict(self)
        d.pop("_computed", None)
        return d

    def to_json(self) -> str:
        """Serialize to JSON string for file export."""
        import json
        return json.dumps(self.as_dict(), indent=2)

    @staticmethod
    def from_json(json_str: str) -> "MotorGeometryParams":
        """Deserialize from JSON string."""
        import json
        data = json.loads(json_str)
        return MotorGeometryParams(**data)


# ============================================================================
# Geometry Generation (returns polygon vertices for plotting)
# ============================================================================

def generate_cross_section(geo: MotorGeometryParams, num_radial_segments: int = 64):
    """
    Generate 2D cross-section polygons for the motor.

    Returns a dict of component → list of (r, theta) or (x, y) vertices.
    """
    comps = {}
    theta = np.linspace(0, 2 * math.pi, num_radial_segments, endpoint=False)

    def circle_points(R, N=num_radial_segments):
        """Generate (x,y) points along a circle of radius R."""
        t = np.linspace(0, 2 * math.pi, N, endpoint=False)
        return np.column_stack([R * np.cos(t), R * np.sin(t)])

    def arc_points(R, theta_start, theta_end, N=32):
        """Generate (x,y) points along an arc."""
        t = np.linspace(theta_start, theta_end, N)
        return np.column_stack([R * np.cos(t), R * np.sin(t)])

    # === Shaft (filled circle) ===
    comps["shaft"] = circle_points(geo.shaft_radius)

    # === Rotor core (annulus between Rri and Rro-magnet_thickness) ===
    comps["rotor_core"] = circle_points(geo.Rro - geo.magnet_thickness)

    # === Magnet segments ===
    magnets = []
    pole_angle = 2 * math.pi / geo.num_poles
    magnet_arc = pole_angle * geo.magnet_span_ratio
    for p in range(geo.num_poles):
        theta_center = p * pole_angle
        theta_start = theta_center - magnet_arc / 2
        theta_end = theta_center + magnet_arc / 2
        # Outer arc (at rotor surface Rro)
        outer_arc = arc_points(geo.Rro, theta_start, theta_end)
        # Inner arc (at magnet inner radius)
        Rmag_inner = geo.Rro - geo.magnet_thickness
        inner_arc = arc_points(Rmag_inner, theta_end, theta_start)  # reversed
        magnet_poly = np.vstack([outer_arc, inner_arc])
        magnets.append(magnet_poly)
    comps["magnets"] = magnets

    # === Airgap (just a circle for visualization) ===
    comps["airgap_outer"] = circle_points(geo.Rsi)  # stator bore
    comps["airgap_inner"] = circle_points(geo.Rro)  # rotor surface

    # === Stator (with or without slots) ===
    if geo.structure_type == "Slotless":
        comps["stator_core"] = circle_points(geo.Rso)
        comps["winding"] = []  # airgap winding - just show airgap region
    else:
        # For slotted: generate stator core outline with teeth and slots
        stator_polys = _generate_slotted_stator(geo)
        comps["stator_core"] = stator_polys["stator_yoke"]
        comps["stator_teeth"] = stator_polys["teeth"]
        comps["slot_windings"] = stator_polys["slot_regions"]

    # === Housing ===
    comps["housing"] = circle_points(geo.Rso + geo.housing_wall_thickness)

    return comps


def _generate_slotted_stator(geo: MotorGeometryParams):
    """
    Generate polygons for slotted stator: yoke, teeth, slot windings.
    """
    results = {"stator_yoke": None, "teeth": [], "slot_regions": []}

    N = 4  # points per arc segment
    slot_pitch = 2 * math.pi / geo.num_slots
    Rsi = geo.Rsi
    Rso = geo.Rso
    Rslot_bottom = geo._computed["Rslot_bottom"]
    slot_opening_half = geo.slot_opening / 2 / Rsi  # half-angle of slot opening

    # Tooth tip at stator bore
    tooth_tip_half_width = geo.tooth_width_min / 2 / Rsi  # half-angle

    # Yoke outer circle
    yoke_outer = _circle_points(Rso, 128)
    results["stator_yoke"] = yoke_outer

    for s in range(geo.num_slots):
        theta_center = s * slot_pitch
        theta_slot_start = theta_center - slot_pitch / 2
        theta_slot_end = theta_center + slot_pitch / 2

        # Tooth geometry
        # Each tooth is between two adjacent slots
        tooth_center = theta_center + slot_pitch / 2
        t_start = tooth_center - tooth_tip_half_width
        t_end = tooth_center + tooth_tip_half_width

        # Build tooth polygon (from bore outward to slot bottom, then back)
        # Tooth tip at Rsi
        tip_pts = _arc_points(Rsi, t_start, t_end, N)
        # Side flanks going radially inward (toward slot bottom)
        # Actually teeth go from Rsi outward, so:
        # Tooth goes from Rsi (bore) radially outward to Rslot_bottom
        # At Rsi: width = tooth_width_min
        # At Rslot_bottom: wider (or same if parallel)
        tooth_bottom_half = geo._computed["tooth_bottom_width"] / 2 / Rslot_bottom

        # Left flank from Rsi to Rslot_bottom
        flank_outer_left = _arc_points(Rslot_bottom, t_start, t_start, 2)  # single point
        flank_outer_right = _arc_points(Rslot_bottom, t_end, t_end, 2)

        # Build tooth polygon
        tooth_poly = np.vstack([
            _arc_points(Rsi, t_start, t_end, N),
            _arc_points(Rslot_bottom, t_end, t_start, N)
        ])
        results["teeth"].append(tooth_poly)

        # Slot region (between two teeth)
        next_tooth_center = tooth_center + slot_pitch
        next_t_start = next_tooth_center - tooth_tip_half_width

        slot_poly = np.vstack([
            _arc_points(Rsi, t_end, next_t_start, N),
            _arc_points(Rslot_bottom, next_t_start, t_end, N)
        ])
        results["slot_regions"].append(slot_poly)

    return results


def _circle_points(R, N=64):
    """Generate (x,y) points forming a circle of radius R."""
    t = np.linspace(0, 2 * math.pi, N, endpoint=False)
    return np.column_stack([R * np.cos(t), R * np.sin(t)])


def _arc_points(R, theta_start, theta_end, N=32):
    """Generate (x,y) points along a circular arc."""
    # Handle wrap-around
    if theta_end < theta_start:
        theta_end += 2 * math.pi
    t = np.linspace(theta_start, theta_end, N)
    return np.column_stack([R * np.cos(t), R * np.sin(t)])


# ============================================================================
# Mesh export helpers (for Gmsh path)
# ============================================================================

def get_region_bounds(geo: MotorGeometryParams):
    """
    Return dict of region definitions for mesh generation.
    Each region: (r_min, r_max, theta_start, theta_end) or full annular.
    """
    pole_angle = 2 * math.pi / geo.num_poles
    magnet_arc = pole_angle * geo.magnet_span_ratio

    regions = {
        "shaft": (0, geo.shaft_radius, 0, 2 * math.pi),
        "rotor_core": (geo.shaft_radius, geo.Rro - geo.magnet_thickness, 0, 2 * math.pi),
        "magnet": (geo.Rro - geo.magnet_thickness, geo.Rro, 0, 2 * math.pi),
        "airgap": (geo.Rro, geo.Rsi, 0, 2 * math.pi),
        "stator": (geo.Rsi, geo.Rso, 0, 2 * math.pi),
        "housing": (geo.Rso, geo.Rso + geo.housing_wall_thickness, 0, 2 * math.pi),
    }
    return regions
