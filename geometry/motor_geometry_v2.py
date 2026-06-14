"""
Parametric 2D radial cross-section geometry for radial-flux motor.
Version 2 - Uses proper slot geometry definitions (JMAG/Maxwell inspired).

Two structure types:
  1. "Slotless": Concentric rings with winding region defined by inner/outer radii
  2. "Slotted": Slots defined by SlotGeometry parameters

All dimensions in mm.
Angle convention: 0 = 3 o'clock, positive CCW (matplotlib standard).
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Optional, Dict
import math
import numpy as np

from geometry.slot_geometry import SlotGeometry, slot_from_simple_params


@dataclass
class MotorGeometry:
    """
    Complete parametric motor geometry for radial-flux motors.

    Structure options:
      - "Slotless": Airgap winding between stator bore and winding inner radius
      - "Slotted": Conventional slotted stator with windings in slots
    """

    # === Structure ===
    structure_type: str = "Slotted"  # "Slotless" or "Slotted"

    # === Core dimensions ===
    Rso: float = 50.0          # Stator outer radius [mm]
    Rsi: float = 30.0          # Stator inner radius (bore) [mm]
    stack_length: float = 60.0  # Axial stack length [mm]

    # === Slotless-only: Winding region ===
    # For slotless: the winding sits in the airgap region
    winding_inner_radius: float = 30.0   # Inner radius of winding band [mm] (for slotless)
    winding_outer_radius: float = 34.0   # Outer radius of winding band [mm] (for slotless)

    # === Slotted-only: Slot definition ===
    num_slots: int = 24
    slot: SlotGeometry = field(default_factory=SlotGeometry)

    # === Winding (used by both types) ===
    winding_layers: int = 2
    conductor_diameter: float = 1.0      # Bare wire diameter [mm]
    turns_per_slot: int = 20
    fill_factor: float = 0.45            # Target fill factor
    insulation_class: str = "Class H (180°C)"

    # === Rotor ===
    Rro: float = 29.0          # Rotor outer radius [mm]
    Rri: float = 10.0          # Rotor inner radius [mm]

    # === Magnet ===
    num_poles: int = 8
    magnet_thickness: float = 4.0        # Radial thickness [mm]
    magnet_span_ratio: float = 0.82      # Pole arc / pole pitch (0-1)
    magnet_grade: str = "N35SH"
    magnet_max_temp: float = 150.0       # Max operating temp [°C]

    # === Airgap ===
    airgap_length: float = 1.0           # Mechanical airgap [mm]

    # === Housing ===
    housing_wall_thickness: float = 5.0  # Housing wall thickness [mm]
    housing_finned: bool = False

    # === Shaft ===
    shaft_radius: float = 10.0           # Shaft radius (=Rri typically) [mm]

    # === Derived computed values ===
    _computed: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Validate and compute derived quantities."""
        self._validate()
        self._compute_derived()

    def _validate(self):
        """Basic consistency checks."""
        if self.Rsi >= self.Rso:
            raise ValueError(f"Stator inner radius Rsi={self.Rsi} must be < Rso={self.Rso}")

        if self.structure_type == "Slotted":
            if self.num_slots <= 0:
                raise ValueError("Number of slots must be positive for Slotted type")
            # Check slot fits within stator yoke
            slot_bottom_r = self.slot.get_slot_bottom_r(self.Rsi)
            if slot_bottom_r > self.Rso:
                raise ValueError(
                    f"Slot bottom radius {slot_bottom_r:.1f} exceeds "
                    f"stator outer radius {self.Rso:.1f}"
                )
        else:  # Slotless
            if self.winding_outer_radius > self.Rso:
                raise ValueError(
                    f"Winding outer radius {self.winding_outer_radius} "
                    f"exceeds stator OD {self.Rso}"
                )
            if self.winding_inner_radius < self.Rsi:
                # In slotless motor, winding is in the gap; Rsi is typically
                # at the winding inner surface. Adjust if needed.
                pass

        if self.Rro >= self.Rsi:
            raise ValueError(f"Rotor OD {self.Rro} must be < stator bore {self.Rsi}")
        if self.Rri >= self.Rro:
            raise ValueError(f"Rotor ID {self.Rri} must be < rotor OD {self.Rro}")
        if self.airgap_length <= 0:
            raise ValueError("Airgap must be positive")

        expected_airgap = self.Rsi - self.Rro
        if abs(expected_airgap - self.airgap_length) > 0.01:
            raise ValueError(
                f"Airgap {self.airgap_length} mm doesn't match "
                f"Rsi - Rro = {expected_airgap:.2f} mm"
            )

        if self.magnet_thickness > self.Rro - self.Rri:
            raise ValueError("Magnet thickness exceeds rotor back-iron depth")
        if self.num_poles <= 0:
            raise ValueError("Pole count must be positive")

        if self.structure_type not in ["Slotless", "Slotted"]:
            raise ValueError(f"Unknown structure type: {self.structure_type}")

    def _compute_derived(self):
        """Compute derived geometric quantities."""
        is_slotted = self.structure_type == "Slotted"

        # Stator yoke thickness
        if is_slotted:
            slot_bottom_r = self.slot.get_slot_bottom_r(self.Rsi)
            stator_yoke_thickness = self.Rso - slot_bottom_r
        else:
            # For slotless: stator yoke from winding OD to stator OD
            stator_yoke_thickness = self.Rso - self.winding_outer_radius

        # Rotor back-iron
        rotor_back_iron = self.Rro - self.magnet_thickness

        # Pole pitch
        pole_pitch_angle = 360.0 / self.num_poles
        magnet_arc_angle = pole_pitch_angle * self.magnet_span_ratio

        # Slot pitch
        if is_slotted:
            slot_pitch_angle = 360.0 / self.num_slots
        else:
            slot_pitch_angle = 0.0

        # Slot area
        if is_slotted:
            slot_area = self._compute_slot_area()
        else:
            # For slotless: winding area = π*(Rwo² - Rwi²)*fill_factor
            slot_area = (math.pi * (self.winding_outer_radius**2 - self.winding_inner_radius**2)
                         * self.fill_factor / self.num_poles)  # rough estimate

        # Conductor area
        conductor_area = math.pi * (self.conductor_diameter / 2) ** 2
        total_conductor_area = self.turns_per_slot * conductor_area

        self._computed = {
            "is_slotted": is_slotted,
            "slot_bottom_radius": self.slot.get_slot_bottom_r(self.Rsi) if is_slotted else self.winding_outer_radius,
            "stator_yoke_thickness": stator_yoke_thickness,
            "rotor_back_iron": rotor_back_iron,
            "pole_pitch_angle_deg": pole_pitch_angle,
            "magnet_arc_angle_deg": magnet_arc_angle,
            "magnet_arc_angle_rad": magnet_arc_angle * math.pi / 180,
            "slot_pitch_angle_deg": slot_pitch_angle,
            "slot_area_mm2": slot_area,
            "conductor_area_mm2": conductor_area,
            "total_conductor_area_mm2": total_conductor_area,
            "effective_slot_fill": total_conductor_area / slot_area if slot_area > 0 else 0,
        }

    def _compute_slot_area(self) -> float:
        """
        Compute slot cross-sectional area using trapezoidal integration.
        More accurate than simple rectangle.
        """
        s = self.slot
        Rsi = self.Rsi
        N = 20  # integration steps

        r0 = Rsi + s.Hs0               # start of shoulder
        r1 = r0 + s.Hs1                # start of slot body
        r2 = r1 + s.Hs2                # slot bottom

        def slot_width_at(r):
            """Interpolate slot width at radius r."""
            if r <= Rsi + s.Hs0:
                return s.Bs0
            elif r <= r1:
                frac = (r - r0) / (r1 - r0) if r1 > r0 else 0
                return s.Bs0 + (s.Bs1 - s.Bs0) * frac
            elif r <= r2:
                frac = (r - r1) / (r2 - r1) if r2 > r1 else 0
                return s.Bs1 + (s.Bs2 - s.Bs1) * frac
            return s.Bs2

        # Numerical integration using trapezoidal rule
        dr = (r2 - r0) / N
        area = 0.0
        for i in range(N):
            r_left = r0 + i * dr
            r_right = r_left + dr
            w_left = slot_width_at(r_left)
            w_right = slot_width_at(r_right)
            area += (w_left + w_right) / 2 * dr

        return area

    @property
    def computed(self) -> Dict:
        return self._computed

    def update(self, **kwargs):
        """Update parameters and recompute."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.__post_init__()

    def as_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# 2D Cross-Section Generation for Plotting
# ============================================================================

def generate_cross_section_polygons(geo: MotorGeometry, num_segments: int = 256) -> Dict:
    """
    Generate 2D polygon vertices for all motor components.

    Returns dict with keys:
      - 'shaft': array of (x,y) vertices
      - 'rotor_core': array of (x,y) vertices
      - 'magnets': list of arrays (one per pole)
      - 'airgap_outer', 'airgap_inner': boundary circles
      - 'stator_yoke': array of (x,y) vertices
      - 'stator_teeth': list of arrays (one per tooth, slotted only)
      - 'slot_windings': list of arrays (one per slot, slotted only)
      - 'winding_band': array (slotless only)
      - 'housing': array of (x,y) vertices
    """
    comps = {}
    is_slotted = geo.structure_type == "Slotted"

    def circle(R, N=num_segments):
        t = np.linspace(0, 2 * math.pi, N, endpoint=False)
        return np.column_stack([R * np.cos(t), R * np.sin(t)])

    def arc(R, t_start, t_end, N=32):
        if t_end < t_start:
            t_end += 2 * math.pi
        t = np.linspace(t_start, t_end, N)
        return np.column_stack([R * np.cos(t), R * np.sin(t)])

    # === Shaft ===
    comps["shaft"] = circle(geo.shaft_radius)

    # === Rotor core ===
    comps["rotor_core"] = circle(geo.Rro - geo.magnet_thickness)

    # === Magnets (segmented) ===
    magnets = []
    pole_angle = 2 * math.pi / geo.num_poles
    magnet_arc = pole_angle * geo.magnet_span_ratio
    Rmag_inner = geo.Rro - geo.magnet_thickness

    for p in range(geo.num_poles):
        theta_center = p * pole_angle
        t_start = theta_center - magnet_arc / 2
        t_end = theta_center + magnet_arc / 2
        outer = arc(geo.Rro, t_start, t_end, 32)
        inner = arc(Rmag_inner, t_end, t_start, 32)  # reversed
        magnets.append(np.vstack([outer, inner]))
    comps["magnets"] = magnets

    # === Airgap boundaries ===
    comps["airgap_outer"] = circle(geo.Rsi)
    comps["airgap_inner"] = circle(geo.Rro)

    # === Stator ===
    if is_slotted:
        stator_polys = _generate_slotted_stator_polygons(geo)
        comps["stator_yoke"] = stator_polys["stator_yoke"]
        comps["stator_teeth"] = stator_polys["teeth"]
        comps["slot_windings"] = stator_polys["slot_regions"]
    else:
        # Slotless: yoke + winding band
        comps["stator_yoke"] = circle(geo.Rso)
        # Winding band (annular ring)
        Rwi = geo.winding_inner_radius
        Rwo = geo.winding_outer_radius
        comps["winding_band"] = circle(Rwo)  # outer boundary shown
        comps["winding_bore"] = circle(Rwi)  # inner boundary

        # For the stator yoke in slotless, we need to show the inner bore
        # which is at the winding outer radius
        comps["stator_bore"] = circle(Rwi)

    # === Housing ===
    comps["housing"] = circle(geo.Rso + geo.housing_wall_thickness)

    return comps


def _generate_slotted_stator_polygons(geo: MotorGeometry) -> Dict:
    """
    Generate detailed polygons for slotted stator.
    Uses proper slot geometry (Hs0, Hs1, Hs2, Bs0, Bs1, Bs2).
    """
    results = {"stator_yoke": None, "teeth": [], "slot_regions": []}

    s = geo.slot
    Rsi = geo.Rsi
    Rso = geo.Rso
    N = geo.num_slots

    slot_pitch = 2 * math.pi / N
    Rtooth_tip = Rsi
    Rshoulder_bottom = Rsi + s.Hs0 + s.Hs1  # wedge bottom
    Rslot_bottom = s.get_slot_bottom_r(Rsi)

    # Yoke is the outer ring
    results["stator_yoke"] = _circle_points(Rso, 128)

    # For each slot position, compute tooth and slot polygons
    for i in range(N):
        theta_c = i * slot_pitch  # slot center

        # === Tooth polygon (between this slot and next) ===
        # Tooth center is between slots
        tooth_center = theta_c + slot_pitch / 2

        # At bore: tooth spans from (slot_end_of_prev) to (slot_start_of_current)
        # Slot opening at theta_c has width Bs0
        slot_half_angle_0 = s.Bs0 / 2 / Rsi  # half-angle at bore opening
        slot_half_angle_1 = s.Bs1 / 2 / Rsi  # half-angle at shoulder
        slot_half_angle_2 = s.Bs2 / 2 / Rslot_bottom  # half-angle at bottom

        # Slot boundaries
        slot_theta_start = theta_c - slot_half_angle_0  # at bore opening
        slot_theta_end = theta_c + slot_half_angle_0

        # Shoulder boundaries (wider)
        shoulder_theta_start = theta_c - slot_half_angle_1
        shoulder_theta_end = theta_c + slot_half_angle_1

        # Slot bottom boundaries
        bottom_theta_start = theta_c - slot_half_angle_2
        bottom_theta_end = theta_c + slot_half_angle_2

        # Build slot region polygon
        # Slot region: from opening at Rsi down to Rslot_bottom
        # Points go around the slot boundary:
        #   opening_left (Rsi) → shoulder_left (Rshoulder) → bottom_left (Rslot_bottom)
        #   → bottom_right (Rslot_bottom) → shoulder_right → opening_right (Rsi) → close

        N_pts = 12  # points per curved edge

        # Left edge (going radially outward)
        left_edge = np.array([
            [Rsi * math.cos(slot_theta_start), Rsi * math.sin(slot_theta_start)],
            [Rshoulder_bottom * math.cos(shoulder_theta_start),
             Rshoulder_bottom * math.sin(shoulder_theta_start)],
            [Rslot_bottom * math.cos(bottom_theta_start),
             Rslot_bottom * math.sin(bottom_theta_start)],
        ])

        # Bottom arc (rightward)
        bottom_arc = arc(Rslot_bottom,
                         bottom_theta_start,
                         bottom_theta_end, N_pts)

        # Right edge (going radially inward)
        right_edge = np.array([
            [Rslot_bottom * math.cos(bottom_theta_end),
             Rslot_bottom * math.sin(bottom_theta_end)],
            [Rshoulder_bottom * math.cos(shoulder_theta_end),
             Rshoulder_bottom * math.sin(shoulder_theta_end)],
            [Rsi * math.cos(slot_theta_end),
             Rsi * math.sin(slot_theta_end)],
        ])

        # Top arc (opening - going leftward)
        top_arc = arc(Rsi, slot_theta_end, slot_theta_start, N_pts)

        slot_poly = np.vstack([left_edge, bottom_arc, right_edge, top_arc])
        results["slot_regions"].append(slot_poly)

        # === Tooth polygon ===
        # Tooth spans from end of this slot to start of next slot
        # At bore: between theta_c + Bs0_half and next_theta_c - Bs0_half
        next_theta_c = (i + 1) * slot_pitch
        next_bottom_theta_start = next_theta_c - slot_half_angle_2
        next_shoulder_theta_start = next_theta_c - slot_half_angle_1
        next_slot_theta_start = next_theta_c - slot_half_angle_0

        # Tooth left side = slot right side (already computed)
        # Tooth goes from: slot_right_edge → outward to shoulder → to bottom
        # → along bottom to next slot bottom_left → inward → back to bore

        tooth_left = np.array([
            [Rsi * math.cos(slot_theta_end), Rsi * math.sin(slot_theta_end)],
            [Rshoulder_bottom * math.cos(shoulder_theta_end),
             Rshoulder_bottom * math.sin(shoulder_theta_end)],
            [Rslot_bottom * math.cos(bottom_theta_end),
             Rslot_bottom * math.sin(bottom_theta_end)],
        ])

        # Bottom of tooth (between slots)
        tooth_bottom = arc(Rslot_bottom,
                           bottom_theta_end,
                           next_bottom_theta_start, N_pts)

        tooth_right = np.array([
            [Rslot_bottom * math.cos(next_bottom_theta_start),
             Rslot_bottom * math.sin(next_bottom_theta_start)],
            [Rshoulder_bottom * math.cos(next_shoulder_theta_start),
             Rshoulder_bottom * math.sin(next_shoulder_theta_start)],
            [Rsi * math.cos(next_slot_theta_start),
             Rsi * math.sin(next_slot_theta_start)],
        ])

        tooth_top = arc(Rsi, next_slot_theta_start, slot_theta_end, N_pts)

        tooth_poly = np.vstack([tooth_left, tooth_bottom, tooth_right, tooth_top])
        results["teeth"].append(tooth_poly)

    return results


# ============================================================================
# Helper functions
# ============================================================================

def _circle_points(R, N=64):
    t = np.linspace(0, 2 * math.pi, N, endpoint=False)
    return np.column_stack([R * np.cos(t), R * np.sin(t)])


def arc(R, t_start, t_end, N=32):
    """Generate (x,y) points along a circular arc."""
    if t_end < t_start:
        t_end += 2 * math.pi
    t = np.linspace(t_start, t_end, N)
    return np.column_stack([R * np.cos(t), R * np.sin(t)])


# ============================================================================
# Utility
# ============================================================================

def get_geometry_summary(geo: MotorGeometry) -> Dict:
    """Return a dict of human-readable geometry info."""
    c = geo.computed
    info = {
        "Motor Type": geo.structure_type,
        "Stator OD": f"{geo.Rso:.1f} mm",
        "Stator Bore ID": f"{geo.Rsi:.1f} mm",
        "Stack Length": f"{geo.stack_length:.1f} mm",
        "Stator Yoke Thickness": f"{c['stator_yoke_thickness']:.2f} mm",
        "Rotor OD": f"{geo.Rro:.1f} mm",
        "Rotor ID": f"{geo.Rri:.1f} mm",
        "Rotor Back-iron": f"{c['rotor_back_iron']:.2f} mm",
        "Airgap": f"{geo.airgap_length:.2f} mm",
    }

    if geo.structure_type == "Slotted":
        s = geo.slot
        info.update({
            "Number of Slots": str(geo.num_slots),
            "Slot Opening (Bs0)": f"{s.Bs0:.2f} mm",
            "Slot Shoulder (Bs1)": f"{s.Bs1:.2f} mm",
            "Slot Bottom (Bs2)": f"{s.Bs2:.2f} mm",
            "Slot Opening Ht (Hs0)": f"{s.Hs0:.2f} mm",
            "Slot Shoulder Ht (Hs1)": f"{s.Hs1:.2f} mm",
            "Slot Body Ht (Hs2)": f"{s.Hs2:.2f} mm",
            "Slot Area": f"{c['slot_area_mm2']:.2f} mm²",
            "Fill Factor": f"{c['effective_slot_fill']*100:.1f}%",
            "Slot Pitch": f"{c['slot_pitch_angle_deg']:.1f}°",
        })
    else:
        # Slotless
        info.update({
            "Winding Inner R": f"{geo.winding_inner_radius:.1f} mm",
            "Winding Outer R": f"{geo.winding_outer_radius:.1f} mm",
        })

    info.update({
        "Number of Poles": str(geo.num_poles),
        "Magnet Thickness": f"{geo.magnet_thickness:.2f} mm",
        "Magnet Arc": f"{c['magnet_arc_angle_deg']:.1f}°",
        "Magnet Grade": geo.magnet_grade,
        "Max Magnet Temp": f"{geo.magnet_max_temp:.0f} °C",
        "Housing Wall": f"{geo.housing_wall_thickness:.1f} mm",
        "Insulation Class": geo.insulation_class,
    })
    return info
