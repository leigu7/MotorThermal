"""
Slot geometry definitions inspired by JMAG / Ansys Maxwell conventions.

For a typical slotted stator, the slot is defined by:

     Stator Yoke (outer)
    ┌─────────────────────┐
    │   ┌─────────────┐   │  ← Slot bottom (Rsb)
    │   │  SLOT       │   │
    │   │             │   │
    │   │  ┌───────┐  │   │  ← Tooth tip start
    │   │  │       │  │   │
    │   │  │       ├──┼───┤  ← Slot opening (Sop)
    │   └──┤       │  │   │
    └──────┴───────┴──┴───┘  ← Stator bore (Rsi)

Key parameters (JMAG/Maxwell naming):
  - Hs0: Slot opening height (wedge/tooth tip height)
  - Hs1: Slot wedge height / tooth tip height
  - Hs2: Slot body height (straight section)
  - Bs0: Slot opening width
  - Bs1: Tooth tip width / shoulder width
  - Bs2: Slot bottom width
  - Rs: Slot bottom fillet radius
  - Tooth_width: Tooth width at airgap (or parallel tooth)

For parallel tooth (most common):
  Tooth width is constant, slot width varies with radius.

For parallel slot:
  Slot width is constant, tooth width varies with radius.

We'll implement both styles.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import math


@dataclass
class SlotGeometry:
    """
    Slot geometry parameters in JMAG/Maxwell convention.

    Stator bore: Rsi
    Slot bottom radius: Rsb = Rsi + Hs0 + Hs1 + Hs2

    Tooth tip starts at Rsi, extends to Rsi + Hs0
    Slot wedge region from Rsi + Hs0 to Rsi + Hs0 + Hs1
    Slot body from Rsi + Hs0 + Hs1 to Rsb
    """

    # === Slot heights (radial) ===
    Hs0: float = 0.5    # Slot opening height [mm] (tooth tip thickness at bore)
    Hs1: float = 0.5    # Wedge/Shoulder height [mm]
    Hs2: float = 9.0    # Slot body height [mm]

    # === Slot widths (tangential) ===
    Bs0: float = 2.0    # Slot opening width [mm] (at bore surface)
    Bs1: float = 4.0    # Slot shoulder width [mm] (below tooth tip)
    Bs2: float = 6.0    # Slot bottom width [mm]

    # === Tooth ===
    tooth_type: str = "Parallel tooth"  # "Parallel tooth" or "Parallel slot"
    tooth_width: float = 3.0  # Tooth width at airgap [mm] (for parallel tooth)

    # === Fillet ===
    Rs_fillet: float = 0.5   # Slot bottom fillet radius [mm]

    # === Derived ===
    @property
    def total_height(self) -> float:
        return self.Hs0 + self.Hs1 + self.Hs2

    @property
    def slot_bottom_radius(self) -> float:
        """Return the slot bottom radius (requires stator bore Rsi set separately)."""
        return None  # needs Rsi context

    def get_slot_bottom_r(self, Rsi: float) -> float:
        return Rsi + self.total_height

    def get_tooth_tip_r(self, Rsi: float) -> float:
        return Rsi + self.Hs0

    def get_wedge_bottom_r(self, Rsi: float) -> float:
        return Rsi + self.Hs0 + self.Hs1

    def get_slot_area_mm2(self) -> float:
        """
        Approximate slot area (trapezoid approximation).
        Proper area needs Rsi context, this is a rough estimate.
        """
        avg_width = (self.Bs0 + self.Bs2) / 2
        return avg_width * (self.Hs1 + self.Hs2)


def slot_from_simple_params(
    Rsi: float,
    Rso: float,
    num_slots: int,
    slot_opening: float,
    slot_depth: float,
    tooth_width_min: float,
    wedge_height: float,
) -> SlotGeometry:
    """
    Convert simple parameters to SlotGeometry.
    This bridges the old simple interface to the detailed one.

    Simple parameters:
      - slot_opening: width at bore (→ Bs0)
      - slot_depth: radial depth from bore (→ Hs0 + Hs1 + Hs2)
      - tooth_width_min: minimum tooth width (used to derive Bs1)
      - wedge_height: height of wedge / tooth tip region

    Returns a SlotGeometry with heuristic estimates for other params.
    """
    slot_pitch_angle = 2 * math.pi / num_slots
    slot_pitch_at_bore = slot_pitch_angle * Rsi

    # Tooth width at bore = tooth_width_min
    # Slot pitch at bore = arc length per slot
    slot_pitch_at_bore_arc = slot_pitch_angle * Rsi

    # Bs0 = slot_opening
    Bs0 = slot_opening

    # Estimate Bs1 from tooth_width_min and slot pitch
    Bs1 = slot_pitch_at_bore_arc - tooth_width_min

    # At slot bottom, estimate width
    Rslot_bottom = Rsi + slot_depth
    slot_pitch_at_bottom = slot_pitch_angle * Rslot_bottom
    # If parallel tooth: tooth width same, so slot bottom width = pitch - tooth_width
    Bs2 = slot_pitch_at_bottom - tooth_width_min

    return SlotGeometry(
        Hs0=wedge_height,
        Hs1=slot_depth * 0.1,  # heuristic: 10% for shoulder
        Hs2=slot_depth * 0.9 - wedge_height,  # remaining
        Bs0=Bs0,
        Bs1=max(Bs1, Bs0 + 0.5),  # ensure Bs1 > Bs0
        Bs2=max(Bs2, Bs1),  # ensure Bs2 >= Bs1
        tooth_width=tooth_width_min,
        tooth_type="Parallel tooth",
        Rs_fillet=0.5,
    )
