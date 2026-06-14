"""
Clean 2D cross-section plotter for motor geometry.
Uses Wedge patches for slots - much simpler and symmetrical!
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Wedge, Circle, Polygon
from typing import Dict

COLORS = {
    "shaft": "#4a4a4a",
    "rotor_core": "#8B7355",
    "magnet": "#2E8B57",
    "airgap": "#F0F0F0",
    "stator_yoke": "#909090",
    "stator_tooth": "#808080",
    "winding": "#D4A017",
    "winding_slotless": "#C8961E",
    "housing": "#4169E1",
    "background": "#FFFFFF",
}


class MotorGeometryCanvasV3(FigureCanvas):
    """Simple, clean motor cross-section renderer."""

    def __init__(self, parent=None, width=8, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor(COLORS["background"])
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)

    def clear(self):
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x [mm]")
        self.ax.set_ylabel("y [mm]")
        self.ax.grid(True, alpha=0.3)

    def draw_section(self, geo, show_labels=True, show_dimensions=True):
        """Draw the motor cross-section."""
        from .motor_geometry_v2 import MotorGeometry

        self.clear()
        ax = self.ax
        is_slotted = geo.structure_type == "Slotted"

        Rso, Rsi = geo.Rso, geo.Rsi
        Rro, Rshaft = geo.Rro, geo.shaft_radius
        hw = geo.housing_wall_thickness

        # === 1. Housing ===
        ax.add_patch(Circle((0, 0), Rso + hw,
                           facecolor=COLORS["housing"],
                           edgecolor="black", linewidth=1.2, alpha=0.6))

        # === 2. Stator ===
        if is_slotted:
            self._draw_slotted(ax, geo)
        else:
            self._draw_slotless(ax, geo)

        # === 3. Airgap (thin ring between Rro and Rsi) ===
        ax.add_patch(Circle((0, 0), Rsi,
                           facecolor=COLORS["airgap"],
                           edgecolor="gray", linewidth=0.5, alpha=0.3))

        # === 4. Magnets ===
        pole_angle = 2 * math.pi / geo.num_poles
        mag_arc = pole_angle * geo.magnet_span_ratio
        for p in range(geo.num_poles):
            theta_c = p * pole_angle
            t_start = math.degrees(theta_c - mag_arc / 2)
            t_end = math.degrees(theta_c + mag_arc / 2)
            ax.add_patch(Wedge((0, 0), Rro, t_start, t_end,
                               width=geo.magnet_thickness,
                               facecolor=COLORS["magnet"],
                               edgecolor="black", linewidth=0.6, alpha=0.85))
            if show_labels:
                tm = (t_start + t_end) / 2 * math.pi / 180
                rl = Rro - geo.magnet_thickness / 2
                ax.annotate("Mag", (rl * math.cos(tm), rl * math.sin(tm)),
                           fontsize=5, ha="center", va="center", color="white")

        # === 5. Rotor core ===
        rotor_r = Rro - geo.magnet_thickness
        ax.add_patch(Circle((0, 0), rotor_r,
                           facecolor=COLORS["rotor_core"],
                           edgecolor="black", linewidth=1.0, alpha=0.7))

        # === 6. Shaft ===
        ax.add_patch(Circle((0, 0), Rshaft,
                           facecolor=COLORS["shaft"],
                           edgecolor="black", linewidth=1.5))

        # === Labels ===
        if show_labels:
            self._draw_labels(ax, geo)

        # === Dimensions ===
        if show_dimensions:
            self._draw_dims(ax, geo)

        max_r = Rso + hw + 12
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.set_aspect("equal")
        ax.set_title("Motor Cross-Section (Radial View)", fontweight="bold")
        self.fig.tight_layout()
        self.draw()

    def _draw_slotted(self, ax, geo):
        """Draw slotted stator using Wedge patches - simple and symmetrical."""
        s = geo.slot
        Rsi, Rso = geo.Rsi, geo.Rso
        N = geo.num_slots

        # Stator yoke (just the outer ring)
        ax.add_patch(Circle((0, 0), Rso,
                           facecolor=COLORS["stator_yoke"],
                           edgecolor="black", linewidth=1.2, alpha=0.7))

        # Inner bore of stator yoke = slot bottom radius
        Rslot_bottom = Rsi + s.Hs0 + s.Hs1 + s.Hs2

        # Draw each slot as a Wedge going from bore outward
        # The slot has three radial segments:
        #   Hs0: opening (width Bs0)
        #   Hs1: shoulder (width Bs0 → Bs1)
        #   Hs2: body (width Bs1 → Bs2)
        #
        # For simplicity, we approximate the slot as a single wedge
        # whose angular width changes at each radial height.
        # Using the opening width at the bore vs. bottom width at the bottom,
        # we draw a trapezoidal shape.

        slot_pitch = 2 * math.pi / N

        for i in range(N):
            theta_c = i * slot_pitch

            # === Draw the slot as a series of annular wedges ===
            # Segment 1: Opening (height Hs0, width Bs0)
            ha0 = s.Bs0 / 2 / Rsi
            t_deg_l = math.degrees(theta_c - ha0)
            t_deg_r = math.degrees(theta_c + ha0)

            # Use Polygon for the slot to get trapezoidal shape
            # Build polygon vertices going CCW around the slot

            # Define radii
            r_br = Rsi              # bore
            r_ot = Rsi + s.Hs0      # opening top = shoulder start
            r_st = Rsi + s.Hs0 + s.Hs1  # shoulder bottom
            r_sb = Rsi + s.Hs0 + s.Hs1 + s.Hs2  # slot bottom

            # Half-angles at each level
            ha_br = s.Bs0 / 2 / r_br
            ha_ot = s.Bs0 / 2 / r_ot  # Same width Bs0 at opening top
            ha_st = s.Bs1 / 2 / r_st
            ha_sb = s.Bs2 / 2 / r_sb

            # Build polygon
            pts = []

            # Bottom of slot opening: left → right
            # (we'll trace the outer boundary of the slot)
            # Going CCW: start bottom-left, go outward, right, back inward

            # Helper
            def add_arc_pts(out, r, t1, t2, n=10):
                for j in range(n + 1):
                    t = t1 + (t2 - t1) * j / n
                    out.append([r * math.cos(t), r * math.sin(t)])

            # 1. Bore left to opening top left (radial line)
            pts.append([r_br * math.cos(theta_c - ha_br),
                        r_br * math.sin(theta_c - ha_br)])
            pts.append([r_ot * math.cos(theta_c - ha_ot),
                        r_ot * math.sin(theta_c - ha_ot)])

            # 2. Taper from opening top to shoulder bottom (left side)
            Nt = 6
            for j in range(1, Nt + 1):
                frac = j / Nt
                r = r_ot + frac * (r_st - r_ot)
                w = s.Bs0 + frac * (s.Bs1 - s.Bs0)
                ha = w / 2 / r
                pts.append([r * math.cos(theta_c - ha), r * math.sin(theta_c - ha)])

            # 3. Taper from shoulder bottom to slot bottom (left side)
            for j in range(1, Nt + 1):
                frac = j / Nt
                r = r_st + frac * (r_sb - r_st)
                w = s.Bs1 + frac * (s.Bs2 - s.Bs1)
                ha = w / 2 / r
                pts.append([r * math.cos(theta_c - ha), r * math.sin(theta_c - ha)])

            # 4. Bottom arc (left → right)
            add_arc_pts(pts, r_sb, theta_c - ha_sb, theta_c + ha_sb, 8)

            # 5. Taper from slot bottom to shoulder bottom (right side)
            for j in range(Nt, 0, -1):
                frac = j / Nt
                r = r_st + frac * (r_sb - r_st)
                w = s.Bs1 + frac * (s.Bs2 - s.Bs1)
                ha = w / 2 / r
                pts.append([r * math.cos(theta_c + ha), r * math.sin(theta_c + ha)])

            # 6. Taper from shoulder bottom to opening top (right side)
            for j in range(Nt, 0, -1):
                frac = j / Nt
                r = r_ot + frac * (r_st - r_ot)
                w = s.Bs0 + frac * (s.Bs1 - s.Bs0)
                ha = w / 2 / r
                pts.append([r * math.cos(theta_c + ha), r * math.sin(theta_c + ha)])

            # 7. Opening top right to bore right
            pts.append([r_ot * math.cos(theta_c + ha_ot),
                        r_ot * math.sin(theta_c + ha_ot)])
            pts.append([r_br * math.cos(theta_c + ha_br),
                        r_br * math.sin(theta_c + ha_br)])

            # 8. Bore arc (right → left) closing
            add_arc_pts(pts, r_br, theta_c + ha_br, theta_c - ha_br, 6)

            ax.add_patch(Polygon(pts, facecolor=COLORS["winding"],
                                 edgecolor="black", linewidth=0.4, alpha=0.85))

        # Draw the yoke internal cutout (inner bore at slot bottom)
        # This makes the yoke visible only outside the slot bottom
        yoke_inner = Circle((0, 0), Rslot_bottom,
                            facecolor=COLORS["background"],
                            edgecolor="none", linewidth=0, alpha=0.0)
        ax.add_patch(yoke_inner)

        # Now draw teeth as complementary wedges between slots
        for i in range(N):
            theta_c = i * slot_pitch
            next_c = ((i + 1) * slot_pitch)

            # The tooth is the region between slot i's right edge
            # and slot i+1's left edge

            r_br = Rsi
            r_ot = Rsi + s.Hs0
            r_st = Rsi + s.Hs0 + s.Hs1
            r_sb = Rsi + s.Hs0 + s.Hs1 + s.Hs2

            ha_br = s.Bs0 / 2 / r_br
            ha_ot = s.Bs0 / 2 / r_ot
            ha_st = s.Bs1 / 2 / r_st
            ha_sb = s.Bs2 / 2 / r_sb

            n_ha_br = s.Bs0 / 2 / r_br
            n_ha_ot = s.Bs0 / 2 / r_ot
            n_ha_st = s.Bs1 / 2 / r_st
            n_ha_sb = s.Bs2 / 2 / r_sb

            # Tooth left edge = slot right edge at each level
            t_left_br = theta_c + ha_br
            t_left_ot = theta_c + ha_ot
            t_left_st = theta_c + ha_st
            t_left_sb = theta_c + ha_sb

            # Tooth right edge = next slot left edge at each level
            t_right_br = next_c - n_ha_br
            t_right_ot = next_c - n_ha_ot
            t_right_st = next_c - n_ha_st
            t_right_sb = next_c - n_ha_sb

            # Build tooth polygon
            # Outer boundary (at slot bottom): left → right
            # Inner boundary (at bore): right → left
            tp = []
            Nt2 = 6

            # Bottom arc (left → right)
            for j in range(Nt2 + 1):
                t = t_left_sb + (t_right_sb - t_left_sb) * j / Nt2
                tp.append([r_sb * math.cos(t), r_sb * math.sin(t)])

            # Right side inward (slot bottom → shoulder bottom)
            for j in range(Nt2 + 1):
                frac = j / Nt2
                r = r_sb - frac * (r_sb - r_st)
                w = (s.Bs2 - frac * (s.Bs2 - s.Bs1)) / 2
                ha = w / r
                tp.append([r * math.cos(next_c - ha), r * math.sin(next_c - ha)])

            # Continue inward (shoulder bottom → opening top)
            for j in range(1, Nt2 + 1):
                frac = j / Nt2
                r = r_st - frac * (r_st - r_ot)
                w = (s.Bs1 - frac * (s.Bs1 - s.Bs0)) / 2
                ha = w / r
                tp.append([r * math.cos(next_c - ha), r * math.sin(next_c - ha)])

            # Opening top → bore
            tp.append([r_br * math.cos(next_c - n_ha_br),
                       r_br * math.sin(next_c - n_ha_br)])

            # Bore arc (right → left)
            for j in range(Nt2 + 1):
                t = t_right_br - (t_right_br - t_left_br) * j / Nt2
                tp.append([r_br * math.cos(t), r_br * math.sin(t)])

            # Left side outward (bore → opening top)
            tp.append([r_ot * math.cos(theta_c + ha_ot),
                       r_ot * math.sin(theta_c + ha_ot)])

            # Continue outward (opening top → shoulder bottom)
            for j in range(1, Nt2 + 1):
                frac = j / Nt2
                r = r_ot + frac * (r_st - r_ot)
                w = (s.Bs0 + frac * (s.Bs1 - s.Bs0)) / 2
                ha = w / r
                tp.append([r * math.cos(theta_c + ha), r * math.sin(theta_c + ha)])

            # Shoulder bottom → slot bottom
            for j in range(1, Nt2 + 1):
                frac = j / Nt2
                r = r_st + frac * (r_sb - r_st)
                w = (s.Bs1 + frac * (s.Bs2 - s.Bs1)) / 2
                ha = w / r
                tp.append([r * math.cos(theta_c + ha), r * math.sin(theta_c + ha)])

            ax.add_patch(Polygon(tp, facecolor=COLORS["stator_tooth"],
                                 edgecolor="black", linewidth=0.6, alpha=0.9))

    def _draw_slotless(self, ax, geo):
        """Draw slotless stator: yoke + winding band."""
        # Yoke
        ax.add_patch(Circle((0, 0), geo.Rso,
                           facecolor=COLORS["stator_yoke"],
                           edgecolor="black", linewidth=1.2, alpha=0.7))
        # Winding band
        Rwi, Rwo = geo.winding_inner_radius, geo.winding_outer_radius
        ax.add_patch(Circle((0, 0), Rwo,
                           facecolor=COLORS["winding_slotless"],
                           edgecolor="black", linewidth=1.0, alpha=0.8))
        # Inner bore of winding
        ax.add_patch(Circle((0, 0), Rwi,
                           facecolor=COLORS["airgap"],
                           edgecolor="black", linewidth=0.8))

    def _draw_labels(self, ax, geo):
        """Component labels."""
        Rso = geo.Rso
        hw = geo.housing_wall_thickness
        is_slotted = geo.structure_type == "Slotted"

        def lbl(x, y, text, **kw):
            ax.annotate(text, (x, y), fontsize=7, ha="center",
                       va="center", fontweight="bold", **kw)

        lbl(0, Rso + hw + 4, "Housing", color="white")
        lbl(Rso * 0.7, Rso * 0.7, "Stator\nYoke", color="black")

        if is_slotted:
            r_mid = geo.Rsi + (geo.slot.Hs0 + geo.slot.Hs1 + geo.slot.Hs2) / 2
            lbl(0, max(r_mid, 1), "Winding", color="#8B6914")
        else:
            r_mid = (geo.winding_inner_radius + geo.winding_outer_radius) / 2
            lbl(0, r_mid, "Winding", color="#8B6914")

        lbl(geo.Rro + (geo.Rsi - geo.Rro) / 2, 0, "Airgap",
            color="gray", rotation=90)
        lbl(0, (geo.Rro - geo.magnet_thickness) / 2, "Rotor", color="black")
        lbl(0, 0, "Shaft", color="white")

    def _draw_dims(self, ax, geo):
        """Dimension annotations."""
        Rso, Rsi, Rro = geo.Rso, geo.Rsi, geo.Rro
        hw = geo.housing_wall_thickness
        max_r = Rso + hw
        yd = max_r + 6

        ax.annotate(f"Rso={Rso}", xy=(Rso, 0), xytext=(Rso, yd),
                   arrowprops=dict(arrowstyle="->", lw=0.6),
                   fontsize=7, ha="center", va="bottom")
        ax.annotate(f"Rsi={Rsi}", xy=(Rsi, 0), xytext=(Rsi, yd - 3),
                   arrowprops=dict(arrowstyle="->", lw=0.6),
                   fontsize=7, ha="center", va="bottom")
        ax.annotate(f"Rro={Rro}", xy=(Rro, 0), xytext=(Rro, yd - 6),
                   arrowprops=dict(arrowstyle="->", lw=0.6),
                   fontsize=7, ha="center", va="bottom")

        stype = "Slotted" if geo.structure_type == "Slotted" else "Slotless"
        info = (f"{geo.num_poles}-pole | {stype}\n"
                f"Magnet: {geo.magnet_grade} | Lstk={geo.stack_length} mm")
        ax.annotate(info, xy=(0, -max_r * 0.88), fontsize=8,
                   ha="center", va="top",
                   bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="lightyellow",
                           edgecolor="black", alpha=0.8))
