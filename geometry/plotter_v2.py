"""
Matplotlib-based 2D cross-section plotter for motor geometry v2.
Draws an interactive figure with proper slot geometry and slotless support.
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Wedge, Circle, Arc
from matplotlib.collections import PatchCollection
from typing import Dict

# Color scheme for motor components
COLORS = {
    "shaft": "#4a4a4a",
    "rotor_core": "#8B7355",
    "magnet": "#2E8B57",
    "magnet_alt": "#1a5e3a",
    "airgap": "#F0F0F0",
    "stator_core": "#A0A0A0",
    "stator_yoke": "#909090",
    "stator_tooth": "#808080",
    "winding": "#D4A017",
    "winding_slotless": "#C8961E",
    "slot_liner": "#8B4513",
    "housing": "#4169E1",
    "housing_edge": "#2a4fa8",
    "background": "#FFFFFF",
}


class MotorGeometryCanvasV2(FigureCanvas):
    """
    Matplotlib canvas for rendering 2D motor cross-section.
    """

    def __init__(self, parent=None, width=8, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor(COLORS["background"])
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x [mm]")
        self.ax.set_ylabel("y [mm]")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Motor Cross-Section (Radial View)", fontweight="bold")

    def clear(self):
        """Clear the plot."""
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x [mm]")
        self.ax.set_ylabel("y [mm]")
        self.ax.grid(True, alpha=0.3)

    def draw_cross_section(self, geo, show_labels=True, show_dimensions=True):
        """
        Main drawing function using MotorGeometry.
        """
        from .motor_geometry_v2 import MotorGeometry, generate_cross_section_polygons

        self.clear()
        ax = self.ax
        is_slotted = geo.structure_type == "Slotted"

        Rso = geo.Rso
        Rsi = geo.Rsi
        Rro = geo.Rro
        Rshaft = geo.shaft_radius
        hw = geo.housing_wall_thickness
        num_poles = geo.num_poles

        # === 1. Housing (outermost ring) ===
        housing = Circle((0, 0), Rso + hw,
                         facecolor=COLORS["housing"],
                         edgecolor=COLORS["housing_edge"],
                         linewidth=1.5, alpha=0.6)
        ax.add_patch(housing)
        # Housing inner edge
        housing_inner = Circle((0, 0), Rso,
                               facecolor="none",
                               edgecolor=COLORS["housing_edge"],
                               linewidth=1.0)
        ax.add_patch(housing_inner)

        # === 2. Stator ===
        if is_slotted:
            self._draw_slotted_stator(ax, geo)
        else:
            self._draw_slotless_stator(ax, geo)

        # === 3. Airgap (region between rotor and stator) ===
        # Draw as a thin ring
        airgap_outline = Circle((0, 0), Rsi,
                                facecolor=COLORS["airgap"],
                                edgecolor="gray", linewidth=0.5, alpha=0.4)
        ax.add_patch(airgap_outline)

        # Rotor outline at airgap
        rotor_outline = Circle((0, 0), Rro,
                               facecolor="none",
                               edgecolor="black", linewidth=0.8)
        ax.add_patch(rotor_outline)

        # === 4. Magnets (segmented wedges) ===
        pole_angle = 2 * math.pi / num_poles
        magnet_arc = pole_angle * geo.magnet_span_ratio
        Rmag_inner = Rro - geo.magnet_thickness

        for p in range(num_poles):
            theta_center = p * pole_angle
            t_start = math.degrees(theta_center - magnet_arc / 2)
            t_end = math.degrees(theta_center + magnet_arc / 2)

            magnet_patch = Wedge((0, 0), Rro,
                                 t_start, t_end,
                                 width=geo.magnet_thickness,
                                 facecolor=COLORS["magnet"],
                                 edgecolor="black", linewidth=0.6,
                                 alpha=0.85)
            ax.add_patch(magnet_patch)

            if show_labels:
                tm = (t_start + t_end) / 2 * math.pi / 180
                rl = Rro - geo.magnet_thickness / 2
                ax.annotate("Magnet",
                            (rl * math.cos(tm), rl * math.sin(tm)),
                            fontsize=6, ha="center", va="center",
                            color="white", fontweight="bold")

        # === 5. Rotor core ===
        rotor_r = Rro - geo.magnet_thickness
        rotor_core = Circle((0, 0), rotor_r,
                            facecolor=COLORS["rotor_core"],
                            edgecolor="black", linewidth=1.0, alpha=0.7)
        ax.add_patch(rotor_core)

        # Rotor inner boundary (if it differs from shaft)
        if Rro - geo.magnet_thickness > geo.shaft_radius:
            rotor_inner = Circle((0, 0), geo.shaft_radius,
                                 facecolor="none",
                                 edgecolor="black", linewidth=0.5, alpha=0.3)
            ax.add_patch(rotor_inner)

        # === 6. Shaft ===
        shaft = Circle((0, 0), Rshaft,
                       facecolor=COLORS["shaft"],
                       edgecolor="black", linewidth=1.5)
        ax.add_patch(shaft)

        # === 7. Labels ===
        if show_labels:
            self._draw_labels(ax, geo)

        # === 8. Dimensions ===
        if show_dimensions:
            self._draw_dimensions(ax, geo)

        # Set plot limits
        max_r = Rso + hw + 12
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.set_aspect("equal")
        ax.set_title("Motor Cross-Section (Radial View)", fontweight="bold")

        self.fig.tight_layout()
        self.draw()

    def _draw_slotted_stator(self, ax, geo):
        """Draw slotted stator with proper tooth/slot geometry.
        
        Slot definition (JMAG/Maxwell inspired):
          - Hs0: tooth tip height (opening height)
          - Hs1: wedge/shoulder height
          - Hs2: slot body height
          - Bs0: slot opening width (at bore)
          - Bs1: slot shoulder width (at wedge bottom)
          - Bs2: slot bottom width
        
        The slot expands linearly from Bs0 → Bs1 → Bs2 along the radial direction.
        
        Drawing approach:
          1. Draw stator yoke as a filled circle with inner bore cutout
          2. For each slot, draw the slot region (winding) as a polygon
          3. The teeth are the regions between slots (already visible as yoke color)
        """
        s = geo.slot
        Rsi = geo.Rsi
        Rso = geo.Rso
        N = geo.num_slots
        slot_pitch = 2 * math.pi / N
        
        # Radial positions
        r0 = Rsi                               # bore (tooth tip, opening bottom)
        r1 = Rsi + s.Hs0                       # opening top / shoulder bottom
        r2 = Rsi + s.Hs0 + s.Hs1               # shoulder top / slot body bottom  
        r3 = Rsi + s.Hs0 + s.Hs1 + s.Hs2       # slot bottom
        
        # === Stator yoke (outer ring) ===
        yoke = Circle((0, 0), Rso,
                      facecolor=COLORS["stator_yoke"],
                      edgecolor="black", linewidth=1.2, alpha=0.7)
        ax.add_patch(yoke)
        
        # Draw a circle at slot bottom so yoke doesn't show where slots are
        slot_bottom_circle = Circle((0, 0), r3,
                                    facecolor="none",
                                    edgecolor="black", linewidth=0.5, alpha=0.3)
        ax.add_patch(slot_bottom_circle)
        
        def slot_half_angle(r):
            """Return the half-angle span of the slot at a given radius.
            Linearly interpolates based on the slot profile."""
            if r <= r0 + 1e-9:
                return s.Bs0 / 2 / r0
            elif r <= r1:
                return s.Bs0 / 2 / r  # constant width = Bs0
            elif r <= r2:
                frac = (r - r1) / (r2 - r1)
                w = s.Bs0 + frac * (s.Bs1 - s.Bs0)
                return w / 2 / r
            else:
                frac = min((r - r2) / (r3 - r2), 1.0)
                w = s.Bs1 + frac * (s.Bs2 - s.Bs1)
                return w / 2 / r
        
        # === Draw each slot polygon ===
        for i in range(N):
            theta_c = i * slot_pitch  # center of this slot
            
            # Build slot polygon going CCW
            slot_pts = []
            
            # Number of interpolation points
            N_seg = 16
            
            # --- Left side: from bore (r0) to slot bottom (r3) ---
            for j in range(N_seg + 1):
                frac = j / N_seg
                r = r0 + frac * (r3 - r0)
                ha = slot_half_angle(r)
                slot_pts.append([r * math.cos(theta_c - ha),
                                 r * math.sin(theta_c - ha)])
            
            # --- Bottom arc: left to right ---
            ha_bottom = slot_half_angle(r3)
            for j in range(1, N_seg + 1):
                t = theta_c - ha_bottom + (2 * ha_bottom) * j / N_seg
                slot_pts.append([r3 * math.cos(t), r3 * math.sin(t)])
            
            # --- Right side: from slot bottom (r3) back to bore (r0) ---
            for j in range(N_seg, -1, -1):
                frac = j / N_seg
                r = r0 + frac * (r3 - r0)
                ha = slot_half_angle(r)
                slot_pts.append([r * math.cos(theta_c + ha),
                                 r * math.sin(theta_c + ha)])
            
            # --- Top arc (bore): right to left (closing) ---
            ha_top = slot_half_angle(r0)
            for j in range(1, N_seg):
                t = theta_c + ha_top - (2 * ha_top) * j / N_seg
                slot_pts.append([r0 * math.cos(t), r0 * math.sin(t)])
            
            slot_poly = Polygon(slot_pts, facecolor=COLORS["winding"],
                                edgecolor="black", linewidth=0.4, alpha=0.85)
            ax.add_patch(slot_poly)
        
        # === Draw tooth outlines (boundaries between slots) ===
        for i in range(N):
            theta_c = i * slot_pitch
            next_c = ((i + 1) * slot_pitch)
            
            # Left edge of tooth = right edge of slot i
            # Right edge of tooth = left edge of slot i+1
            
            # We just draw the visible edges: the tooth radial sides
            # Left side of tooth (= right side of slot i)
            ha_right = slot_half_angle(r0)
            for r in (r0, r1, r2, r3):
                ha = slot_half_angle(r)
                # Right edge of slot i
                ax.plot(r * math.cos(theta_c + ha), 
                        r * math.sin(theta_c + ha),
                        'o', color=COLORS["stator_tooth"], markersize=0.5)
            
            # Draw radial lines along tooth edges for visual clarity
            # Right edge of slot i (left edge of tooth)
            pts_left = []
            for j in range(10):
                frac = j / 9
                r = r0 + frac * (r3 - r0)
                ha = slot_half_angle(r)
                pts_left.append([r * math.cos(theta_c + ha), 
                                 r * math.sin(theta_c + ha)])
            
            # Left edge of slot i+1 (right edge of tooth)
            pts_right = []
            for j in range(10):
                frac = j / 9
                r = r0 + frac * (r3 - r0)
                n_ha = slot_half_angle(r)
                pts_right.append([r * math.cos(next_c - n_ha), 
                                  r * math.sin(next_c - n_ha)])
            
            # Draw tooth as a quadrilateral
            if len(pts_left) >= 2 and len(pts_right) >= 2:
                # Tooth at bore
                ha_l = slot_half_angle(r0)
                ha_r = slot_half_angle(r0)
                t_l = theta_c + ha_l
                t_r = next_c - ha_r
                if t_r > t_l:
                    tooth_pts = []
                    # Bore arc (left to right)
                    for j in range(6):
                        t = t_l + (t_r - t_l) * j / 5
                        tooth_pts.append([r0 * math.cos(t), r0 * math.sin(t)])
                    # Right side down
                    for j in range(1, 7):
                        frac = j / 6
                        r = r0 + frac * (r3 - r0)
                        n_ha = slot_half_angle(r)
                        tooth_pts.append([r * math.cos(next_c - n_ha),
                                          r * math.sin(next_c - n_ha)])
                    # Bottom arc (right to left)  
                    ha_lb = slot_half_angle(r3)
                    ha_rb = slot_half_angle(r3)
                    t_lb = theta_c + ha_lb
                    t_rb = next_c - ha_rb
                    for j in range(5, -1, -1):
                        t = t_lb + (t_rb - t_lb) * j / 5
                        tooth_pts.append([r3 * math.cos(t), r3 * math.sin(t)])
                    # Left side up
                    for j in range(5, -1, -1):
                        frac = j / 6
                        r = r0 + frac * (r3 - r0)
                        ha = slot_half_angle(r)
                        tooth_pts.append([r * math.cos(theta_c + ha),
                                          r * math.sin(theta_c + ha)])
                    
                    tooth_poly = Polygon(tooth_pts,
                                         facecolor=COLORS["stator_tooth"],
                                         edgecolor="black", linewidth=0.6, alpha=0.9)
                    ax.add_patch(tooth_poly)

    def _draw_slotless_stator(self, ax, geo):
        """Draw slotless stator: yoke ring + winding band."""
        # Stator yoke (outer ring)
        yoke = Circle((0, 0), geo.Rso,
                      facecolor=COLORS["stator_yoke"],
                      edgecolor="black", linewidth=1.2, alpha=0.7)
        ax.add_patch(yoke)

        # Winding band (annular ring between winding_inner_r and winding_outer_r)
        Rwi = geo.winding_inner_radius
        Rwo = geo.winding_outer_radius

        winding_band = Circle((0, 0), Rwo,
                              facecolor=COLORS["winding_slotless"],
                              edgecolor="black", linewidth=1.0, alpha=0.8)
        ax.add_patch(winding_band)
        # Inner bore of winding
        winding_inner = Circle((0, 0), Rwi,
                               facecolor=COLORS["airgap"],
                               edgecolor="black", linewidth=0.8)
        ax.add_patch(winding_inner)

    def _draw_labels(self, ax, geo):
        """Draw component labels."""
        is_slotted = geo.structure_type == "Slotted"
        Rso = geo.Rso
        hw = geo.housing_wall_thickness

        # Housing label
        ax.annotate("Housing",
                     (0, Rso + hw + 4),
                     fontsize=8, ha="center", va="bottom",
                     color=COLORS["housing_edge"], fontweight="bold")

        if is_slotted:
            ax.annotate("Stator\nYoke",
                         (Rso * 0.75, Rso * 0.75),
                         fontsize=7, ha="center",
                         color="black", fontweight="bold")
            ax.annotate("Winding\n(Slots)",
                         (0, geo.Rsi + geo.slot.Hs0 + geo.slot.Hs1 + geo.slot.Hs2 / 2),
                         fontsize=7, ha="center", va="center",
                         color="#8B6914", fontweight="bold")
        else:
            ax.annotate("Stator\nYoke",
                         (Rso * 0.75, Rso * 0.75),
                         fontsize=7, ha="center",
                         color="black", fontweight="bold")
            Rwi = geo.winding_inner_radius
            Rwo = geo.winding_outer_radius
            ax.annotate("Airgap\nWinding",
                         (0, (Rwi + Rwo) / 2),
                         fontsize=7, ha="center", va="center",
                         color="#8B6914", fontweight="bold")

        ax.annotate("Airgap",
                     (geo.Rro + (geo.Rsi - geo.Rro) / 2, 0),
                     fontsize=7, ha="center", va="center",
                     color="gray", fontweight="bold", rotation=90)

        ax.annotate("Rotor\nCore",
                     (0, (geo.Rro - geo.magnet_thickness) / 2),
                     fontsize=7, ha="center", va="center",
                     color="black", fontweight="bold")

        ax.annotate("Shaft",
                     (0, 0), fontsize=7,
                     ha="center", va="center",
                     color="white", fontweight="bold")

    def _draw_dimensions(self, ax, geo):
        """Draw dimension annotations."""
        Rso = geo.Rso
        Rsi = geo.Rsi
        Rro = geo.Rro
        hw = geo.housing_wall_thickness
        max_r = Rso + hw
        y_dim = max_r + 6

        # Stator OD
        ax.annotate(f"R$_{{\\mathrm{{so}}}}$={Rso}",
                     xy=(Rso, 0), xytext=(Rso, y_dim),
                     arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
                     fontsize=7, ha="center", va="bottom")

        # Stator ID
        ax.annotate(f"R$_{{\\mathrm{{si}}}}$={Rsi}",
                     xy=(Rsi, 0), xytext=(Rsi, y_dim - 3),
                     arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
                     fontsize=7, ha="center", va="bottom")

        # Rotor OD
        ax.annotate(f"R$_{{\\mathrm{{ro}}}}$={Rro}",
                     xy=(Rro, 0), xytext=(Rro, y_dim - 6),
                     arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
                     fontsize=7, ha="center", va="bottom")

        # Info box
        if geo.structure_type == "Slotted":
            info = (f"{geo.num_poles}-pole, {geo.num_slots}-slot\n"
                    f"Magnet: {geo.magnet_grade} (Tmax={geo.magnet_max_temp:.0f}°C)\n"
                    f"Lstk={geo.stack_length} mm | Slotted")
        else:
            info = (f"{geo.num_poles}-pole, Slotless\n"
                    f"Magnet: {geo.magnet_grade} (Tmax={geo.magnet_max_temp:.0f}°C)\n"
                    f"Lstk={geo.stack_length} mm")

        ax.annotate(info,
                     xy=(0, -max_r * 0.88),
                     fontsize=8, ha="center", va="top",
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="lightyellow",
                               edgecolor="black", alpha=0.8))
