"""
Matplotlib-based 2D cross-section plotter for motor geometry.
Draws an interactive figure that updates when parameters change.
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Wedge, Circle, Arc
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt


# Color scheme for motor components
COLORS = {
    "shaft": "#4a4a4a",           # Dark gray
    "rotor_core": "#8B7355",      # Brown/steel
    "magnet": "#2E8B57",          # Green (common for magnets in thermal maps)
    "magnet_alt": "#006400",      # Dark green
    "airgap": "#F0F0F0",          # Light gray
    "stator_core": "#A0A0A0",     # Gray
    "stator_tooth": "#808080",    # Medium gray
    "winding": "#D4A017",         # Copper/gold
    "slot_liner": "#8B4513",      # Brown
    "housing": "#4169E1",         # Blue
    "background": "#FFFFFF",
}

PATTERNS = {
    "shaft": "",
    "rotor_core": "",
    "magnet": "",
    "stator_core": "",
    "stator_tooth": "",
    "winding": "//",
    "housing": "",
}


class MotorGeometryCanvas(FigureCanvas):
    """
    Matplotlib canvas that renders the 2D motor cross-section.
    Reusable as a Qt widget.
    """

    def __init__(self, parent=None, width=8, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor(COLORS["background"])
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111, aspect="equal")
        self.ax.set_xlabel("x [mm]")
        self.ax.set_ylabel("y [mm]")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Motor Cross-Section (Radial View)")

        self._annotations = []
        self._dim_lines = []

    def clear(self):
        """Clear the plot."""
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x [mm]")
        self.ax.set_ylabel("y [mm]")
        self.ax.grid(True, alpha=0.3)
        self._annotations = []
        self._dim_lines = []

    def draw_cross_section(self, geo_params, show_labels=True, show_dimensions=True,
                            node_temperatures=None, font_scale=1.0):
        """
        Main drawing function.
        Takes a MotorGeometryParams object and renders the cross-section.
        
        If node_temperatures is provided (dict of name -> (temp, loss)), 
        annotations are drawn on the geometry.
        """
        from .motor_geometry import MotorGeometryParams, _circle_points, _arc_points

        self.clear()
        ax = self.ax

        # Axial view - draw from outside in
        Rso = geo_params.Rso
        Rsi = geo_params.Rsi
        Rro = geo_params.Rro
        Rri = geo_params.Rri
        Rshaft = geo_params.shaft_radius
        hw = geo_params.housing_wall_thickness
        num_poles = geo_params.num_poles
        num_slots = geo_params.num_slots

        # === 1. Housing (outermost) ===
        housing = Circle((0, 0), Rso + hw, facecolor=COLORS["housing"],
                         edgecolor="black", linewidth=1.5, alpha=0.7)
        ax.add_patch(housing)

        # === 2. Stator yoke ===
        if geo_params.structure_type == "Slotless":
            stator = Circle((0, 0), Rso, facecolor=COLORS["stator_core"],
                            edgecolor="black", linewidth=1.5, alpha=0.7)
            ax.add_patch(stator)
            # Inner bore
            bore = Circle((0, 0), Rsi, facecolor=COLORS["airgap"],
                          edgecolor="black", linewidth=1.0)
            ax.add_patch(bore)
        else:
            # Slotted stator - draw yoke ring
            slot_depth = geo_params.slot_depth
            Rslot_bottom = Rsi + slot_depth

            # Stator yoke (annulus from Rslot_bottom to Rso)
            yoke = Circle((0, 0), Rso, facecolor=COLORS["stator_core"],
                          edgecolor="black", linewidth=1.5, alpha=0.7)
            ax.add_patch(yoke)
            yoke_inner = Circle((0, 0), Rslot_bottom, facecolor=COLORS["background"],
                                edgecolor="black", linewidth=1.0)
            ax.add_patch(yoke_inner)

            # Draw teeth and slots
            slot_pitch = 2 * math.pi / num_slots
            tooth_tip_half = geo_params.tooth_width_min / 2 / Rsi
            slot_opening_half = geo_params.slot_opening / 2 / Rsi

            for s in range(num_slots):
                theta_center = s * slot_pitch

                # Tooth center (between slots)
                tooth_center = theta_center + slot_pitch / 2

                # Slot region (between teeth) - draw as wedge
                t_start = tooth_center + tooth_tip_half - slot_pitch / 2
                t_end = tooth_center - tooth_tip_half + slot_pitch / 2
                # Actually: slot spans from end of one tooth to start of next
                slot_t_start = theta_center - slot_pitch / 2 + tooth_tip_half
                slot_t_end = theta_center + slot_pitch / 2 - tooth_tip_half

                # Draw slot (winding region) as a wedge
                if slot_t_end > slot_t_start:
                    slot_wedge = Wedge((0, 0), Rslot_bottom,
                                       math.degrees(slot_t_start),
                                       math.degrees(slot_t_end),
                                       width=slot_depth,
                                       facecolor=COLORS["winding"],
                                       edgecolor="black", linewidth=0.5, alpha=0.8)
                    ax.add_patch(slot_wedge)

                    # Slot opening (small gap at bore surface)
                    opening_wedge = Wedge((0, 0), Rsi + geo_params.slot_wedge_height,
                                          math.degrees(slot_t_start),
                                          math.degrees(slot_t_end),
                                          width=geo_params.slot_wedge_height,
                                          facecolor=COLORS["background"],
                                          edgecolor="black", linewidth=0.3, alpha=0.5)
                    ax.add_patch(opening_wedge)

            # Draw tooth outlines on top
            for s in range(num_slots):
                theta_center = s * slot_pitch
                tooth_center = theta_center + slot_pitch / 2
                t_start = tooth_center - tooth_tip_half
                t_end = tooth_center + tooth_tip_half

                # Tooth polygon
                Rtip = Rsi
                Rb = Rslot_bottom
                N = 16
                # Outer arc (at slot bottom)
                outer_arc_t = np.linspace(t_end, t_start, N)
                outer_x = Rb * np.cos(outer_arc_t)
                outer_y = Rb * np.sin(outer_arc_t)
                # Inner arc (at bore)
                inner_arc_t = np.linspace(t_start, t_end, N)
                inner_x = Rtip * np.cos(inner_arc_t)
                inner_y = Rtip * np.sin(inner_arc_t)

                tooth_verts = np.vstack([
                    np.column_stack([inner_x, inner_y]),
                    np.column_stack([outer_x, outer_y]),
                ])
                tooth_poly = Polygon(tooth_verts, facecolor=COLORS["stator_tooth"],
                                     edgecolor="black", linewidth=0.8, alpha=0.9)
                ax.add_patch(tooth_poly)

        # === 3. Airgap ===
        airgap = Circle((0, 0), Rsi, facecolor=COLORS["airgap"],
                        edgecolor="black", linewidth=0.5, alpha=0.3)
        ax.add_patch(airgap)
        rotor_outline = Circle((0, 0), Rro, facecolor=COLORS["background"],
                               edgecolor="black", linewidth=0.5, alpha=0.0)
        ax.add_patch(rotor_outline)

        # === 4. Magnets (segmented) ===
        pole_angle = 2 * math.pi / num_poles
        magnet_arc = pole_angle * geo_params.magnet_span_ratio
        Rmag_inner = Rro - geo_params.magnet_thickness

        for p in range(num_poles):
            theta_center = p * pole_angle
            t_start = theta_center - magnet_arc / 2
            t_end = theta_center + magnet_arc / 2

            magnet_patch = Wedge((0, 0), Rro,
                                 math.degrees(t_start),
                                 math.degrees(t_end),
                                 width=geo_params.magnet_thickness,
                                 facecolor=COLORS["magnet"],
                                 edgecolor="black", linewidth=0.8, alpha=0.85)
            ax.add_patch(magnet_patch)

            # Magnet label
            if show_labels:
                t_mid = (t_start + t_end) / 2
                r_label = Rro - geo_params.magnet_thickness / 2
                ax.annotate("Magnet",
                            (r_label * math.cos(t_mid), r_label * math.sin(t_mid)),
                            fontsize=6*font_scale, ha="center", va="center", color="white",
                            fontweight="bold")

        # === 5. Rotor core ===
        rotor_core = Circle((0, 0), Rro - geo_params.magnet_thickness,
                            facecolor=COLORS["rotor_core"],
                            edgecolor="black", linewidth=1.0, alpha=0.7)
        ax.add_patch(rotor_core)

        # === 6. Shaft ===
        shaft = Circle((0, 0), Rshaft,
                       facecolor=COLORS["shaft"],
                       edgecolor="black", linewidth=1.5)
        ax.add_patch(shaft)

        # === 7. Dimensions / Annotations ===
        if show_labels:
            # Component labels
            # Housing label
            ax.annotate("Housing", (0, Rso + hw + 3),
                        fontsize=8*font_scale, ha="center", va="bottom",
                        color=COLORS["housing"], fontweight="bold")

            # Stator label
            if geo_params.structure_type == "Slotless":
                winding_thickness = geo_params.slot_depth
                Rwinding_outer = Rsi + winding_thickness
                ax.annotate("Stator Core", (0, (Rso + Rwinding_outer) / 2),
                            fontsize=8*font_scale, ha="center", va="center",
                            color="black", fontweight="bold")
                if winding_thickness > 0:
                    ax.annotate("Winding", (0, (Rsi + Rwinding_outer) / 2),
                                fontsize=7*font_scale, ha="center", va="center",
                                color="#8B6914", fontweight="bold")
            else:
                ax.annotate("Stator Yoke", (Rso * 0.7, Rso * 0.7),
                            fontsize=7*font_scale, ha="center", color="black", fontweight="bold")

            if geo_params.structure_type != "Slotless":
                ax.annotate("Winding", (0, Rsi + geo_params.slot_depth / 2),
                            fontsize=7*font_scale, ha="center", va="center",
                            color="#8B6914", fontweight="bold")

            ax.annotate("Airgap", (Rro + (Rsi - Rro) / 2, 0),
                        fontsize=7*font_scale, ha="center", va="center",
                        color="gray", fontweight="bold", rotation=90)

            ax.annotate("Rotor Core", (0, (Rro - geo_params.magnet_thickness) / 2),
                        fontsize=7*font_scale, ha="center", va="center",
                        color="black", fontweight="bold")

            ax.annotate("Shaft", (0, 0), fontsize=7*font_scale,
                        ha="center", va="center", color="white", fontweight="bold")

        if show_dimensions:
            self._draw_dimensions(ax, geo_params, font_scale=font_scale)

        # === 8. Temperature annotations (from LPTN solver) ===
        if node_temperatures is not None and len(node_temperatures) > 0:
            self._draw_temperature_annotations(ax, geo_params, node_temperatures, font_scale=font_scale)

        # Set plot limits with some padding
        max_r = Rso + hw + 10
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.set_aspect("equal")
        ax.set_title("Motor Cross-Section (Radial View)", fontweight="bold")

        self.draw()

    def _draw_temperature_annotations(self, ax, geo, node_temps, font_scale=1.0):
        """
        Draw temperature annotations on the geometry.
        node_temps: dict of name -> (temperature, loss)
        Maps node names to radial positions on the cross-section.
        """
        Rso = geo.Rso
        Rsi = geo.Rsi
        slot_depth = getattr(geo, 'slot_depth', 10)
        Rslot_bottom = Rsi + slot_depth
        hw = geo.housing_wall_thickness
        Rro = geo.Rro

        # Define annotation positions (name -> (r, angle_deg, side_offset))
        # Positions match the component centroids for annotation
        slot_depth = getattr(geo, 'slot_depth', 10)
        Rslot_bottom = Rsi + slot_depth

        # For slotless, use winding outer instead of slot bottom
        if getattr(geo, 'structure_type', '') == "Slotless":
            Rwinding_outer = Rsi + slot_depth
        else:
            Rwinding_outer = Rslot_bottom

        positions = {
            "ambient": (Rso + hw + 6, 0, "above"),
            "housing": ((Rso + Rso + hw) / 2, 90, "out"),
            "stator_yoke": ((Rso + Rwinding_outer) / 2, 45, "mid"),
            "stator_tooth": ((Rsi + Rslot_bottom) / 2, 0, "in"),
            "slot_winding": ((Rsi + Rwinding_outer) / 2, 180, "mid"),
            "stator_tip": (Rsi, -90, "in"),
            "magnet": ((Rro + Rro - geo.magnet_thickness) / 2, -45, "mid"),
            "rotor_core": ((Rro - geo.magnet_thickness) / 2, 135, "mid"),
            "shaft": (0, -1, "center"),
            "end_winding": ((Rsi + Rwinding_outer) / 2, 225, "mid"),
        }

        for name, (temp, loss) in node_temps.items():
            if name not in positions:
                continue

            r_pos, angle_deg, _ = positions[name]
            angle_rad = math.radians(angle_deg)
            x = r_pos * math.cos(angle_rad)
            y = r_pos * math.sin(angle_rad)

            # Color-code by temperature
            if temp < 80:
                bg = "#4CAF50"  # green (cool)
            elif temp < 120:
                bg = "#FF9800"  # orange (warm)
            elif temp < 180:
                bg = "#E65100"  # dark orange (hot)
            else:
                bg = "#C62828"  # red (very hot)

            # Display name and temperature
            label = f"{name}\n{temp:.0f}°C"
            if loss > 0:
                label += f"\n{loss:.1f}W"

            ax.annotate(label, xy=(x, y),
                        fontsize=6*font_scale, ha="center", va="center",
                        color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor=bg,
                                  edgecolor="black",
                                  linewidth=0.5,
                                  alpha=0.85))

    def _draw_dimensions(self, ax, geo, font_scale=1.0):
        """Draw dimension lines and annotations on the plot."""
        # Only show a few key dimensions to avoid clutter
        max_r = geo.Rso + geo.housing_wall_thickness
        # Horizontal dimension lines at y=0 plus offset to top
        y_dim = max_r + 5

        # Stator OD
        ax.annotate(f"$R_{{so}}$={geo.Rso}",
                     xy=(geo.Rso, 0), xytext=(geo.Rso, y_dim),
                     arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                     fontsize=7*font_scale, ha="center", va="bottom")

        # Stator ID
        ax.annotate(f"$R_{{si}}$={geo.Rsi}",
                     xy=(geo.Rsi, 0), xytext=(geo.Rsi, y_dim - 3),
                     arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                     fontsize=7*font_scale, ha="center", va="bottom")

        # Rotor OD
        ax.annotate(f"$R_{{ro}}$={geo.Rro}",
                     xy=(geo.Rro, 0), xytext=(geo.Rro, y_dim - 6),
                     arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                     fontsize=7*font_scale, ha="center", va="bottom")

        # Stack length annotation
        ax.annotate(f"$L_{{stk}}$={geo.stack_length} mm",
                     xy=(0, -max_r * 0.85),
                     fontsize=8*font_scale, ha="center", va="top",
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="lightyellow",
                               edgecolor="black", alpha=0.8))

        # Pole count and magnet info
        info_str = (f"{geo.num_poles}-pole, {geo.num_slots}-slot\n"
                    f"Magnet: {geo.magnet_grade} (Tmax={geo.magnet_max_temp}°C)\n"
                    f"{geo.structure_type}")
        ax.annotate(info_str,
                     xy=(0, -max_r * 0.95),
                     fontsize=8*font_scale, ha="center", va="top",
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="lightblue",
                               edgecolor="black", alpha=0.7))


def get_geometry_data(geo_params):
    """Return structured data about the geometry for GUI display."""
    comp = geo_params.computed
    is_slotless = geo_params.structure_type == "Slotless"
    data = {
        "Stator OD": f"{geo_params.Rso:.1f} mm",
        "Stator ID (bore)": f"{geo_params.Rsi:.1f} mm",
        "Stack Length": f"{geo_params.stack_length:.1f} mm",
        "Stator Yoke Thickness": f"{comp['stator_yoke_thickness']:.2f} mm",
        "Rotor OD": f"{geo_params.Rro:.1f} mm",
        "Rotor ID": f"{geo_params.Rri:.1f} mm",
        "Rotor Back-iron": f"{comp['rotor_back_iron']:.2f} mm",
        "Airgap": f"{geo_params.airgap_length:.2f} mm",
        "Magnet Thickness": f"{geo_params.magnet_thickness:.2f} mm",
        "Magnet Arc": f"{comp['magnet_arc_angle_deg']:.1f}°",
        "Pole Pitch": f"{comp['pole_pitch_angle_deg']:.1f}°",
        "Housing Wall": f"{geo_params.housing_wall_thickness:.1f} mm",
    }
    if is_slotless:
        data["Winding Thickness"] = f"{geo_params.slot_depth:.1f} mm"
        data["Winding Inner"] = f"{geo_params.Rsi:.1f} mm"
        data["Winding Outer"] = f"{geo_params.Rsi + geo_params.slot_depth:.1f} mm"
    else:
        data["Slot Pitch"] = f"{comp['slot_pitch_angle_deg']:.1f}°"
        data["Slot Depth"] = f"{geo_params.slot_depth:.1f} mm"
        data["Slot Area"] = f"{comp['slot_area_mm2']:.2f} mm²"
        data["Fill Factor (effective)"] = f"{comp['effective_slot_fill']*100:.1f}%"
    return data
