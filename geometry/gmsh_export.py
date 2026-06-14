"""
Gmsh .geo file generator from MotorGeometryParams.
Creates a 2D cross-section (or 3D extrusion) geometry for FEA meshing.
"""

import math
import os
from typing import Dict, Optional
from .motor_geometry import MotorGeometryParams


def generate_gmsh_geo(geo: MotorGeometryParams,
                      material_assignments: Optional[Dict[str, str]] = None,
                      mesh_size: float = 1.0,
                      extrude_3d: bool = False) -> str:
    """
    Generate a Gmsh .geo script from the motor geometry.

    Args:
        geo: Motor geometry parameters
        material_assignments: Dict of component -> material name
        mesh_size: Characteristic mesh length [mm]
        extrude_3d: If True, extrude the 2D cross-section into 3D

    Returns:
        String containing the .geo file contents
    """
    if material_assignments is None:
        material_assignments = {}

    lines = []
    lines.append("// =========================================")
    lines.append("// Motor Thermal Model - Gmsh Geometry")
    lines.append(f"// {geo.structure_type} | {geo.num_poles}-pole | "
                 f"{geo.num_slots}-slot")
    lines.append("// =========================================")
    lines.append("")
    lines.append(f"// Mesh size control")
    lines.append(f"Mesh.CharacteristicLengthMax = {mesh_size};")
    lines.append(f"Mesh.CharacteristicLengthMin = {mesh_size * 0.3};")
    lines.append("")

    # Define geometry using Gmsh built-in geometry kernel (not CAD kernel)
    # We'll use a simpler approach: define points, lines, and plane surfaces

    Rso = geo.Rso
    Rsi = geo.Rsi
    Rro = geo.Rro
    Rri = geo.Rri
    Rshaft = geo.shaft_radius
    hw = geo.housing_wall_thickness
    num_poles = geo.num_poles

    # For periodic sector approach (1 pole pitch)
    sector_angle = 2 * math.pi / num_poles
    sector_deg = 360.0 / num_poles

    # Define key points for one pole sector
    pt_counter = [1]
    line_counter = [1]
    surf_counter = [1]

    def add_point(x, y, z=0, lc=mesh_size):
        n = pt_counter[0]
        lines.append(f"Point({n}) = {{{x}, {y}, {z}, {lc}}};")
        pt_counter[0] += 1
        return n

    def add_line(p1, p2):
        n = line_counter[0]
        lines.append(f"Line({n}) = {{{p1}, {p2}}};")
        line_counter[0] += 1
        return n

    def add_circle_arc(p_start, p_center, p_end):
        n = line_counter[0]
        lines.append(f"Circle({n}) = {{{p_start}, {p_center}, {p_end}}};")
        line_counter[0] += 1
        return n

    def add_curve_loop(*edges):
        n = line_counter[0]
        edges_str = ", ".join(str(e) for e in edges)
        lines.append(f"Curve Loop({n}) = {{{edges_str}}};")
        line_counter[0] += 1
        return n

    def add_plane_surface(loop):
        n = surf_counter[0]
        lines.append(f"Plane Surface({n}) = {{{loop}}};")
        surf_counter[0] += 1
        return n

    # Origin
    O = add_point(0, 0, 0)

    # Create a single pole sector
    theta_start = -sector_angle / 2
    theta_end = sector_angle / 2

    def polar_point(r, theta_rad):
        """Return (x, y) from polar coordinates."""
        return (r * math.cos(theta_rad), r * math.sin(theta_rad))

    # Radii list for all concentric boundaries
    radii = [
        ("shaft", 0, Rshaft),
        ("rotor_core", Rshaft, Rro - geo.magnet_thickness),
        ("magnet", Rro - geo.magnet_thickness, Rro),
        ("airgap", Rro, Rsi),
    ]

    if geo.structure_type == "Slotless":
        radii.append(("stator", Rsi, Rso))
    else:
        radii.append(("stator_yoke", Rsi + geo.slot_depth, Rso))
        # Slot region will need special handling

    radii.append(("housing", Rso, Rso + hw))

    # Generate concentric regions as plane surfaces
    surfaces = {}
    prev_outer_loop = None

    for i, (name, r_in, r_out) in enumerate(radii):
        if r_in == 0:
            # Inner circle (shaft)
            # Points on circle
            p1 = add_point(r_out * math.cos(theta_start),
                           r_out * math.sin(theta_start))
            p2 = add_point(r_out * math.cos(theta_end),
                           r_out * math.sin(theta_end))
            # Arc from p1 to p2 (counterclockwise)
            arc = add_circle_arc(p1, O, p2)
            # Line from p2 back to p1 along sector edges
            l1 = add_line(p2, p1)
            loop = add_curve_loop(arc, l1)
            surf = add_plane_surface(loop)
            surfaces[name] = surf
            prev_outer_loop = None
        else:
            # Annular sector
            # Inner arc points
            p1_in = add_point(r_in * math.cos(theta_start),
                              r_in * math.sin(theta_start))
            p2_in = add_point(r_in * math.cos(theta_end),
                              r_in * math.sin(theta_end))
            # Outer arc points
            p1_out = add_point(r_out * math.cos(theta_start),
                               r_out * math.sin(theta_start))
            p2_out = add_point(r_out * math.cos(theta_end),
                               r_out * math.sin(theta_end))

            # Center point for arcs
            O_center = O

            # Inner arc (from p2_in to p1_in, CCW)
            inner_arc = add_circle_arc(p2_in, O_center, p1_in)
            # Outer arc (from p1_out to p2_out, CCW)
            outer_arc = add_circle_arc(p1_out, O_center, p2_out)
            # Radial lines
            l1 = add_line(p1_in, p1_out)
            l2 = add_line(p2_out, p2_in)

            loop = add_curve_loop(inner_arc, l1, outer_arc, l2)
            surf = add_plane_surface(loop)
            surfaces[name] = surf

    # Add physical groups for each material region
    lines.append("")
    lines.append("// Physical groups for thermal simulation")
    for name, surf_id in surfaces.items():
        mat_name = material_assignments.get(name, "Unknown")
        # Sanitize name for Gmsh
        clean_name = name.replace(" ", "_")
        lines.append(f'Physical Surface("{clean_name}") = {{{surf_id}}};')

    lines.append("")
    lines.append("// Mesh generation")
    lines.append("Mesh 2;")

    if extrude_3d:
        stack = geo.stack_length
        lines.append("")
        lines.append("// 3D extrusion")
        for name, surf_id in surfaces.items():
            clean_name = name.replace(" ", "_")
            lines.append(
                f'Extrude {{0, 0, {stack}}} {{'
                f'  Surface{{{surf_id}}}; Layers{{10}}; '
                f'  Recombine;'
                f'}}'
            )

    return "\n".join(lines)


def generate_gmsh_mesh(geo: MotorGeometryParams,
                       material_assignments: Optional[Dict[str, str]] = None,
                       filename: str = "motor_temp.geo",
                       mesh_size: float = 1.0,
                       run_gmsh: bool = False) -> str:
    """
    Generate and optionally run Gmsh to create a mesh file.

    Returns the path to the generated .geo file (and .msh if run_gmsh=True).
    """
    geo_code = generate_gmsh_geo(geo, material_assignments, mesh_size)

    with open(filename, "w") as f:
        f.write(geo_code)

    if run_gmsh:
        import gmsh
        gmsh.initialize()
        gmsh.open(filename)
        gmsh.model.mesh.generate(2)
        msh_file = filename.replace(".geo", ".msh")
        gmsh.write(msh_file)
        gmsh.finalize()
        return msh_file

    return filename
