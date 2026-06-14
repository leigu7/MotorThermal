"""
LPT-FE: Lumped Parameter Thermal Network for motor thermal analysis.
"""

from lpt_fe.node import ThermalNode, ThermalResistance, ThermalNetwork
from lpt_fe.correlations import (
    AIR_K,
    h_airgap_natural,
    h_housing_natural, h_housing_forced, h_housing_water_jacket,
    k_slot_equivalent,
    r_cylindrical_radial, r_cylindrical_axial,
    r_interface_slot_liner, r_convective,
)
from lpt_fe.geometry_mapper import (
    NetworkBuilderConfig, GeometryMapper, build_thermal_network,
)
from lpt_fe.solver import SteadyStateSolver, solve_steady_state

__all__ = [
    "ThermalNode", "ThermalResistance", "ThermalNetwork",
    "NetworkBuilderConfig", "GeometryMapper", "build_thermal_network",
    "SteadyStateSolver", "solve_steady_state",
    "h_airgap_natural", "h_housing_natural", "h_housing_forced",
    "h_housing_water_jacket", "k_slot_equivalent",
    "r_cylindrical_radial", "r_cylindrical_axial",
    "r_interface_slot_liner", "r_convective",
]

