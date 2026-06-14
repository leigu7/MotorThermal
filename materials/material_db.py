"""
Material property database for motor thermal modeling.
Properties at ~20°C unless noted; temperature-dependent coefficients included.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Material:
    """Base material properties for thermal analysis."""
    name: str
    k_radial: float           # Thermal conductivity - radial [W/m·K]
    k_axial: float             # Thermal conductivity - axial [W/m·K]
    rho: float                 # Density [kg/m³]
    cp: float                  # Specific heat capacity [J/kg·K]
    k_tangential: float = None  # Thermal conductivity - tangential [W/m·K]
    max_temp: float = 1000.0   # Maximum operating temperature [°C]
    description: str = ""
    temp_coeff_k: float = 0.0  # Temp coefficient of conductivity [1/K]

    def __post_init__(self):
        if self.k_tangential is None:
            self.k_tangential = self.k_radial

    @property
    def k_effective(self):
        """Isotropic effective conductivity for simplified models."""
        return (self.k_radial + self.k_tangential + self.k_axial) / 3


# ============================================================================
# Pre-defined materials
# ============================================================================

STEEL_M19_24GA = Material(
    name="M19 24Ga Lamination",
    k_radial=30.0,      # In-plane (radial/tangential)
    k_axial=3.0,        # Through-plane (lamination direction)
    rho=7650,
    cp=490,
    max_temp=250,
    description="M19 non-oriented silicon steel, 0.635mm laminations",
    temp_coeff_k=-0.0002,
)

STEEL_M400_50A = Material(
    name="M400-50A Lamination",
    k_radial=28.0,
    k_axial=2.5,
    rho=7700,
    cp=480,
    max_temp=250,
    description="M400-50A non-oriented silicon steel, 0.5mm",
    temp_coeff_k=-0.0002,
)

STEEL_NO20 = Material(
    name="NO20 Lamination",
    k_radial=25.0,
    k_axial=2.0,
    rho=7600,
    cp=490,
    max_temp=250,
    description="NO20 high-grade non-oriented silicon steel",
    temp_coeff_k=-0.0002,
)

COPPER_WIRE = Material(
    name="Copper (Winding)",
    k_radial=385.0,
    k_axial=385.0,
    rho=8960,
    cp=385,
    max_temp=250,  # Insulation dependent; copper melts at 1083°C
    description="Electrolytic copper wire (99.9% IACS)",
    temp_coeff_k=-0.0004,
)

# Equivalent winding conductivity (includes slot liner + impregnation)
# Significantly reduced from pure copper due to composite nature
WINDING_EQUIV = Material(
    name="Winding (Equivalent)",
    k_radial=1.2,      # Radial: through slot liner, varnish, and copper
    k_axial=385.0,     # Axial: along copper wires
    rho=5000,          # Average density (copper + insulation + air)
    cp=400,
    max_temp=200,
    description="Homogenized winding: copper + insulation + varnish + air",
)

# NdFeB Magnets - various grades
MAGNET_N35SH = Material(
    name="NdFeB N35SH",
    k_radial=8.0,
    k_axial=8.0,
    rho=7500,
    cp=460,
    max_temp=150,
    description="N35SH, Br=1.21T, Hcj=1592kA/m, Max working temp 150°C",
    temp_coeff_k=-0.001,
)

MAGNET_N40UH = Material(
    name="NdFeB N40UH",
    k_radial=8.0,
    k_axial=8.0,
    rho=7500,
    cp=460,
    max_temp=180,
    description="N40UH, Br=1.27T, Hcj=1990kA/m, Max working temp 180°C",
    temp_coeff_k=-0.001,
)

MAGNET_N42UH = Material(
    name="NdFeB N42UH",
    k_radial=8.0,
    k_axial=8.0,
    rho=7500,
    cp=460,
    max_temp=180,
    description="N42UH, Br=1.30T, Hcj=1990kA/m, Max working temp 180°C",
    temp_coeff_k=-0.001,
)

MAGNET_N45SH = Material(
    name="NdFeB N45SH",
    k_radial=8.0,
    k_axial=8.0,
    rho=7500,
    cp=460,
    max_temp=150,
    description="N45SH, Br=1.35T, Hcj=1592kA/m, Max working temp 150°C",
    temp_coeff_k=-0.001,
)

MAGNET_SMCO = Material(
    name="SmCo 2:17",
    k_radial=10.0,
    k_axial=10.0,
    rho=8400,
    cp=420,
    max_temp=300,
    description="Samarium Cobalt 2:17, high temp capability",
    temp_coeff_k=-0.0003,
)

ALUMINUM_6061 = Material(
    name="Aluminum 6061",
    k_radial=167.0,
    k_axial=167.0,
    rho=2700,
    cp=896,
    max_temp=200,
    description="Housing material, good thermal conductivity",
    temp_coeff_k=-0.0002,
)

ALUMINUM_DIE_CAST = Material(
    name="Aluminum Die Cast (ADC12)",
    k_radial=96.0,
    k_axial=96.0,
    rho=2710,
    cp=920,
    max_temp=200,
    description="ADC12 die-cast aluminum, common for motor housings",
    temp_coeff_k=-0.0002,
)

STEEL_STRUCTURAL = Material(
    name="Steel (Structural)",
    k_radial=50.0,
    k_axial=50.0,
    rho=7850,
    cp=460,
    max_temp=300,
    description="Structural steel for shaft/housing",
    temp_coeff_k=-0.0002,
)

STEEL_SHAFT = Material(
    name="Steel (Shaft)",
    k_radial=45.0,
    k_axial=45.0,
    rho=7800,
    cp=450,
    max_temp=300,
    description="Medium-carbon steel for motor shaft",
    temp_coeff_k=-0.0002,
)

SLOT_LINER = Material(
    name="Slot Liner (NMN)",
    k_radial=0.25,
    k_axial=0.25,
    rho=1200,
    cp=1500,
    max_temp=200,
    description="NMN (Nomex-Mylar-Nomex) slot insulation",
)

AIR = Material(
    name="Air",
    k_radial=0.026,
    k_axial=0.026,
    rho=1.2,
    cp=1005,
    max_temp=200,
    description="Air at 20°C, used for airgap effective conductivity",
    temp_coeff_k=0.003,
)

IMPREGNATION = Material(
    name="Varnish/Impregnation",
    k_radial=0.2,
    k_axial=0.2,
    rho=1100,
    cp=1500,
    max_temp=200,
    description="Epoxy or polyester varnish for winding impregnation",
)


# Lookup dictionaries
MATERIAL_CATALOG: Dict[str, Material] = {
    "M19_24Ga": STEEL_M19_24GA,
    "M400_50A": STEEL_M400_50A,
    "NO20": STEEL_NO20,
    "Copper": COPPER_WIRE,
    "Winding_Eq": WINDING_EQUIV,
    "N35SH": MAGNET_N35SH,
    "N40UH": MAGNET_N40UH,
    "N42UH": MAGNET_N42UH,
    "N45SH": MAGNET_N45SH,
    "SmCo": MAGNET_SMCO,
    "Al6061": ALUMINUM_6061,
    "ADC12": ALUMINUM_DIE_CAST,
    "Steel_Struct": STEEL_STRUCTURAL,
    "Shaft_Steel": STEEL_SHAFT,
    "Slot_Liner": SLOT_LINER,
    "Air": AIR,
    "Varnish": IMPREGNATION,
}

MAGNET_GRADES = {
    "N35SH": MAGNET_N35SH,
    "N40UH": MAGNET_N40UH,
    "N42UH": MAGNET_N42UH,
    "N45SH": MAGNET_N45SH,
    "SmCo 2:17": MAGNET_SMCO,
}

INSULATION_CLASSES = {
    "Class A (105°C)": 105,
    "Class E (120°C)": 120,
    "Class B (130°C)": 130,
    "Class F (155°C)": 155,
    "Class H (180°C)": 180,
    "Class N (200°C)": 200,
    "Class R (220°C)": 220,
}


def get_material(name: str) -> Optional[Material]:
    """Look up a material by name."""
    return MATERIAL_CATALOG.get(name)

def get_magnet_for_grade(grade: str) -> Optional[Material]:
    """Get material properties for a magnet grade string."""
    return MAGNET_GRADES.get(grade)

def get_insulation_temp(insulation_class: str) -> float:
    """Get max temperature for an insulation class."""
    return INSULATION_CLASSES.get(insulation_class, 155.0)
