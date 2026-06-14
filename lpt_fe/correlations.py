"""
Thermal correlations for LPTN.
Provides heat transfer coefficients and equivalent conductivities
for airgap convection, housing convection, slot equivalent conductivity, etc.
"""

import math

# Air properties at ~50°C (typical motor internal temperature)
AIR_K = 0.028         # [W/m·K] thermal conductivity
AIR_NU = 1.9e-5       # [m²/s] kinematic viscosity
AIR_PR = 0.71         # Prandtl number
AIR_BETA = 0.0031     # [1/K] thermal expansion coefficient


def h_airgap_natural(Rro: float, gap: float, Lstk: float,
                     speed_rpm: float, T_surface: float = 80.0) -> float:
    """
    Airgap heat transfer coefficient [W/m²K].
    
    Uses Taylor number correlation for rotating concentric cylinders.
    
    Parameters:
        Rro: Rotor outer radius [m]
        gap: Airgap radial length [m] (Rsi - Rro)
        Lstk: Stack length [m]
        speed_rpm: Rotor speed [RPM]
        T_surface: Average surface temperature [°C] (for fluid props)
    
    Returns:
        h: Heat transfer coefficient [W/m²K]
    
    References:
        - Gazley (1958): for smooth concentric cylinders
        - Becker & Kaye (1962): Taylor vortex flow correlation
    """
    # Angular velocity [rad/s]
    omega = speed_rpm * 2 * math.pi / 60.0
    
    # Taylor number
    # Ta = (omega^2 * Rro * gap^3) / nu^2
    Ta = (omega ** 2) * Rro * (gap ** 3) / (AIR_NU ** 2)
    
    # Critical Taylor number for onset of vortices
    Ta_critical = 1700
    
    # Nusselt number correlation
    if Ta < Ta_critical:
        # Laminar: pure conduction across gap
        Nu = 2.0
    elif Ta < 1e6:
        # Taylor vortex flow
        Nu = 0.409 * (Ta ** 0.241)
    else:
        # Turbulent
        Nu = 0.135 * (Ta ** 0.333)
    
    # Effective heat transfer coefficient
    # Using gap as characteristic length: h = Nu * k / gap
    h = Nu * AIR_K / gap
    
    # Apply aspect ratio correction (short motors have different behavior)
    aspect = Lstk / gap if gap > 0 else 100
    if aspect < 50:
        # Short motor: reduced effective convection
        h *= max(0.7, aspect / 100)
    
    return h


def h_airgap_pumping(Rro: float, gap: float, speed_rpm: float) -> float:
    """
    Enhanced airgap convection with rotor pumping effect.
    Used when there is axial airflow through the airgap.
    
    This is a simplified correlation for through-ventilation.
    """
    # Base Taylor correlation
    h_base = h_airgap_natural(Rro, gap, 0.1, speed_rpm)
    
    # Pumping enhancement (up to 3x for high speeds with axial flow)
    # Simplified: no axial velocity known, so estimate from speed
    enhancement = 1.0  # default: no enhancement
    
    if speed_rpm > 3000:
        # At high speeds, rotor pumping can enhance convection
        enhancement = 1.0 + 0.3 * math.log10(speed_rpm / 3000)
    
    return h_base * min(enhancement, 3.0)


# ============================================================================
# Housing-to-Ambient Correlations
# ============================================================================

def h_housing_natural(R_housing: float, Lstk: float,
                      T_housing: float = 70.0, T_ambient: float = 40.0) -> float:
    """
    Natural convection from housing surface [W/m²K].
    
    Uses Churchill-Chu correlation for vertical cylinder.
    
    Parameters:
        R_housing: Housing outer radius [m]
        Lstk: Stack length [m] (≈ height of vertical surface)
        T_housing: Housing surface temperature [°C]
        T_ambient: Ambient temperature [°C]
    """
    T_film = (T_housing + T_ambient) / 2 + 273.15  # film temperature [K]
    delta_T = T_housing - T_ambient
    
    # Rayleigh number: Ra = g * beta * delta_T * L^3 / (nu * alpha)
    # Using Lstk as characteristic length for vertical cylinder
    g = 9.81
    beta = 1.0 / T_film
    
    # Air properties at film temp (simplified, using constants at 50°C)
    nu = AIR_NU
    alpha = nu / AIR_PR
    
    if Lstk <= 0:
        Lstk = 0.01
    
    Ra = g * beta * delta_T * (Lstk ** 3) / (nu * alpha)
    
    # Churchill-Chu correlation for vertical plate/cylinder
    # Nu = (0.825 + 0.387 * Ra^(1/6) / (1 + (0.492/Pr)^(9/16))^(8/27))^2
    Pr = AIR_PR
    denom = (1 + (0.492 / Pr) ** (9/16)) ** (8/27)
    
    if Ra < 1e-5:
        Nu = 1.0
    else:
        Nu = (0.825 + 0.387 * (Ra ** (1/6)) / denom) ** 2
    
    h = Nu * AIR_K / Lstk
    
    # Ensure minimum natural convection
    return max(h, 5.0)


def h_housing_forced(R_housing: float, air_speed: float = 2.0) -> float:
    """
    Forced convection from housing surface [W/m²K].
    
    For TEFC (fan-cooled) motors.
    
    Parameters:
        R_housing: Housing outer radius [m]
        air_speed: External air speed over housing [m/s] (default 2 m/s ≈ light fan)
    """
    if air_speed <= 0:
        return h_housing_natural(R_housing, 0.1)
    
    # Characteristic length: housing diameter
    D = 2 * R_housing
    
    # Reynolds number
    Re = air_speed * D / AIR_NU
    
    # Zukauskas correlation for flow across cylinder
    if Re < 1e2:
        Nu = 0.75 * (Re ** 0.4) * (AIR_PR ** 0.37)
    elif Re < 1e3:
        Nu = 0.51 * (Re ** 0.5) * (AIR_PR ** 0.37)
    else:
        Nu = 0.26 * (Re ** 0.6) * (AIR_PR ** 0.37)
    
    h = Nu * AIR_K / D
    
    return h


def h_housing_water_jacket(flow_rate: float = 10.0,
                           channel_hydraulic_diameter: float = 0.008,
                           T_coolant: float = 50.0) -> float:
    """
    Forced convection inside water jacket channels [W/m²K].
    
    Parameters:
        flow_rate: Coolant flow rate [L/min]
        channel_hydraulic_diameter: Channel equivalent diameter [m]
        T_coolant: Coolant temperature [°C]
    
    Returns:
        h: Heat transfer coefficient [W/m²K] (typically 1000-5000)
    """
    # Convert flow rate to velocity
    # Assume typical cross-sectional area of water jacket ~ 100 mm² per channel
    area_per_channel = 1e-4  # [m²] typical
    Q = flow_rate / 1000 / 60  # [m³/s]
    velocity = Q / area_per_channel
    
    if velocity < 0.01:
        return 500  # minimum natural
    
    # Water properties at ~50°C
    water_k = 0.64      # [W/m·K]
    water_nu = 5.5e-7   # [m²/s]
    water_Pr = 3.55
    
    Re = velocity * channel_hydraulic_diameter / water_nu
    
    # Dittus-Boelter correlation (turbulent, heating)
    if Re > 4000:
        Nu = 0.023 * (Re ** 0.8) * (water_Pr ** 0.4)
    elif Re > 2300:
        # Transition
        Nu = 0.023 * (Re ** 0.8) * (water_Pr ** 0.4) * 0.5
    else:
        # Laminar: fully developed, constant heat flux
        Nu = 4.36
    
    h = Nu * water_k / channel_hydraulic_diameter
    return h


# ============================================================================
# Slot Equivalent Conductivity (Homogenized Winding)
# ============================================================================

def k_slot_equivalent(fill_factor: float,
                      k_copper: float = 385.0,
                      k_enamel: float = 0.25,
                      k_impregnation: float = 0.2,
                      k_slot_liner: float = 0.25,
                      num_layers: int = 2) -> dict:
    """
    Compute equivalent thermal conductivity of a slot winding.

    Uses the Hashin-Shtrikman bounds for a two-phase composite
    (copper + impregnation), then applies a series insulation correction.

    References:
        - Simpson et al. (2013), "Estimation of Equivalent Thermal Parameters
          of Electrical Windings", IEEE Trans. Ind. Appl.

    Parameters:
        fill_factor: Fraction of slot filled with copper (0.3-0.6)
        k_copper: Copper conductivity [W/m·K]
        k_enamel: Wire enamel conductivity [W/m·K]
        k_impregnation: Varnish/impregnation conductivity [W/m·K]
        k_slot_liner: Slot liner paper conductivity [W/m·K]
        num_layers: Winding layers in radial direction

    Returns:
        dict with 'k_radial', 'k_tangential', 'k_axial', 'k_effective', 'description'
    """
    # Volume fractions
    f_cu = fill_factor
    f_imp = 1.0 - f_cu  # everything else is impregnation + enamel
    
    # === Axial direction: parallel model (heat flows along copper wires) ===
    # Copper wires run axially, so axial conductivity is high
    k_axial = f_cu * k_copper + f_imp * k_impregnation
    
    # === Radial/Tangential direction: Hashin-Shtrikman upper bound ===
    # This gives the effective conductivity of a two-phase composite
    # where the continuous phase is impregnation and dispersed is copper
    #
    # k_eff = k_imp * (1 + 2*f_cu*(k_cu - k_imp)/(2*k_imp + k_cu - f_cu*(k_cu - k_imp)))
    
    k_cont = k_impregnation  # continuous phase (varnish fills gaps)
    k_disp = k_copper        # dispersed phase (copper wires)
    
    denom = 2 * k_cont + k_disp - f_cu * (k_disp - k_cont)
    if abs(denom) < 1e-10:
        k_radial_base = k_cont
    else:
        k_radial_base = k_cont * (1 + 2 * f_cu * (k_disp - k_cont) / denom)
    
    # Apply empirical correction for real windings based on measurements:
    # - Round wires have point contacts (not full area)
    # - Enamel coating adds a barrier
    # - Literature shows k_radial ≈ 0.5-2.0 W/mK for typical windings
    
    # Gap reduction factor: round wires have ~78% of the contact area of square wires
    geometry_factor = 0.78
    
    # Insulation barrier factor: each layer adds resistance
    # R_ins = N_layers * t_enamel / (k_enamel * A)
    # This effectively reduces k by factor ~ (1 + N_layers * k_eff * t_enamel / (k_enamel * L))
    t_enamel_ratio = 0.02  # enamel thickness / wire diameter (~2%)
    insulation_factor = 1.0 / (1.0 + num_layers * t_enamel_ratio * k_radial_base / k_enamel)
    
    k_radial = k_radial_base * geometry_factor * insulation_factor
    k_tangential = k_radial  # same in tangential
    
    # Clamp to realistic range
    k_radial = max(0.3, min(k_radial, 5.0))
    k_tangential = k_tangential
    
    # Effective (isotropic average for slot homogenization)
    k_effective = (k_radial + k_tangential + k_axial) / 3
    
    return {
        "k_radial": round(k_radial, 3),
        "k_tangential": round(k_tangential, 3),
        "k_axial": round(k_axial, 1),
        "k_effective": round(k_effective, 3),
        "description": (
            f"Fill={fill_factor:.2f}, Layers={num_layers}, "
            f"k_rad={k_radial:.3f}, k_ax={k_axial:.1f} W/mK"
        )
    }


# ============================================================================
# Interface / Contact Resistances
# ============================================================================

def r_interface_slot_liner(Lstk: float, slot_width: float,
                           liner_thickness: float = 0.0003,
                           k_liner: float = 0.25) -> float:
    """
    Thermal resistance of slot liner [K/W].
    
    Parameters:
        Lstk: Stack length [m]
        slot_width: Slot width at contact [m] (e.g., Bs2 for bottom)
        liner_thickness: Slot liner thickness [m] (default 0.3mm)
        k_liner: Liner thermal conductivity [W/m·K] (Nomex ~0.25)
    
    Returns:
        R: Thermal resistance [K/W]
    """
    area = Lstk * slot_width
    if area <= 0:
        return 1e6
    return liner_thickness / (k_liner * area)


def r_interface_press_fit(R_inner: float, R_outer: float, Lstk: float,
                          gap_thickness: float = 5e-5,
                          k_fill: float = 0.1) -> float:
    """
    Interference fit thermal resistance [K/W].
    
    Parameters:
        R_inner: Inner radius of interface [m]
        R_outer: Outer radius [m]
        Lstk: Stack length [m]
        gap_thickness: Effective gap at interface [m] (default 50 µm)
        k_fill: Conductivity of gap filler (air ~0.026, thermal paste ~3)
    """
    # Average radius of interface
    R_avg = (R_inner + R_outer) / 2
    
    # Contact area (cylindrical)
    area = 2 * math.pi * R_avg * Lstk
    
    if area <= 0:
        return 1e6
    
    return gap_thickness / (k_fill * area)


# ============================================================================
# Cylindrical Conduction Resistance
# ============================================================================

def r_cylindrical_radial(r_inner: float, r_outer: float,
                         Lstk: float, k: float,
                         sector_fraction: float = 1.0) -> float:
    """
    Radial conduction resistance through a hollow cylinder [K/W].
    
    R = ln(r_outer / r_inner) / (2 * pi * Lstk * k * sector_fraction)
    
    Parameters:
        r_inner: Inner radius [m]
        r_outer: Outer radius [m]
        Lstk: Axial length [m]
        k: Thermal conductivity [W/m·K]
        sector_fraction: Fraction of full circle (1.0 = full, 1/6 = 1 pole of 6-pole)
    """
    if r_inner <= 0 or r_outer <= r_inner:
        return 0.0  # no resistance (same node or degenerate)
    if Lstk <= 0 or k <= 0:
        return 1e6
    
    denom = 2 * math.pi * Lstk * k * sector_fraction
    if denom <= 0:
        return 1e6
    
    return math.log(r_outer / r_inner) / denom


def r_cylindrical_axial(Lstk: float, area: float, k: float) -> float:
    """
    Axial conduction resistance through a solid cylinder [K/W].
    
    R = L / (A * k)
    
    Parameters:
        Lstk: Length in axial direction [m]
        area: Cross-sectional area [m²]
        k: Thermal conductivity [W/m·K]
    """
    if area <= 0 or k <= 0:
        return 1e6
    return Lstk / (area * k)


# ============================================================================
# Convective Resistance
# ============================================================================

def r_convective(area: float, h: float) -> float:
    """
    Convective thermal resistance [K/W].
    
    R = 1 / (h * A)
    """
    if area <= 0 or h <= 0:
        return 1e6
    return 1.0 / (h * area)
