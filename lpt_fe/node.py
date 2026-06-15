"""
Thermal network data structures for Lumped Parameter Thermal Network (LPTN).
Defines nodes (capacitances) and resistances (conduction/convection paths).
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np


@dataclass
class ThermalNode:
    """
    A single thermal node representing a lumped mass.
    """
    name: str
    index: int
    volume: float                    # [m³]
    density: float                   # [kg/m³]
    cp: float                        # [J/kg·K]
    temperature: float = 40.0        # [°C]
    loss: float = 0.0                # [W]
    loss_temperature_dependent: bool = False
    loss_ref_temp: float = 20.0
    loss_alpha: float = 0.00393
    loss_user_override: Optional[float] = None
    fixed_temperature: Optional[float] = None

    def __post_init__(self):
        if self.fixed_temperature is not None:
            self.temperature = self.fixed_temperature

    @property
    def capacitance(self) -> float:
        return self.volume * self.density * self.cp

    @property
    def effective_loss(self) -> float:
        if self.loss_user_override is not None:
            return self.loss_user_override
        if not self.loss_temperature_dependent:
            return self.loss
        return self.loss * (1 + self.loss_alpha * (self.temperature - self.loss_ref_temp))

    def brief(self) -> str:
        return f"{self.name} ({self.index}): T={self.temperature:.1f}°C, P={self.effective_loss:.1f}W"


@dataclass
class ThermalResistance:
    """
    A thermal resistance connecting two nodes.
    """
    name: str
    node_from: int
    node_to: int
    resistance: float                # [K/W]
    resistance_type: str = "conduction"
    effective_length: float = 0.0    # [m]
    effective_area: float = 0.0      # [m²]
    conductivity: float = 0.0        # [W/m·K]
    h_coefficient: float = 0.0       # [W/m²K] for convection (inferred or stored)
    liner_thickness: float = 0.0     # [m] slot liner thickness
    liner_conductivity: float = 0.0  # [W/mK] slot liner thermal conductivity
    user_override: Optional[float] = None

    @property
    def effective_resistance(self) -> float:
        if self.user_override is not None:
            return self.user_override
        return self.resistance

    def brief(self) -> str:
        return f"{self.name}: R={self.effective_resistance:.4f} K/W ({self.resistance_type})"


@dataclass
class ThermalNetwork:
    """
    Complete thermal network: nodes + resistances.
    """
    nodes: List[ThermalNode]
    resistances: List[ThermalResistance]
    name: str = "Motor Thermal Network"

    def __post_init__(self):
        self._validate()

    def _validate(self):
        n_nodes = len(self.nodes)
        for r in self.resistances:
            if r.node_from < 0 or r.node_from >= n_nodes:
                raise ValueError(f"R '{r.name}': node_from={r.node_from} out of range")
            if r.node_to < 0 or r.node_to >= n_nodes:
                raise ValueError(f"R '{r.name}': node_to={r.node_to} out of range")

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def assemble_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n_nodes
        G = np.zeros((n, n))
        P = np.zeros(n)

        for res in self.resistances:
            R = res.effective_resistance
            if R <= 0:
                continue
            G_val = 1.0 / R
            i, j = res.node_from, res.node_to
            G[i, i] += G_val
            G[i, j] -= G_val
            G[j, i] -= G_val
            G[j, j] += G_val

        for node in self.nodes:
            P[node.index] += node.effective_loss

        for node in self.nodes:
            if node.fixed_temperature is not None:
                idx = node.index
                G[idx, :] = 0.0
                G[idx, idx] = 1.0
                P[idx] = node.fixed_temperature

        return G, P

    def get_node_by_name(self, name: str) -> Optional[ThermalNode]:
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    def get_temperatures(self) -> Dict[str, float]:
        return {n.name: n.temperature for n in self.nodes}

    def print_summary(self):
        print(f"\n=== {self.name} ===")
        print(f"{'Node':25s} {'T [°C]':10s} {'Loss [W]':10s}")
        print("-" * 45)
        for n in self.nodes:
            print(f"{n.name:25s} {n.temperature:8.1f}°C  {n.effective_loss:8.2f}W")
        print("=" * 45)

    def to_dataframe(self):
        rows = []
        for n in self.nodes:
            rows.append({
                "Node": n.name,
                "Temperature [°C]": round(n.temperature, 1),
                "Loss [W]": round(n.effective_loss, 2),
                "Volume [cm³]": round(n.volume * 1e6, 2),
            })
        return rows

