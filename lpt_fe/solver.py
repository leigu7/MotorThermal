"""
Solver for Lumped Parameter Thermal Network.
Supports steady-state direct solve and iterative solve with temperature-dependent losses.
"""

import logging
from typing import Optional, Dict, List
import numpy as np

from lpt_fe.node import ThermalNetwork

logger = logging.getLogger(__name__)


class SteadyStateSolver:
    """
    Solves the steady-state thermal network [G]·{T} = {P}.
    For temperature-dependent losses, iterates until convergence.
    """

    def __init__(self, network: ThermalNetwork,
                 max_iterations: int = 50,
                 convergence_tolerance: float = 0.1,
                 relaxation: float = 0.5):
        self.network = network
        self.max_iterations = max_iterations
        self.tolerance = convergence_tolerance
        self.relaxation = max(0.1, min(1.0, relaxation))
        self.converged = False
        self.iteration_count = 0
        self._history: Dict[str, list] = {}

    def solve(self) -> np.ndarray:
        n = self.network.n_nodes
        has_temp_dependent = any(
            n.loss_temperature_dependent and n.loss > 0
            for n in self.network.nodes
        )
        if not has_temp_dependent:
            return self._direct_solve()

        T = np.full(n, 40.0)
        prev_delta = 0.0
        divergence_count = 0
        max_temp_history = []

        for self.iteration_count in range(1, self.max_iterations + 1):
            for node in self.network.nodes:
                node.temperature = T[node.index]

            G, P = self.network.assemble_matrices()
            T_new = np.linalg.solve(G, P)

            if self.relaxation < 1.0:
                T_new = self.relaxation * T_new + (1 - self.relaxation) * T

            delta = np.max(np.abs(T_new - T))
            max_temp = np.max(T_new)
            max_temp_history.append(max_temp)

            # Divergence detection: temperatures consistently increasing
            # past a reasonable threshold
            if max_temp > 500 and self.iteration_count >= 5:
                # Check if temperatures keep rising (monotonic over last 4 iters)
                if len(max_temp_history) >= 4:
                    increasing = all(
                        max_temp_history[i] > max_temp_history[i-1]
                        for i in range(-4, 0)
                    )
                    if increasing and max_temp > 1000:
                        divergence_count += 1
                    else:
                        divergence_count = 0
            else:
                divergence_count = 0

            if divergence_count >= 3:
                logger.warning(
                    f"Diverging at iteration {self.iteration_count} "
                    f"(T_max={max_temp:.0f}C, delta={delta:.0f}C). Thermal runaway."
                )
                self.converged = False
                break

            for node in self.network.nodes:
                if node.name not in self._history:
                    self._history[node.name] = []
                self._history[node.name].append(T_new[node.index])

            T = T_new

            if delta < self.tolerance:
                self.converged = True
                break

        for node in self.network.nodes:
            node.temperature = T[node.index]

        # Attach solver info to network for downstream use
        self.network.solver_converged = self.converged
        self.network.solver_iterations = self.iteration_count
        self.network.solver_max_temp = float(np.max(T))

        if not self.converged:
            logger.warning(
                "Solver not converged after %d iters. delta=%.1f C > %.1f C",
                self.iteration_count, delta, self.tolerance
            )
        return T

    def _direct_solve(self) -> np.ndarray:
        G, P = self.network.assemble_matrices()
        T = np.linalg.solve(G, P)
        for node in self.network.nodes:
            node.temperature = T[node.index]
        self.converged = True
        self.iteration_count = 1
        return T

    @property
    def summary(self) -> str:
        if not self.converged:
            return "Not solved"
        return f"Steady-state: {self.iteration_count} iterations, converged={self.converged}"


def solve_steady_state(network: ThermalNetwork,
                       max_iterations: int = 50,
                       tolerance: float = 0.1,
                       relaxation: float = 0.5) -> np.ndarray:
    solver = SteadyStateSolver(network, max_iterations, tolerance, relaxation)
    return solver.solve()
