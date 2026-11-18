"""Deterministic structural causal models with user-supplied equations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
from types import MappingProxyType

from robotorchan.causal.graph import CausalGraph
from robotorchan.causal.intervention import HardIntervention

StructuralEquation = Callable[[Mapping[str, float]], float]


class StructuralCausalModel:
    """Evaluate a known-DAG deterministic SCM with hard interventions.

    Each equation receives a read-only map of its declared parent values.
    No causal discovery, stochastic exogenous noise, or effect identification
    from observational data is performed.
    """

    def __init__(
        self,
        graph: CausalGraph,
        equations: Mapping[str, StructuralEquation],
    ) -> None:
        if set(equations) != set(graph.nodes):
            raise ValueError("Provide exactly one structural equation per node.")
        if not all(callable(equation) for equation in equations.values()):
            raise TypeError("Structural equations must be callable.")
        self.graph = graph
        self._equations = MappingProxyType(dict(equations))

    def evaluate(
        self,
        intervention: HardIntervention | None = None,
    ) -> dict[str, float]:
        assigned = {} if intervention is None else dict(intervention.values)
        unknown = set(assigned) - set(self.graph.nodes)
        if unknown:
            raise ValueError(f"Unknown intervention nodes: {sorted(unknown)}")
        values: dict[str, float] = {}
        for node in self.graph.topological_order:
            if node in assigned:
                value = float(assigned[node])
            else:
                parents = {p: values[p] for p in self.graph.parents_of(node)}
                value = float(self._equations[node](MappingProxyType(parents)))
            if not isfinite(value):
                raise ValueError(f"Non-finite value for node {node}.")
            values[node] = value
        return values
