"""Known directed acyclic graphs for causal workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class CausalGraph:
    """A validated, immutable DAG with a deterministic topological ordering."""

    def __init__(self, parents: Mapping[str, Sequence[str]]) -> None:
        nodes = tuple(parents)
        if not nodes or any(not isinstance(n, str) or not n for n in nodes):
            raise ValueError("Graph must contain nonempty string node names.")
        if len(nodes) != len(set(nodes)):
            raise ValueError("Node names must be unique.")
        normalized: dict[str, tuple[str, ...]] = {}
        for node, incoming in parents.items():
            edges = tuple(incoming)
            if len(edges) != len(set(edges)):
                raise ValueError(f"Duplicate parents for {node}.")
            if any(parent not in parents for parent in edges):
                raise ValueError(f"Unknown parent for {node}.")
            if node in edges:
                raise ValueError(f"Self-loop at {node}.")
            normalized[node] = edges
        order: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError("Causal graph must be acyclic.")
            if node in visited:
                return
            visiting.add(node)
            for parent in normalized[node]:
                visit(parent)
            visiting.remove(node)
            visited.add(node)
            order.append(node)

        for node in nodes:
            visit(node)
        self._parents = normalized
        self._order = tuple(order)

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(self._parents)

    @property
    def topological_order(self) -> tuple[str, ...]:
        return self._order

    def parents_of(self, node: str) -> tuple[str, ...]:
        return self._parents[node]
