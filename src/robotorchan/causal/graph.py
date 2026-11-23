"""Known directed acyclic graphs for causal workflows.

Inputs may be parent mappings, directed edge lists, adjacency matrices, or
optional NetworkX DiGraphs. All formats use the same validated DAG contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any


class CausalGraph:
    """A validated DAG with a read-only parent map and stable topological order.

    Node names are nonempty strings. Converters do not perform causal discovery,
    and their edges always mean cause -> effect.
    """

    def __init__(self, parents: Mapping[str, Sequence[str]]) -> None:
        nodes = tuple(parents)
        if not nodes or any(not isinstance(n, str) or not n for n in nodes):
            raise ValueError("Graph must contain nonempty string node names.")

        normalized: dict[str, tuple[str, ...]] = {}
        for node, incoming in parents.items():
            if isinstance(incoming, (str, bytes)):
                raise TypeError(f"Parents for {node} must be a sequence, not a string.")
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
        self._parents = MappingProxyType(normalized)
        self._order = tuple(order)

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[str, str]],
        *,
        nodes: Sequence[str] | None = None,
    ) -> CausalGraph:
        """Build from directed (source, target) pairs; include isolates via nodes.

        When nodes is omitted, nodes are inferred in first-appearance order.
        """
        pairs: list[tuple[str, str]] = []
        for edge in edges:
            if isinstance(edge, (str, bytes)):
                raise ValueError("Each edge must be a (source, target) pair.")
            try:
                source, target = edge
            except (TypeError, ValueError) as exc:
                raise ValueError("Each edge must be a (source, target) pair.") from exc
            if (
                not isinstance(source, str)
                or not source
                or not isinstance(target, str)
                or not target
            ):
                raise ValueError("Edge nodes must be nonempty strings.")
            pairs.append((source, target))

        if nodes is None:
            names = tuple(dict.fromkeys(node for pair in pairs for node in pair))
        else:
            if isinstance(nodes, (str, bytes)):
                raise TypeError("nodes must be a sequence of node names, not a string.")
            names = tuple(nodes)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Graph must contain nonempty string node names.")
        if len(names) != len(set(names)):
            raise ValueError("Duplicate graph nodes.")

        parents: dict[str, list[str]] = {name: [] for name in names}
        for source, target in pairs:
            if source not in parents or target not in parents:
                raise ValueError(f"Unknown node in edge: ({source}, {target}).")
            if source in parents[target]:
                raise ValueError(f"Duplicate edge: ({source}, {target}).")
            parents[target].append(source)
        return cls(parents)

    @classmethod
    def from_adjacency_matrix(
        cls,
        matrix: Any,
        *,
        nodes: Sequence[str],
    ) -> CausalGraph:
        """Build from binary A, where A[i][j] == 1 means nodes[i] -> nodes[j].

        Accepts 2D Python nested sequences, NumPy arrays, and dense torch
        tensors without importing or requiring NumPy.
        """
        if isinstance(nodes, (str, bytes)):
            raise TypeError("nodes must be a sequence of node names, not a string.")
        names = tuple(nodes)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Graph must contain nonempty string node names.")
        if len(names) != len(set(names)):
            raise ValueError("Duplicate graph nodes.")
        if isinstance(matrix, (str, bytes)):
            raise ValueError("Adjacency matrix must be square with shape [n, n].")
        try:
            rows = tuple(tuple(row) for row in matrix)
        except (TypeError, ValueError) as exc:
            raise ValueError("Adjacency matrix must be square with shape [n, n].") from exc
        n = len(names)
        if len(rows) != n or any(len(row) != n for row in rows):
            raise ValueError(f"Adjacency matrix must have shape [{n}, {n}].")

        parents: dict[str, list[str]] = {name: [] for name in names}
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                if isinstance(cell, (str, bytes)):
                    raise ValueError(f"Adjacency entry [{i}, {j}] must be 0 or 1.")
                try:
                    value = float(cell)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"Adjacency entry [{i}, {j}] must be 0 or 1.") from exc
                if value not in (0.0, 1.0):
                    raise ValueError(f"Adjacency entry [{i}, {j}] must be 0 or 1.")
                if value == 1.0:
                    parents[names[j]].append(names[i])
        return cls(parents)

    @classmethod
    def from_networkx(cls, graph: Any) -> CausalGraph:
        """Convert a NetworkX DiGraph, preserving nodes and isolated nodes."""
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "NetworkX support requires the optional dependency: pip install robotorchan[causal]"
            ) from exc
        if not isinstance(graph, nx.DiGraph) or isinstance(graph, nx.MultiDiGraph):
            raise TypeError("graph must be a networkx.DiGraph (not an undirected/multi graph).")
        return cls({node: tuple(graph.predecessors(node)) for node in graph.nodes})

    def to_edges(self) -> tuple[tuple[str, str], ...]:
        """Return directed (source, target) pairs, preserving parent order."""
        return tuple((parent, node) for node in self.nodes for parent in self.parents_of(node))

    def to_adjacency_matrix(self) -> tuple[tuple[int, ...], ...]:
        """Return a binary adjacency matrix in self.nodes order (source rows)."""
        return tuple(
            tuple(int(source in self._parents[target]) for target in self.nodes)
            for source in self.nodes
        )

    def to_networkx(self) -> Any:
        """Return an independent NetworkX DiGraph; import NetworkX on demand."""
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "NetworkX support requires the optional dependency: pip install robotorchan[causal]"
            ) from exc
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.to_edges())
        return graph

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(self._parents)

    @property
    def topological_order(self) -> tuple[str, ...]:
        return self._order

    def parents_of(self, node: str) -> tuple[str, ...]:
        return self._parents[node]
