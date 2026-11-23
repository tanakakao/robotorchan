"""Phase 2: accepted known-DAG formats and backward-compatible graph behavior."""

import sys

import pytest
import torch

from robotorchan.causal import CausalGraph, HardIntervention, StructuralCausalModel


def _assert_same_dag(graph: CausalGraph) -> None:
    assert graph.nodes == ("X", "Z", "Y", "isolated")
    assert graph.parents_of("X") == ()
    assert graph.parents_of("Z") == ("X",)
    assert graph.parents_of("Y") == ("X", "Z")
    assert graph.parents_of("isolated") == ()
    assert set(graph.to_edges()) == {("X", "Z"), ("X", "Y"), ("Z", "Y")}
    assert graph.to_adjacency_matrix() == (
        (0, 1, 1, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )


def test_parent_dict_edge_list_and_matrix_agree() -> None:
    names = ("X", "Z", "Y", "isolated")
    parents = {"X": (), "Z": ("X",), "Y": ("X", "Z"), "isolated": ()}
    edges = [("X", "Z"), ("X", "Y"), ("Z", "Y")]
    adjacency = [
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    for graph in (
        CausalGraph(parents),
        CausalGraph.from_edges(edges, nodes=names),
        CausalGraph.from_adjacency_matrix(adjacency, nodes=names),
    ):
        _assert_same_dag(graph)
        assert (
            CausalGraph.from_edges(graph.to_edges(), nodes=graph.nodes).to_adjacency_matrix()
            == graph.to_adjacency_matrix()
        )
        assert CausalGraph.from_adjacency_matrix(
            graph.to_adjacency_matrix(), nodes=graph.nodes
        ).to_edges() == graph.to_edges()


def test_edges_infer_nodes_and_keep_explicit_isolates() -> None:
    inferred = CausalGraph.from_edges(edge for edge in [("Y", "X"), ("X", "Z")])
    assert inferred.nodes == ("Y", "X", "Z")
    assert inferred.topological_order == ("Y", "X", "Z")
    empty = CausalGraph.from_edges([], nodes=("A", "B"))
    assert empty.to_edges() == ()
    assert empty.to_adjacency_matrix() == ((0, 0), (0, 0))


@pytest.mark.parametrize(
    ("edges", "nodes", "error"),
    [
        ([("A", "B"), ("A", "B")], None, "Duplicate edge"),
        ([("A", "B")], ("A",), "Unknown node"),
        ([("A", "B")], ("A", "A", "B"), "Duplicate graph nodes"),
        ([("A", "B")], ("A", 1, "B"), "nonempty string"),
        ([("A", "A")], None, "Self-loop"),
        ([("A", "B"), ("B", "A")], None, "acyclic"),
        ([("A", "B", "C")], None, "pair"),
        (["AB"], None, "pair"),
    ],
)
def test_edges_reject_invalid_graphs(edges, nodes, error) -> None:
    with pytest.raises(ValueError, match=error):
        CausalGraph.from_edges(edges, nodes=nodes)


@pytest.mark.parametrize(
    ("matrix", "nodes", "error"),
    [
        ([[0]], ("A", "B"), "shape"),
        ([[0, 1], [0]], ("A", "B"), "shape"),
        ([0, 1], ("A", "B"), "square"),
        ([[0, 2], [0, 0]], ("A", "B"), "0 or 1"),
        ([[0, float("nan")], [0, 0]], ("A", "B"), "0 or 1"),
        ([["0", 1], [0, 0]], ("A", "B"), "0 or 1"),
        ([[0, 1], [1, 0]], ("A", "B"), "acyclic"),
        ([[1]], ("A",), "Self-loop"),
        ([[0, 0], [0, 0]], ("A", "A"), "Duplicate graph nodes"),
        ([[0]], (4,), "nonempty string"),
    ],
)
def test_matrix_rejects_invalid_values_and_shapes(matrix, nodes, error) -> None:
    with pytest.raises(ValueError, match=error):
        CausalGraph.from_adjacency_matrix(matrix, nodes=nodes)


def test_matrix_accepts_numpy_and_dense_torch() -> None:
    np = pytest.importorskip("numpy")
    values = [[0, 1], [0, 0]]
    for matrix in (np.array(values), torch.tensor(values, dtype=torch.float64)):
        graph = CausalGraph.from_adjacency_matrix(matrix, nodes=["A", "B"])
        assert graph.parents_of("B") == ("A",)
        assert graph.to_adjacency_matrix() == ((0, 1), (0, 0))


def test_edge_and_matrix_nodes_arguments_reject_strings() -> None:
    with pytest.raises(TypeError, match="sequence"):
        CausalGraph.from_edges([("A", "B")], nodes="AB")
    with pytest.raises(TypeError, match="sequence"):
        CausalGraph.from_adjacency_matrix([[0]], nodes="A")


def test_networkx_roundtrip_and_isolated_nodes() -> None:
    nx = pytest.importorskip("networkx")
    source = nx.DiGraph()
    source.add_nodes_from(("X", "Z", "Y", "isolated"))
    source.add_edges_from((("X", "Z"), ("X", "Y"), ("Z", "Y")))
    converted = CausalGraph.from_networkx(source)
    _assert_same_dag(converted)
    reconstructed = converted.to_networkx()
    assert tuple(reconstructed.nodes) == tuple(source.nodes)
    assert set(reconstructed.edges) == set(source.edges)
    reconstructed.add_edge("isolated", "Y")
    assert converted.parents_of("Y") == ("X", "Z")


def test_networkx_rejects_unsupported_graphs() -> None:
    nx = pytest.importorskip("networkx")
    for graph in (nx.Graph([("A", "B")]), nx.MultiDiGraph([("A", "B")])):
        with pytest.raises(TypeError, match="DiGraph"):
            CausalGraph.from_networkx(graph)
    with pytest.raises(ValueError, match="acyclic"):
        CausalGraph.from_networkx(nx.DiGraph([("A", "B"), ("B", "A")]))
    with pytest.raises(ValueError, match="nonempty string"):
        CausalGraph.from_networkx(nx.DiGraph([(0, "B")]))


def test_optional_networkx_is_only_needed_when_called(monkeypatch) -> None:
    graph = CausalGraph.from_edges([("A", "B")])
    monkeypatch.setitem(sys.modules, "networkx", None)
    with pytest.raises(ImportError, match="robotorchan\\[causal\\]"):
        graph.to_networkx()
    with pytest.raises(ImportError, match="robotorchan\\[causal\\]"):
        CausalGraph.from_networkx(object())
    assert graph.parents_of("B") == ("A",)


def test_scm_can_use_all_graph_formats_unchanged() -> None:
    graphs = (
        CausalGraph({"X": (), "Y": ("X",)}),
        CausalGraph.from_edges([("X", "Y")]),
        CausalGraph.from_adjacency_matrix([[0, 1], [0, 0]], nodes=("X", "Y")),
    )
    for graph in graphs:
        scm = StructuralCausalModel(graph, {"X": lambda p: 1.0, "Y": lambda p: p["X"] + 2})
        assert scm.evaluate(HardIntervention({"X": 4.0})) == {"X": 4.0, "Y": 6.0}
