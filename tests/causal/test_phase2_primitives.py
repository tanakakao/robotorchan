"""Targeted tests for the known-graph causal primitives."""

import pytest

from robotorchan.causal import (
    CausalGraph,
    CausalObservation,
    HardIntervention,
    StructuralCausalModel,
)


def test_topological_order_and_graph_validation():
    graph = CausalGraph({"Y": ("M",), "M": ("X",), "X": ()})
    assert graph.topological_order == ("X", "M", "Y")
    with pytest.raises(ValueError, match="acyclic"):
        CausalGraph({"X": ("Y",), "Y": ("X",)})
    with pytest.raises(ValueError, match="Unknown parent"):
        CausalGraph({"Y": ("missing",)})
    with pytest.raises(TypeError, match="not a string"):
        CausalGraph({"X": (), "Y": "X"})


def test_intervention_replaces_equation_not_conditioning():
    graph = CausalGraph({"X": (), "M": ("X",), "Y": ("M",)})
    model = StructuralCausalModel(
        graph,
        {"X": lambda p: 2.0, "M": lambda p: p["X"] * 3, "Y": lambda p: p["M"] + 1},
    )
    assert model.evaluate()["Y"] == 7.0
    assert model.evaluate(HardIntervention({"M": 10.0})) == {
        "X": 2.0,
        "M": 10.0,
        "Y": 11.0,
    }
    assert model.evaluate(HardIntervention({"X": 4.0}))["Y"] == 13.0


def test_input_snapshots_are_read_only():
    parents = {"X": [], "Y": ["X"]}
    graph = CausalGraph(parents)
    parents["Y"].clear()
    assert graph.parents_of("Y") == ("X",)
    with pytest.raises(TypeError):
        graph._parents["Y"] = ()

    assigned = {"X": 2}
    intervention = HardIntervention(assigned)
    assigned["X"] = 5
    assert intervention.values["X"] == 2.0
    with pytest.raises(TypeError):
        intervention.values["X"] = 5.0

    recorded = {"X": 2}
    observation = CausalObservation(recorded, intervention)
    recorded["X"] = 5
    assert observation.values["X"] == 2.0


def test_invalid_interventions_and_records():
    graph = CausalGraph({"X": ()})
    model = StructuralCausalModel(graph, {"X": lambda p: 1.0})
    with pytest.raises(ValueError, match="Unknown intervention"):
        model.evaluate(HardIntervention({"Z": 1.0}))
    with pytest.raises(ValueError, match="must match"):
        CausalObservation({"X": 1.0}, intervention=HardIntervention({"X": 2.0}))
    assert not CausalObservation({"X": 1.0}).is_interventional
    assert CausalObservation({"X": 2.0}, HardIntervention({"X": 2.0})).is_interventional
