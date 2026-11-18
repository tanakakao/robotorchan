"""Targeted tests for the known-graph causal primitives."""

import pytest

from robotorchan.causal import (
    CausalGraph,
    CausalObservation,
    HardIntervention,
    StructuralCausalModel,
)


def test_topological_order_and_cycle_rejection():
    graph = CausalGraph({"Y": ("M",), "M": ("X",), "X": ()})
    assert graph.topological_order == ("X", "M", "Y")
    with pytest.raises(ValueError, match="acyclic"):
        CausalGraph({"X": ("Y",), "Y": ("X",)})


def test_intervention_replaces_equation_not_conditioning():
    graph = CausalGraph({"X": (), "M": ("X",), "Y": ("M",)})
    model = StructuralCausalModel(
        graph,
        {"X": lambda p: 2.0, "M": lambda p: p["X"] * 3, "Y": lambda p: p["M"] + 1},
    )
    assert model.evaluate()["Y"] == 7.0
    assert model.evaluate(HardIntervention({"M": 10.0})) == {
        "X": 2.0, "M": 10.0, "Y": 11.0
    }
    assert model.evaluate(HardIntervention({"X": 4.0}))["Y"] == 13.0


def test_invalid_interventions_and_records():
    graph = CausalGraph({"X": ()})
    model = StructuralCausalModel(graph, {"X": lambda p: 1.0})
    with pytest.raises(ValueError, match="Unknown intervention"):
        model.evaluate(HardIntervention({"Z": 1.0}))
    with pytest.raises(ValueError, match="must match"):
        CausalObservation({"X": 1.0}, intervention=HardIntervention({"X": 2.0}))
    assert not CausalObservation({"X": 1.0}).is_interventional
    assert CausalObservation({"X": 2.0}, HardIntervention({"X": 2.0})).is_interventional
