"""Minimal Phase 3 coverage for interventional GP training data and posterior."""

import pytest
import torch

from robotorchan.causal import (
    CausalGraph,
    CausalObservation,
    HardIntervention,
    InterventionGPSurrogate,
    StructuralCausalModel,
)
from robotorchan.models.standard.single_task import SingleTaskGP


def _data():
    graph = CausalGraph({"Y": ("X", "Z"), "Z": (), "X": ()})
    scm = StructuralCausalModel(
        graph,
        {"X": lambda p: 0.0, "Z": lambda p: 0.0, "Y": lambda p: p["X"] + 2 * p["Z"]},
    )
    records = []
    for value in (0.0, 0.3, 0.6, 0.9):
        intervention = HardIntervention({"X": value})
        records.append(CausalObservation(scm.evaluate(intervention), intervention))
    records.append(CausalObservation({"X": 0.5, "Z": 0.0, "Y": 100.0}))
    other = HardIntervention({"Z": 0.5})
    records.append(CausalObservation(scm.evaluate(other), other))
    return graph, records


def test_interventional_records_only_and_native_botorch_posterior():
    graph, records = _data()
    surrogate = InterventionGPSurrogate(graph, "Y", ("X",), records)
    assert isinstance(surrogate.model, SingleTaskGP)
    assert surrogate.intervention_nodes == ("X",)
    assert surrogate.train_X.shape == (4, 1)
    torch.testing.assert_close(
        surrogate.train_Y.squeeze(-1),
        torch.tensor([0.0, 0.3, 0.6, 0.9], dtype=torch.double),
    )
    surrogate.fit()
    with torch.no_grad():
        posterior = surrogate.posterior(torch.tensor([[0.2], [0.8]], dtype=torch.double))
        native = surrogate.model.posterior(torch.tensor([[0.2], [0.8]], dtype=torch.double))
    assert posterior.mean.shape == (2, 1)
    torch.testing.assert_close(posterior.mean, native.mean)
    assert torch.isfinite(posterior.variance).all()


def test_canonical_intervention_column_order():
    graph = CausalGraph({"Y": ("Z", "X"), "Z": (), "X": ()})
    observations = [
        CausalObservation(
            {"X": x, "Z": z, "Y": x + z},
            HardIntervention({"X": x, "Z": z}),
        )
        for x, z in ((0.1, 0.2), (0.8, 0.6))
    ]
    surrogate = InterventionGPSurrogate(graph, "Y", ("X", "Z"), observations)
    assert surrogate.intervention_nodes == ("Z", "X")
    torch.testing.assert_close(
        surrogate.train_X,
        torch.tensor([[0.2, 0.1], [0.6, 0.8]], dtype=torch.double),
    )


def test_rejects_missing_group_and_intervened_outcome():
    graph, observations = _data()
    with pytest.raises(ValueError, match="At least two"):
        InterventionGPSurrogate(graph, "Y", ("Z",), observations)
    with pytest.raises(ValueError, match="cannot be intervened"):
        InterventionGPSurrogate(graph, "Y", ("Y",), observations)
    with pytest.raises(ValueError, match="duplicates"):
        InterventionGPSurrogate(graph, "Y", ("X", "X"), observations)
    with pytest.raises(ValueError, match="shape"):
        InterventionGPSurrogate(graph, "Y", ("X",), observations).posterior(
            torch.zeros(2, 2, dtype=torch.double)
        )
