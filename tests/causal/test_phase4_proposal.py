"""Phase 4: intervention-set comparison, optional costs, and native LogEI."""

from math import isclose, isfinite, log

import pytest
import torch

from robotorchan.causal import (
    CausalGraph,
    CausalObservation,
    HardIntervention,
    InterventionGPSurrogate,
    propose_causal_intervention,
)


@pytest.fixture(scope="module")
def fitted_groups():
    graph = CausalGraph({"X": (), "Z": (), "Y": ("X", "Z")})
    records = []
    for node in ("X", "Z"):
        for x in (0.0, 0.25, 0.5, 0.75):
            do = HardIntervention({node: x})
            records.append(CausalObservation({node: x, "Y": x * x}, do))
    groups = [InterventionGPSurrogate(graph, "Y", (node,), records) for node in ("X", "Z")]
    for group in groups:
        group.fit()
    return groups


def test_native_botorch_logei_optimizes_intervention_value(fitted_groups):
    result = propose_causal_intervention(
        fitted_groups[:1], {"X": (0.0, 1.0)}, optimizer="sobol", raw_samples=32
    )
    assert set(result.intervention.values) == {"X"}
    assert 0.0 <= result.intervention.values["X"] <= 1.0
    assert isfinite(result.log_expected_improvement)
    assert isclose(result.log_utility, result.log_expected_improvement)


def test_cost_adjustment_selects_cheaper_intervention(fitted_groups, monkeypatch):
    seen = []

    def fake_optimize(acquisition, bounds, q, **kwargs):
        assert q == 1
        seen.append((acquisition, bounds))
        return bounds.mean(dim=0, keepdim=True), torch.tensor(0.0, dtype=bounds.dtype)

    monkeypatch.setattr("robotorchan.causal.proposal.optimize_acqf", fake_optimize)
    result = propose_causal_intervention(
        fitted_groups,
        {"X": (0.0, 1.0), "Z": (0.0, 1.0)},
        intervention_costs={frozenset({"X"}): 10.0, frozenset({"Z"}): 2.0},
        direction="minimize",
    )
    assert len(seen) == 2
    assert all(acquisition.maximize is False for acquisition, _ in seen)
    assert set(result.intervention.values) == {"Z"}
    assert result.intervention_cost == 2.0
    assert isclose(result.log_expected_improvement, 0.0)
    assert isclose(result.log_utility, -log(2.0))


def test_rejects_incompatible_inputs(fitted_groups):
    with pytest.raises(ValueError, match="bounds"):
        propose_causal_intervention(fitted_groups[:1], {"Z": (0.0, 1.0)})
    with pytest.raises(ValueError, match="once"):
        propose_causal_intervention(fitted_groups[:1] * 2, {"X": (0.0, 1.0)})
    with pytest.raises(ValueError, match="positive"):
        propose_causal_intervention(
            fitted_groups[:1],
            {"X": (0.0, 1.0)},
            intervention_costs={frozenset({"X"}): 0.0},
        )
    with pytest.raises(ValueError, match="direction"):
        propose_causal_intervention(fitted_groups[:1], {"X": (0.0, 1.0)}, direction="invalid")
