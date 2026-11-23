"""Phase 4: known-DAG GP structural mechanisms and do-intervention sampling."""

from __future__ import annotations

import pytest
import torch

from robotorchan.causal import (
    CausalGraph,
    CausalObservation,
    GPStructuralCausalModel,
    HardIntervention,
    StructuralCausalModel,
)

DTYPE = torch.double


@pytest.fixture(scope="module")
def fitted_scm() -> GPStructuralCausalModel:
    graph = CausalGraph({"X": (), "Z": ("X",), "Y": ("Z",)})
    deterministic = StructuralCausalModel(
        graph,
        {
            "X": lambda parents: 0.0,
            "Z": lambda parents: 2.0 * parents["X"],
            "Y": lambda parents: 1.0 + parents["Z"],
        },
    )
    records = [
        CausalObservation({"X": x, "Z": 2.0 * x, "Y": 1.0 + 2.0 * x})
        for x in (0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95)
    ]
    for intervention in (
        HardIntervention({"X": 0.4}),
        HardIntervention({"Z": 0.6}),
        HardIntervention({"Y": 10.0}),
    ):
        values = deterministic.evaluate(intervention)
        records.append(CausalObservation(values, intervention))

    model = GPStructuralCausalModel(graph, records)
    model.fit()
    return model


def test_fit_uses_native_models_and_excludes_intervened_mechanisms(fitted_scm) -> None:
    model = fitted_scm
    assert tuple(model.mechanisms) == ("Z", "Y")
    assert model.mechanisms["Z"].raw_train_X.shape == (9, 1)
    assert model.mechanisms["Y"].raw_train_X.shape == (9, 1)
    # Intervening on Y must not leak the imposed Y=10 value into Y training.
    assert float(model.mechanisms["Y"].raw_train_Y.max()) < 10.0
    posterior = model.mechanism_posterior("Y", torch.tensor([[0.3], [0.7]], dtype=DTYPE))
    assert posterior.mean.shape == (2, 1)
    assert torch.isfinite(posterior.variance).all()
    with pytest.raises(ValueError, match="Root nodes"):
        model.mechanism_posterior("X", torch.zeros(1, 1, dtype=DTYPE))


def test_intervention_overrides_equation_and_has_mc_uncertainty(fitted_scm) -> None:
    model = fitted_scm
    fixed = model.sample_intervention(
        HardIntervention({"Y": 7.0}), outcome="Y", sample_shape=torch.Size([24])
    )
    torch.testing.assert_close(fixed, torch.full((24,), 7.0, dtype=DTYPE))
    sampled = model.sample_intervention(
        HardIntervention({"Z": 0.5}), outcome="Y", sample_shape=torch.Size([128])
    )
    assert sampled.shape == (128,)
    assert torch.isfinite(sampled).all()
    assert abs(float(sampled.mean()) - 1.5) < 0.8
    assert float(sampled.var()) > 0.0


def test_tensor_interventions_keep_candidate_shape_and_gradients(fitted_scm) -> None:
    model = fitted_scm
    candidates = torch.tensor([[[0.15], [0.65]]], dtype=DTYPE, requires_grad=True)
    draws = model.sample_interventions(
        candidates, intervention_nodes=("X",), outcome="Y", sample_shape=torch.Size([8])
    )
    assert draws.shape == (8, 1, 2)
    assert torch.isfinite(draws).all()
    draws.mean().backward()
    assert candidates.grad is not None
    assert torch.isfinite(candidates.grad).all()


def test_observation_noise_is_passed_to_botorch_posterior(fitted_scm, monkeypatch) -> None:
    gp = fitted_scm.mechanisms["Y"]
    original = gp.posterior
    seen: list[bool] = []

    def capture(*args, observation_noise=False, **kwargs):
        seen.append(observation_noise)
        return original(*args, observation_noise=observation_noise, **kwargs)

    monkeypatch.setattr(gp, "posterior", capture)
    fitted_scm.sample_intervention(
        HardIntervention({"Z": 0.5}),
        outcome="Y",
        sample_shape=torch.Size([5]),
        observation_noise=True,
    )
    assert seen == [True]


def test_root_distribution_is_empirical_and_constant_root_is_deterministic() -> None:
    graph = CausalGraph({"X": (), "I": ()})
    records = [CausalObservation({"X": x, "I": 1.0}) for x in (0.0, 2.0, 4.0)]
    model = GPStructuralCausalModel(graph, records)
    model.fit()
    torch.manual_seed(42)
    xs = model.sample_intervention(
        HardIntervention({"I": 7.0}), outcome="X", sample_shape=torch.Size([2048])
    )
    assert abs(float(xs.mean()) - 2.0) < 0.2
    assert 1.7 < float(xs.std()) < 2.3
    values = model.sample_intervention(
        HardIntervention({"X": 3.0}), outcome="I", sample_shape=torch.Size([15])
    )
    torch.testing.assert_close(values, torch.ones(15, dtype=DTYPE))


def test_missing_data_and_direct_interventions_are_not_imputed() -> None:
    graph = CausalGraph({"X": (), "Y": ("X",)})
    records = [
        CausalObservation({"X": 0.0, "Y": 0.0}),
        CausalObservation({"X": 1.0}),
        CausalObservation({"X": 0.2, "Y": 42.0}, HardIntervention({"Y": 42.0})),
    ]
    model = GPStructuralCausalModel(graph, records)
    with pytest.raises(ValueError, match=r"Node 'Y'.*two complete"):
        model.fit()
    with pytest.raises(RuntimeError, match="Fit"):
        model.sample_intervention(HardIntervention({"X": 0.5}), outcome="Y")


def test_unobserved_root_does_not_get_a_fictitious_prior() -> None:
    graph = CausalGraph({"X": (), "Y": ("X",)})
    records = [
        CausalObservation({"X": 0.0, "Y": 1.0}, HardIntervention({"X": 0.0})),
        CausalObservation({"X": 1.0, "Y": 2.0}, HardIntervention({"X": 1.0})),
    ]
    with pytest.raises(ValueError, match=r"Node 'X'.*two complete"):
        GPStructuralCausalModel(graph, records).fit()


def test_parent_order_is_followed_for_native_gp_inputs() -> None:
    graph = CausalGraph({"A": (), "B": (), "Y": ("B", "A")})
    records = [
        CausalObservation({"A": a, "B": b, "Y": a + 2 * b})
        for a, b in ((0.1, 0.4), (0.3, 0.7), (0.6, 0.1), (0.9, 0.8))
    ]
    model = GPStructuralCausalModel(graph, records)
    model.fit()
    torch.testing.assert_close(
        model.mechanisms["Y"].raw_train_X,
        torch.tensor([[0.4, 0.1], [0.7, 0.3], [0.1, 0.6], [0.8, 0.9]], dtype=DTYPE),
    )


@pytest.mark.parametrize(
    "invalid",
    [
        CausalObservation({"X": 0.1, "Y": 0.2, "UNKNOWN": 2.0}),
        CausalObservation({"X": 0.1, "Y": 0.2}, HardIntervention({"Y": 0.2})),
    ],
)
def test_unknown_nodes_are_rejected(invalid) -> None:
    graph = CausalGraph({"X": (), "Z": ()})
    with pytest.raises(ValueError, match="absent"):
        GPStructuralCausalModel(graph, [invalid])


def test_inputs_are_checked_without_converting_do_to_conditioning(fitted_scm) -> None:
    model = fitted_scm
    with pytest.raises(ValueError, match="belong"):
        model.sample_interventions(
            torch.zeros(1, 1, dtype=DTYPE), intervention_nodes=("BAD",), outcome="Y"
        )
    with pytest.raises(ValueError, match="nonempty and distinct"):
        model.sample_interventions(
            torch.zeros(1, 2, dtype=DTYPE), intervention_nodes=("X", "X"), outcome="Y"
        )
    with pytest.raises(ValueError, match="shape"):
        model.sample_interventions(
            torch.zeros(1, 2, dtype=DTYPE), intervention_nodes=("X",), outcome="Y"
        )
    with pytest.raises(ValueError, match="dtype"):
        model.sample_interventions(
            torch.zeros(1, 1, dtype=torch.float32), intervention_nodes=("X",), outcome="Y"
        )
    with pytest.raises(ValueError, match="positive"):
        model.sample_interventions(
            torch.zeros(1, 1, dtype=DTYPE),
            intervention_nodes=("X",),
            outcome="Y",
            sample_shape=torch.Size([0]),
        )
    with pytest.raises(ValueError, match="non-finite"):
        model.sample_interventions(
            torch.full((1, 1), float("nan"), dtype=DTYPE),
            intervention_nodes=("X",),
            outcome="Y",
        )
    with pytest.raises(ValueError, match="belong"):
        model.sample_interventions(
            torch.zeros(1, 1, dtype=DTYPE), intervention_nodes=("X",), outcome="NO"
        )


def test_root_model_does_not_require_networkx() -> None:
    graph = CausalGraph.from_adjacency_matrix([[0, 0], [0, 0]], nodes=("X", "Y"))
    model = GPStructuralCausalModel(
        graph,
        [CausalObservation({"X": x, "Y": x + 1}) for x in (0.0, 1.0, 2.0)],
    )
    model.fit()
    assert not model.mechanisms
    sample = model.sample_intervention(
        HardIntervention({"X": 0.7}), outcome="Y", sample_shape=torch.Size([10])
    )
    assert sample.shape == (10,)
