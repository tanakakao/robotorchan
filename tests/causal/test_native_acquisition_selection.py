"""Native BoTorch acquisition optimization across known intervention sets."""

from math import isclose, isfinite, log

import pytest
import torch
from botorch.acquisition.analytic import LogExpectedImprovement, PosteriorMean, UpperConfidenceBound
from botorch.models import SingleTaskGP

from robotorchan.causal import CausalAcquisitionProposal, optimize_causal_acqfs


def _model(dim: int, *, dtype: torch.dtype = torch.double) -> SingleTaskGP:
    train_x = torch.linspace(0.0, 1.0, 5, dtype=dtype).unsqueeze(-1).repeat(1, dim)
    train_y = torch.sin(train_x[:, :1] * 2.0)
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    return model


def test_posterior_mean_uses_native_acquisition_and_optimizer() -> None:
    model = _model(1)
    acquisition = PosteriorMean(model)
    result = optimize_causal_acqfs(
        {("X",): acquisition},
        {"X": (0.0, 1.0)},
        score=lambda nodes, value: value,
        optimizer="sobol",
        raw_samples=32,
    )
    assert isinstance(result, CausalAcquisitionProposal)
    assert set(result.intervention.values) == {"X"}
    assert 0.0 <= result.intervention.values["X"] <= 1.0
    assert isfinite(result.acquisition_value)
    assert isclose(result.ranking_score, result.acquisition_value)


def test_different_dimensions_column_order_and_explicit_scores(monkeypatch) -> None:
    acqfs = {
        ("Z", "X"): UpperConfidenceBound(_model(2), beta=0.2),
        ("X",): PosteriorMean(_model(1)),
    }
    seen: list[tuple[tuple[float, ...], ...]] = []

    def fake_optimize(acquisition, bounds, q, **kwargs):
        assert q == 1
        assert bounds.dtype == torch.double
        seen.append(tuple(tuple(row) for row in bounds.tolist()))
        return bounds.mean(0, keepdim=True), torch.tensor(float(bounds.shape[-1]))

    monkeypatch.setattr("robotorchan.causal.proposal.optimize_acqf", fake_optimize)
    result = optimize_causal_acqfs(
        acqfs,
        {"X": (0.0, 2.0), "Z": (4.0, 6.0)},
        score=lambda nodes, value: value - (1.5 if len(nodes) == 2 else 0.0),
        optimizer="sobol",
        raw_samples=16,
    )
    assert seen == [
        ((4.0, 0.0), (6.0, 2.0)),
        ((0.0,), (2.0,)),
    ]
    assert set(result.intervention.values) == {"X"}
    assert result.acquisition_value == 1.0
    assert result.ranking_score == 1.0


def test_cost_ranking_is_explicit_not_inferred(monkeypatch) -> None:
    acqfs = {
        ("X",): LogExpectedImprovement(_model(1), best_f=0.0),
        ("Z",): LogExpectedImprovement(_model(1), best_f=0.0),
    }

    def fake_optimize(acquisition, bounds, q, **kwargs):
        return bounds.mean(0, keepdim=True), torch.tensor(0.0, dtype=bounds.dtype)

    monkeypatch.setattr("robotorchan.causal.proposal.optimize_acqf", fake_optimize)
    costs = {("X",): 10.0, ("Z",): 2.0}
    result = optimize_causal_acqfs(
        acqfs,
        {"X": (0.0, 1.0), "Z": (0.0, 1.0)},
        score=lambda nodes, value: value - log(costs[nodes]),
    )
    assert set(result.intervention.values) == {"Z"}
    assert result.acquisition_value == 0.0
    assert isclose(result.ranking_score, -log(2.0))


def test_bounds_dtype_follows_native_acquisition_model(monkeypatch) -> None:
    acqf = PosteriorMean(_model(1, dtype=torch.float32))

    def fake_optimize(acquisition, bounds, q, **kwargs):
        assert bounds.dtype == torch.float32
        return bounds.mean(0, keepdim=True), torch.tensor(0.5, dtype=bounds.dtype)

    monkeypatch.setattr("robotorchan.causal.proposal.optimize_acqf", fake_optimize)
    result = optimize_causal_acqfs(
        {("X",): acqf}, {"X": (0.0, 1.0)}, score=lambda nodes, value: value
    )
    assert result.acquisition_value == 0.5


@pytest.mark.parametrize(
    ("keys", "bounds", "expected"),
    [
        ({}, {"X": (0.0, 1.0)}, "At least one"),
        ({"X": None}, {"X": (0.0, 1.0)}, "ordered node tuples"),
        ({("X", "X"): None}, {"X": (0.0, 1.0)}, "Duplicate nodes"),
        ({("",): None}, {"": (0.0, 1.0)}, "nonempty strings"),
    ],
)
def test_rejects_invalid_intervention_keys(keys, bounds, expected) -> None:
    with pytest.raises(ValueError, match=expected):
        optimize_causal_acqfs(keys, bounds, score=lambda nodes, value: value)


def test_rejects_reordered_duplicate_groups() -> None:
    acquisition = PosteriorMean(_model(2))
    with pytest.raises(ValueError, match="only once"):
        optimize_causal_acqfs(
            {("X", "Z"): acquisition, ("Z", "X"): acquisition},
            {"X": (0.0, 1.0), "Z": (0.0, 1.0)},
            score=lambda nodes, value: value,
        )


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ({"Z": (0.0, 1.0)}, "exactly"),
        ({"X": (1.0, 0.0)}, "lower < upper"),
        ({"X": (float("nan"), 1.0)}, "finite"),
        ({"X": (0.0, float("inf"))}, "finite"),
        ({"X": (0.0,)}, "lower and upper"),
        ({"X": (False, 1.0)}, "lower and upper"),
    ],
)
def test_rejects_missing_and_invalid_bounds(bounds, expected) -> None:
    with pytest.raises(ValueError, match=expected):
        optimize_causal_acqfs(
            {("X",): PosteriorMean(_model(1))},
            bounds,
            score=lambda nodes, value: value,
        )


@pytest.mark.parametrize(
    ("candidate", "native_value", "expected"),
    [
        ([[0.5, 0.2]], 1.0, "one candidate"),
        ([[float("nan")]], 1.0, "non-finite intervention"),
        ([[1.3]], 1.0, "outside"),
        ([[0.5]], float("nan"), "non-finite acquisition"),
    ],
)
def test_rejects_invalid_optimizer_output(monkeypatch, candidate, native_value, expected) -> None:
    def fake_optimize(acquisition, bounds, q, **kwargs):
        return torch.tensor(candidate, dtype=bounds.dtype), torch.tensor(native_value)

    monkeypatch.setattr("robotorchan.causal.proposal.optimize_acqf", fake_optimize)
    with pytest.raises(ValueError, match=expected):
        optimize_causal_acqfs(
            {("X",): PosteriorMean(_model(1))},
            {"X": (0.0, 1.0)},
            score=lambda nodes, value: value,
        )


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), True, "invalid"])
def test_rejects_nonfinite_or_nonnumeric_ranking(monkeypatch, bad_score) -> None:
    def fake_optimize(acquisition, bounds, q, **kwargs):
        return bounds.mean(0, keepdim=True), torch.tensor(0.0)

    monkeypatch.setattr("robotorchan.causal.proposal.optimize_acqf", fake_optimize)
    with pytest.raises(ValueError, match="score must return"):
        optimize_causal_acqfs(
            {("X",): PosteriorMean(_model(1))},
            {"X": (0.0, 1.0)},
            score=lambda nodes, value: bad_score,
        )


def test_rejects_non_native_acquisition_and_missing_score() -> None:
    with pytest.raises(TypeError, match="AcquisitionFunction"):
        optimize_causal_acqfs(
            {("X",): object()},
            {"X": (0.0, 1.0)},
            score=lambda nodes, value: value,
        )
    with pytest.raises(TypeError, match="score must be"):
        optimize_causal_acqfs(
            {("X",): PosteriorMean(_model(1))},
            {"X": (0.0, 1.0)},
            score=None,
        )
