"""BoTorch compatibility and low-level escape-hatch contracts."""

from __future__ import annotations

import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.semantics import RegressionObjective


def _model() -> HeterogeneousModel:
    train_X = torch.rand(6, 2, dtype=torch.double)
    train_Y = train_X.sum(dim=-1, keepdim=True)
    regression = SingleTaskGP(train_X, train_Y)
    return HeterogeneousModel(
        regression,
        names=["regression"],
        output_names=["score"],
    )


def test_heterogeneous_entry_is_native_botorch_model() -> None:
    model = _model()

    assert isinstance(model["regression"], SingleTaskGP)
    assert model["regression"] is model[0]


def test_entry_posterior_preserves_native_botorch_posterior() -> None:
    model = _model()
    X = torch.rand(3, 1, 2, dtype=torch.double)

    direct = model[0].posterior(X)
    escaped = model.entry_posterior("regression", X)

    assert type(escaped) is type(direct)
    assert torch.allclose(escaped.mean, direct.mean)
    assert torch.allclose(escaped.variance, direct.variance)


def test_semantic_objective_can_feed_native_botorch_acquisition() -> None:
    model = _model()
    native_model = model["regression"]
    objective = RegressionObjective(output="score").to_botorch(model)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))
    best_f = native_model.train_targets.max()
    acquisition = qExpectedImprovement(
        model=native_model,
        best_f=best_f,
        sampler=sampler,
        objective=objective,
    )

    X = torch.rand(2, 1, 2, dtype=torch.double)
    values = acquisition(X)

    assert values.shape == torch.Size([2])


def test_low_level_botorch_objective_does_not_require_semantic_wrapper() -> None:
    model = _model()
    native_model = model[0]
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))
    acquisition = qExpectedImprovement(
        model=native_model,
        best_f=native_model.train_targets.max(),
        sampler=sampler,
    )

    values = acquisition(torch.rand(2, 1, 2, dtype=torch.double))

    assert values.shape == torch.Size([2])
