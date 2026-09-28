"""PosteriorTransform compatibility tests for representative model families."""

import pytest
import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform

from robotorchan.models import (
    KroneckerMultiTaskGP,
    MixedSingleTaskGP,
    SingleTaskGP,
)
from robotorchan.models.expressive.deep_gp import SingleTaskDeepGP
from robotorchan.models.high_dimensional.reduced.base import PCAGP, OutputPCAGP
from robotorchan.models.non_gp.random_forest import RandomForestSurrogate


def _single_output_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.05, 0.95, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(2.0 * torch.pi * train_X)
    return train_X, train_Y


@pytest.mark.parametrize("model_cls", [SingleTaskGP, PCAGP])
def test_single_output_gp_accepts_scalarized_posterior_transform(model_cls) -> None:
    train_X, train_Y = _single_output_data()
    kwargs = {"n_components": 1} if model_cls is PCAGP else {}
    model = model_cls(train_X, train_Y, **kwargs)
    transform = ScalarizedPosteriorTransform(
        weights=torch.tensor([2.0], dtype=train_X.dtype),
        offset=0.5,
    )

    raw = model.posterior(train_X[:3])
    transformed = model.posterior(train_X[:3], posterior_transform=transform)

    torch.testing.assert_close(transformed.mean, 0.5 + 2.0 * raw.mean)
    torch.testing.assert_close(transformed.variance, 4.0 * raw.variance)


def test_mixed_gp_accepts_scalarized_posterior_transform() -> None:
    continuous = torch.linspace(0.05, 0.95, 8, dtype=torch.double).unsqueeze(-1)
    category = torch.arange(8, dtype=torch.double).remainder(2).unsqueeze(-1)
    train_X = torch.cat((continuous, category), dim=-1)
    train_Y = torch.sin(2.0 * torch.pi * continuous) + 0.2 * category
    model = MixedSingleTaskGP(train_X, train_Y, cat_dims=[1])
    transform = ScalarizedPosteriorTransform(
        weights=torch.ones(1, dtype=train_X.dtype),
    )

    posterior = model.posterior(train_X[:3], posterior_transform=transform)

    assert posterior.mean.shape == torch.Size([3, 1])
    assert torch.isfinite(posterior.mean).all()


def test_kronecker_explicitly_rejects_scalarized_posterior_transform() -> None:
    train_X = torch.linspace(0.05, 0.95, 7, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cat(
        (
            torch.sin(2.0 * torch.pi * train_X),
            torch.cos(2.0 * torch.pi * train_X),
        ),
        dim=-1,
    )
    model = KroneckerMultiTaskGP(train_X, train_Y)
    weights = torch.tensor([0.25, 0.75], dtype=train_X.dtype)
    transform = ScalarizedPosteriorTransform(weights=weights)

    with pytest.raises(NotImplementedError, match="Posterior transforms"):
        model.posterior(train_X[:3], posterior_transform=transform)


def test_random_forest_rejects_scalarized_posterior_transform() -> None:
    pytest.importorskip("sklearn")
    train_X, train_Y = _single_output_data()
    model = RandomForestSurrogate(
        train_X,
        train_Y,
        n_estimators=8,
        random_state=0,
    )
    model.fit()
    transform = ScalarizedPosteriorTransform(
        weights=torch.ones(1, dtype=train_X.dtype),
    )

    with pytest.raises(NotImplementedError, match="scalarize_posterior"):
        model.posterior(train_X[:3], posterior_transform=transform)


def test_deep_gp_rejects_scalarized_posterior_transform() -> None:
    train_X, train_Y = _single_output_data()
    model = SingleTaskDeepGP(
        train_X,
        train_Y,
        hidden_dims=(3,),
        num_inducing=4,
        posterior_samples=8,
    )
    transform = ScalarizedPosteriorTransform(
        weights=torch.ones(1, dtype=train_X.dtype),
    )

    with pytest.raises(NotImplementedError, match="scalarize_posterior"):
        model.posterior(
            train_X[:3],
            posterior_transform=transform,
            num_samples=8,
        )


def test_output_reduction_applies_transform_after_original_output_restoration() -> None:
    train_X = torch.linspace(0.05, 0.95, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cat(
        (
            torch.sin(2.0 * torch.pi * train_X),
            torch.cos(2.0 * torch.pi * train_X),
            train_X.square(),
        ),
        dim=-1,
    )
    model = OutputPCAGP(train_X, train_Y, n_components=2)
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=train_X.dtype)
    transform = ScalarizedPosteriorTransform(weights=weights, offset=0.4)

    raw = model.posterior(train_X[:3])
    transformed = model.posterior(train_X[:3], posterior_transform=transform)

    expected_mean = 0.4 + (raw.mean * weights).sum(dim=-1, keepdim=True)
    expected_variance = (raw.variance * weights.square()).sum(dim=-1, keepdim=True)
    torch.testing.assert_close(transformed.mean, expected_mean)
    torch.testing.assert_close(transformed.variance, expected_variance)
