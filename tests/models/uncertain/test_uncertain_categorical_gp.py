"""Tests for uncertain categorical training-input GP."""

import pytest
import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

from robotorchan.models.uncertain.uncertain_categorical import (
    ExpectedCategoricalKernel,
    UncertainCategoricalSingleTaskGP,
)


def _data():
    X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
    categories = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    P = torch.nn.functional.one_hot(categories, num_classes=2).to(dtype=torch.double)
    Y = torch.sin(2 * torch.pi * X) + 0.1 * P[:, :1]
    return X, P, Y


def test_expected_category_kernel_is_symmetric_psd_and_one_hot_exact() -> None:
    _, P, _ = _data()
    kernel = ExpectedCategoricalKernel(2).double()
    covariance = kernel(P, P).to_dense()
    category_covar = kernel.category_covar()

    assert torch.allclose(covariance, covariance.transpose(-1, -2))
    assert torch.linalg.eigvalsh(covariance).min() >= -1e-8
    assert torch.allclose(covariance[0, 1], category_covar[0, 1])
    assert torch.allclose(covariance[0, 0], category_covar[0, 0])


def test_probability_mixture_interpolates_category_covariance() -> None:
    kernel = ExpectedCategoricalKernel(2).double()
    one_hot = torch.eye(2, dtype=torch.double)
    mixture = torch.tensor([[0.25, 0.75]], dtype=torch.double)

    mixed_covariance = kernel(mixture, one_hot).to_dense().squeeze(0)
    expected = 0.25 * kernel.category_covar()[0] + 0.75 * kernel.category_covar()[1]

    assert torch.allclose(mixed_covariance, expected)


@pytest.mark.parametrize(
    "probabilities",
    [
        torch.tensor([[1.1, -0.1]], dtype=torch.double),
        torch.tensor([[0.4, 0.4]], dtype=torch.double),
        torch.tensor([[float("nan"), float("nan")]], dtype=torch.double),
    ],
)
def test_invalid_probability_simplex_is_rejected(probabilities) -> None:
    X = torch.zeros(1, 1, dtype=torch.double)
    Y = torch.zeros(1, 1, dtype=torch.double)
    with pytest.raises(ValueError):
        UncertainCategoricalSingleTaskGP(X, probabilities, Y)


def test_exact_mll_gradients_and_raw_data() -> None:
    X, P, Y = _data()
    model = UncertainCategoricalSingleTaskGP(X, P, Y)
    mll = model.make_mll()
    output = model(*model.train_inputs)
    loss = -mll(output, model.train_targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.category_kernel.category_factor.grad is not None
    assert torch.equal(model.raw_train_X_cont, X)
    assert torch.equal(model.raw_train_category_probabilities, P)


def test_posterior_and_qmc_contract_on_augmented_input() -> None:
    X, P, Y = _data()
    model = UncertainCategoricalSingleTaskGP(X, P, Y)
    posterior = model.posterior_with_category_probabilities(X[:3], P[:3])
    assert posterior.mean.shape == torch.Size([3, 1])

    candidate = model.augment_inputs(X[:2], P[:2]).unsqueeze(0)
    acquisition = qUpperConfidenceBound(model=model, beta=0.2)
    value = acquisition(candidate)
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_dtype_migration_and_state_dict_roundtrip() -> None:
    X, P, Y = _data()
    model = UncertainCategoricalSingleTaskGP(X, P, Y).float()
    assert model.raw_train_X_cont.dtype == torch.float
    assert model.raw_train_category_probabilities.dtype == torch.float

    clone = UncertainCategoricalSingleTaskGP(X, P, Y)
    clone.load_state_dict(UncertainCategoricalSingleTaskGP(X, P, Y).state_dict())
