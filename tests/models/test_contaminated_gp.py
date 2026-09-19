"""Tests for the contamination-mixture observation GP."""

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

from robotorchan.models.contaminated import ContaminatedSingleTaskGP


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(0, 1, 12, dtype=torch.double).unsqueeze(-1)
    Y = torch.sin(2 * torch.pi * X)
    return X, Y


def test_training_loss_is_finite_and_has_gradients() -> None:
    X, Y = _data()
    model = ContaminatedSingleTaskGP(X, Y, num_likelihood_samples=4)
    loss = model.training_loss()
    loss.backward()

    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.response_model.parameters()
        if parameter.requires_grad
    )


def test_large_residual_has_higher_contamination_probability() -> None:
    X, Y = _data()
    model = ContaminatedSingleTaskGP(
        X,
        Y,
        contamination_probability=0.05,
        inlier_scale=0.05,
        outlier_scale=0.5,
    )
    mean = model.posterior(X[:2]).mean.detach()
    observed = mean.clone()
    observed[1] += 2.0
    probability = model.contamination_diagnostic(X[:2], observed)

    assert torch.all((probability >= 0) & (probability <= 1))
    assert probability[1] > probability[0]


def test_mixture_log_prob_is_stable_for_large_residuals() -> None:
    X, Y = _data()
    model = ContaminatedSingleTaskGP(X, Y)
    residual = torch.tensor([0.0, 1e3], dtype=torch.double)

    value = model._mixture_log_prob(residual)
    assert torch.isfinite(value).all()


def test_invalid_mixture_parameters_are_rejected() -> None:
    X, Y = _data()
    for kwargs in (
        {"contamination_probability": 0.0},
        {"contamination_probability": 1.0},
        {"inlier_scale": 0.0},
        {"inlier_scale": 0.5, "outlier_scale": 0.5},
    ):
        try:
            ContaminatedSingleTaskGP(X, Y, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid contamination parameters must be rejected.")


def test_raw_data_posterior_and_qmc_contract() -> None:
    X, Y = _data()
    model = ContaminatedSingleTaskGP(X, Y)

    assert torch.equal(model.raw_train_X, X)
    assert torch.equal(model.raw_train_Y, Y)
    posterior = model.posterior(X[:3])
    assert posterior.mean.shape == torch.Size([3, 1])

    acquisition = qUpperConfidenceBound(model=model, beta=0.2)
    value = acquisition(X[:2].unsqueeze(0))
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_make_mll_is_explicitly_unsupported() -> None:
    X, Y = _data()
    model = ContaminatedSingleTaskGP(X, Y)

    try:
        model.make_mll()
    except RuntimeError as error:
        assert "training_loss" in str(error)
    else:
        raise AssertionError("Contamination mixture must not expose an exact MLL.")
