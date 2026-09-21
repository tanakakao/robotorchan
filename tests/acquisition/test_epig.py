import torch

from robotorchan.acquisition import ExpectedPredictiveInformationGain
from robotorchan.models import SingleTaskGP


def _model() -> SingleTaskGP:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    return SingleTaskGP(train_X, train_Y)


def test_epig_returns_finite_batch_scores() -> None:
    acquisition = ExpectedPredictiveInformationGain(
        _model(),
        torch.linspace(0.0, 1.0, 11, dtype=torch.double).unsqueeze(-1),
        observation_noise=0.05,
    )
    X = torch.tensor([[[0.2]], [[0.7]]], dtype=torch.double)

    value = acquisition(X)

    assert value.shape == torch.Size([2])
    assert torch.isfinite(value).all()
    assert (value >= 0).all()


def test_epig_supports_target_distribution_weights() -> None:
    target_X = torch.tensor([[0.1], [0.9]], dtype=torch.double)
    weights = torch.tensor([1.0, 0.0], dtype=torch.double)
    weighted = ExpectedPredictiveInformationGain(_model(), target_X, target_weights=weights)
    single = ExpectedPredictiveInformationGain(_model(), target_X[:1])
    X = torch.tensor([[[0.2]]], dtype=torch.double)

    assert torch.allclose(weighted(X), single(X), atol=1e-8, rtol=1e-6)


def test_epig_rejects_q_greater_than_one() -> None:
    acquisition = ExpectedPredictiveInformationGain(
        _model(),
        torch.linspace(0.0, 1.0, 5, dtype=torch.double).unsqueeze(-1),
    )

    try:
        acquisition(torch.rand(1, 2, 1, dtype=torch.double))
    except ValueError as error:
        assert "q=1" in str(error)
    else:
        raise AssertionError("Expected q=1 validation.")


def test_epig_rejects_empty_target_distribution() -> None:
    try:
        ExpectedPredictiveInformationGain(_model(), torch.empty(0, 1, dtype=torch.double))
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("Expected empty-target validation.")


def test_epig_rejects_batched_target_distribution() -> None:
    target_X = torch.rand(2, 5, 1, dtype=torch.double)
    try:
        ExpectedPredictiveInformationGain(_model(), target_X)
    except ValueError as error:
        assert "(n_target, d)" in str(error)
    else:
        raise AssertionError("Expected batched-target validation.")
