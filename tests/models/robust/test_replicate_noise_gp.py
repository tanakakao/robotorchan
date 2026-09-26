"""Tests for replicate-aware observation-noise GP."""

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

from robotorchan.models.robust.replicate_noise import ReplicateNoiseSingleTaskGP


def _replicates() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]], dtype=torch.double)
    Y = torch.tensor([[1.0], [2.0], [3.0], [4.0], [4.0], [7.0]], dtype=torch.double)
    return X, Y


def test_replicates_are_aggregated_with_variance_of_mean() -> None:
    X, Y = _replicates()
    model = ReplicateNoiseSingleTaskGP.from_replicates(X, Y, noise_floor=1e-12)

    assert torch.equal(model.raw_train_X, torch.tensor([[0.0], [1.0]], dtype=torch.double))
    assert torch.allclose(model.raw_train_Y, torch.tensor([[2.0], [5.0]], dtype=torch.double))
    assert torch.equal(model.replicate_counts, torch.tensor([3, 3]))
    assert torch.allclose(
        model.replicate_variance,
        torch.tensor([[1.0], [3.0]], dtype=torch.double),
    )
    assert torch.allclose(
        model.raw_train_Yvar,
        torch.tensor([[1.0 / 3.0], [1.0]], dtype=torch.double),
    )


def test_raw_replicates_are_retained() -> None:
    X, Y = _replicates()
    model = ReplicateNoiseSingleTaskGP.from_replicates(X, Y)

    assert torch.equal(model.raw_replicate_X, X)
    assert torch.equal(model.raw_replicate_Y, Y)


def test_constant_replicates_use_noise_floor() -> None:
    X = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.double)
    Y = torch.tensor([[1.0], [1.0], [2.0], [2.0]], dtype=torch.double)
    model = ReplicateNoiseSingleTaskGP.from_replicates(X, Y, noise_floor=1e-5)

    assert torch.allclose(model.raw_train_Yvar, torch.full((2, 1), 1e-5, dtype=torch.double))


def test_single_observation_group_is_rejected() -> None:
    X = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.double)
    Y = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.double)

    try:
        ReplicateNoiseSingleTaskGP.from_replicates(X, Y)
    except ValueError:
        pass
    else:
        raise AssertionError("Singleton replicate groups must be rejected.")


def test_posterior_and_mc_acquisition() -> None:
    X, Y = _replicates()
    model = ReplicateNoiseSingleTaskGP.from_replicates(X, Y)
    candidate = torch.tensor([[0.25], [0.75]], dtype=torch.double)

    posterior = model.posterior(candidate)
    assert posterior.mean.shape == torch.Size([2, 1])

    acquisition = qUpperConfidenceBound(model=model, beta=0.2)
    value = acquisition(candidate.unsqueeze(0))
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_make_mll_uses_exact_gp_contract() -> None:
    X, Y = _replicates()
    model = ReplicateNoiseSingleTaskGP.from_replicates(X, Y)

    mll = model.make_mll()
    output = model(*model.train_inputs)
    loss = -mll(output, model.train_targets)
    assert torch.isfinite(loss)
