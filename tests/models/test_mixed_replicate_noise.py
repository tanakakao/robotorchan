"""Tests for mixed replicate-derived observation noise."""

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

from robotorchan.models import MixedReplicateNoiseSingleTaskGP


def _replicates():
    X = torch.tensor(
        [[0.1, 0.0], [0.1, 0.0], [0.5, 1.0], [0.5, 1.0], [0.9, 0.0], [0.9, 0.0]],
        dtype=torch.double,
    )
    Y = torch.tensor([[0.0], [0.2], [0.8], [1.2], [0.1], [0.3]], dtype=torch.double)
    return X, Y


def test_mixed_replicate_noise_reuses_shared_aggregation_contract() -> None:
    X, Y = _replicates()
    model = MixedReplicateNoiseSingleTaskGP.from_replicates(X, Y, cat_dims=[1])
    assert model.cat_dims == [1]
    assert model.raw_train_X.shape == torch.Size([3, 2])
    assert torch.equal(model.raw_replicate_X, X)
    assert torch.equal(model.raw_replicate_Y, Y)
    assert torch.equal(model.replicate_counts, torch.tensor([2, 2, 2]))
    assert torch.allclose(model.raw_train_Y.squeeze(-1), torch.tensor([0.1, 1.0, 0.2]))


def test_mixed_replicate_noise_supports_exact_mll_and_qucb() -> None:
    X, Y = _replicates()
    model = MixedReplicateNoiseSingleTaskGP.from_replicates(X, Y, cat_dims=[1])
    assert model.supports_mll
    assert model.make_mll() is not None
    acq = qUpperConfidenceBound(model=model, beta=0.2)
    assert torch.isfinite(acq(model.raw_train_X[:2].unsqueeze(0))).all()
