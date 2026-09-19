"""Tests for the mixed nonstationary Gibbs-kernel surrogate."""

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

from robotorchan.models import MixedNonstationarySingleTaskGP


def _data():
    x = torch.linspace(0.05, 0.95, 8, dtype=torch.double)
    category = torch.tensor([0, 1] * 4, dtype=torch.double)
    X = torch.stack([x, category], dim=-1)
    Y = (torch.sin(2 * torch.pi * x) + 0.2 * category).unsqueeze(-1)
    return X, Y


def test_mixed_nonstationary_uses_gibbs_only_on_continuous_dims() -> None:
    X, Y = _data()
    model = MixedNonstationarySingleTaskGP(X, Y, cat_dims=[1])
    assert model.cat_dims == (1,)
    assert model.continuous_dims == (0,)
    additive, interaction = model.local_lengthscale(X[:3])
    assert additive.shape == torch.Size([3, 1])
    assert interaction.shape == torch.Size([3, 1])
    assert torch.isfinite(additive).all()
    assert torch.isfinite(interaction).all()


def test_mixed_nonstationary_supports_exact_mll_and_qucb() -> None:
    X, Y = _data()
    model = MixedNonstationarySingleTaskGP(X, Y, cat_dims=[1])
    assert model.supports_mll
    mll = model.make_mll()
    assert mll is not None
    acq = qUpperConfidenceBound(model=model, beta=0.2)
    assert torch.isfinite(acq(X[:2].unsqueeze(0))).all()


def test_mixed_nonstationary_rejects_all_categorical_inputs() -> None:
    X, Y = _data()
    try:
        MixedNonstationarySingleTaskGP(X, Y, cat_dims=[0, 1])
    except ValueError as error:
        assert "at least one continuous dimension" in str(error)
    else:
        raise AssertionError("Expected all-categorical mixed nonstationary input to be rejected.")
