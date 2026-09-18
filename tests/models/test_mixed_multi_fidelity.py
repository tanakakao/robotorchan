from __future__ import annotations

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import MixedSingleTaskMultiFidelityGP


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(705)
    design = torch.rand(24, 2, dtype=torch.double)
    category = torch.randint(0, 3, (24, 1)).to(dtype=torch.double)
    fidelity = torch.tensor([0.5, 0.75, 1.0], dtype=torch.double).repeat(8).unsqueeze(-1)
    X = torch.cat((design[:, :1], category, design[:, 1:], fidelity), dim=-1)
    Y = torch.sin(2 * torch.pi * design[:, :1]) + 0.2 * category + 0.5 * fidelity
    return X, Y


def test_mixed_multifidelity_retains_raw_data_and_dimension_roles() -> None:
    X, Y = _data()
    model = MixedSingleTaskMultiFidelityGP(X, Y, cat_dims=[1], fidelity_dims=[3])
    assert model.cat_dims == [1]
    assert model.fidelity_dims == [3]
    torch.testing.assert_close(model.raw_train_X, X)
    torch.testing.assert_close(model.raw_train_Y, Y)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_mixed_multifidelity_posterior_accepts_original_q_batch() -> None:
    X, Y = _data()
    model = MixedSingleTaskMultiFidelityGP(X, Y, cat_dims=[1], fidelity_dims=[3])
    model.eval()
    model.likelihood.eval()
    candidates = X[:8].reshape(2, 4, 4)
    posterior = model.posterior(candidates)
    assert posterior.mean.shape == torch.Size([2, 4, 1])
    assert torch.isfinite(posterior.mean).all()


def test_mixed_multifidelity_rejects_overlapping_roles() -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="disjoint"):
        MixedSingleTaskMultiFidelityGP(X, Y, cat_dims=[1, 3], fidelity_dims=[3])


@pytest.mark.parametrize(
    ("cat_dims", "fidelity_dims"),
    [([], [3]), ([1], []), ([1, 1], [3]), ([1], [4])],
)
def test_mixed_multifidelity_validates_dimensions(
    cat_dims: list[int], fidelity_dims: list[int]
) -> None:
    X, Y = _data()
    with pytest.raises(ValueError):
        MixedSingleTaskMultiFidelityGP(X, Y, cat_dims=cat_dims, fidelity_dims=fidelity_dims)
