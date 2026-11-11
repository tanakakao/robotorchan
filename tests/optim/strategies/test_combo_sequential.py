"""Focused sequential binary COMBO workflow tests."""

import pytest
import torch

from robotorchan.optim import propose_combo_candidate, run_combo_sequential


def test_combo_proposal_and_sequential_observations():
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.double)
    Y = X.sum(dim=-1, keepdim=True)
    candidate = propose_combo_candidate(X, Y, fit_model=False)
    assert candidate.shape == (1, 2)
    assert not bool((candidate == X).all(dim=-1).any())
    new_X, new_Y = run_combo_sequential(
        X,
        Y,
        lambda points: points.sum(dim=-1, keepdim=True),
        steps=2,
        fit_model=False,
    )
    assert new_X.shape == (4, 2)
    assert torch.unique(new_X, dim=0).shape[0] == 4
    assert torch.allclose(new_Y, new_X.sum(dim=-1, keepdim=True))


def test_combo_exhaustion_and_dimension_limit():
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    Y = X.clone()
    with pytest.raises(ValueError, match="exhausted"):
        propose_combo_candidate(X, Y, fit_model=False)
    with pytest.raises(ValueError, match="limit"):
        propose_combo_candidate(X, Y, max_dimensions=0, fit_model=False)
