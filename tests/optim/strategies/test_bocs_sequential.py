"""Focused sequential BOCS tests on a finite binary search space."""

import pytest
import torch

from robotorchan.optim import propose_bocs_candidate, run_bocs_sequential


def test_sequential_proposals_are_unseen_and_objective_is_evaluated():
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.double)
    Y = X.sum(dim=-1, keepdim=True)
    proposal = propose_bocs_candidate(X, Y)
    assert proposal.shape == (1, 2)
    assert not bool((proposal == X).all(dim=-1).any())

    X_new, Y_new = run_bocs_sequential(
        X, Y, lambda points: points.sum(dim=-1, keepdim=True), steps=2
    )
    assert X_new.shape == (4, 2)
    assert Y_new.shape == (4, 1)
    assert torch.unique(X_new, dim=0).shape[0] == 4
    assert torch.allclose(Y_new, X_new.sum(dim=-1, keepdim=True))


def test_exhaustion_and_dimension_limit():
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    Y = X.clone()
    with pytest.raises(ValueError, match="exhausted"):
        propose_bocs_candidate(X, Y)
    with pytest.raises(ValueError, match="limit"):
        propose_bocs_candidate(X, Y, max_dimensions=0)
