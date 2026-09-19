import importlib

import pytest
import torch

from robotorchan.optim import ALEBOStrategy

MODELS = importlib.import_module("robotorchan.models")


def test_nominal_mixed_alebo_model_is_not_exposed() -> None:
    assert "MixedALEBOGP" not in MODELS.__all__
    assert not hasattr(MODELS, "MixedALEBOGP")


def test_alebo_linear_reconstruction_does_not_define_categorical_feasibility() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    strategy = ALEBOStrategy(bounds, embedding_dim=1, seed=0)
    samples = strategy.sample_feasible(32, seed=1)
    candidates = strategy.project(samples)

    assert bool(((candidates >= bounds[0]) & (candidates <= bounds[1])).all())
    categorical_like = candidates[:, 1]
    allowed = torch.tensor([0.0, 1.0, 2.0], dtype=candidates.dtype)
    is_allowed = (categorical_like.unsqueeze(-1) == allowed).any(dim=-1)

    # ALEBO reconstructs this coordinate continuously. This is the semantic
    # reason a surrogate-only MixedALEBOGP must not be exposed.
    assert not bool(is_allowed.all())


@pytest.mark.parametrize("name", ["MixedPairwiseGP", "MixedALEBOGP"])
def test_unverified_specialized_mixed_wrappers_are_not_public(name: str) -> None:
    assert name not in MODELS.__all__
    assert not hasattr(MODELS, name)
