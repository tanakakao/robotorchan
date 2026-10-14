"""Phase 10 API and numerical-behavior regression contracts."""

import pytest
import torch
from botorch.acquisition.monte_carlo import qProbabilityOfImprovement, qSimpleRegret
from botorch.acquisition.objective import LinearMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler

import robotorchan.acquisition as acquisition_package
from robotorchan.acquisition.capabilities import AcquisitionPurpose
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models import SingleTaskGP
from robotorchan.models.robust.replicate_noise import (
    MixedReplicateNoiseSingleTaskGP,
    ReplicateNoiseSingleTaskGP,
)


def test_botorch_native_registry_entries_are_not_local_public_api_aliases() -> None:
    """Registry metadata must not silently expand the robotorchan public API."""
    for name, entry in ACQUISITION_REGISTRY.items():
        if entry.capabilities.purpose is AcquisitionPurpose.BAYESIAN_OPTIMIZATION:
            assert entry.implementation_strategy.startswith("BoTorch-native")
            assert name not in acquisition_package.__all__
            assert not hasattr(acquisition_package, name)


@pytest.mark.parametrize("q", [1, 2])
def test_qsimple_regret_matches_mc_expected_best_sample(q: int) -> None:
    """Guard the sampled batch maximum, not just finite acquisition output."""
    train_X = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cat((torch.sin(3 * train_X), torch.cos(3 * train_X)), dim=-1)
    model = SingleTaskGP(train_X, train_Y)
    model.eval()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=100)
    objective = LinearMCObjective(weights=torch.tensor([0.7, 0.3], dtype=torch.double))
    acqf = qSimpleRegret(model=model, sampler=sampler, objective=objective)
    X = torch.linspace(0.15, 0.75, 2 * q, dtype=torch.double).reshape(2, q, 1)

    with torch.no_grad():
        actual = acqf(X)
        samples = sampler(model.posterior(X))
        expected = objective(samples, X=X).max(dim=-1).values.mean(dim=0)

    assert actual.shape == (2,)
    torch.testing.assert_close(actual, expected)


def test_qpi_outcome_feasibility_numerical_ordering() -> None:
    """An impossible outcome constraint must reduce native qPI probability."""
    train_X = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(3 * train_X)
    model = SingleTaskGP(train_X, train_Y)
    model.eval()
    X = train_X[:6].reshape(3, 2, 1)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=101)

    def always_feasible(samples: torch.Tensor) -> torch.Tensor:
        return samples[..., 0] * 0 - 10.0

    def never_feasible(samples: torch.Tensor) -> torch.Tensor:
        return samples[..., 0] * 0 + 10.0

    accepted = qProbabilityOfImprovement(
        model=model, best_f=train_Y.min(), sampler=sampler,
        constraints=[always_feasible],
    )
    rejected = qProbabilityOfImprovement(
        model=model, best_f=train_Y.min(), sampler=sampler,
        constraints=[never_feasible],
    )
    with torch.no_grad():
        accepted_score = accepted(X)
        rejected_score = rejected(X)

    assert accepted_score.shape == rejected_score.shape == (3,)
    assert torch.isfinite(accepted_score).all()
    assert torch.isfinite(rejected_score).all()
    assert (accepted_score >= rejected_score).all()
    assert (rejected_score < 1e-4).all()


def test_replicate_fitting_opt_in_remains_explicit() -> None:
    """Phase 7 fitting support must not leak to arbitrary derived model types."""
    assert ReplicateNoiseSingleTaskGP.supports_fit is True
    assert MixedReplicateNoiseSingleTaskGP.supports_fit is True

    class CustomReplicate(ReplicateNoiseSingleTaskGP):
        pass

    class CustomMixedReplicate(MixedReplicateNoiseSingleTaskGP):
        pass

    assert CustomReplicate.supports_fit is False
    assert CustomMixedReplicate.supports_fit is False
