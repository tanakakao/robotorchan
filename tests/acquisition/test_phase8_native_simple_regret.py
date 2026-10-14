"""Phase 8 acquisition-gap coverage for BoTorch-native qSimpleRegret."""

import inspect

import torch
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.acquisition.objective import LinearMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition.capabilities import AcquisitionPurpose, PosteriorRequirement
from robotorchan.acquisition.compatibility import check_model_acquisition_compatibility
from robotorchan.acquisition.optimizer_compatibility import (
    CandidateOptimizer,
    check_acquisition_optimizer_compatibility,
)
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models import SingleTaskGP
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.workflow.problem import ProblemPurpose, ProblemSpec
from robotorchan.workflow.recommendation import recommend_compatible_workflows


def test_qsimple_regret_is_metadata_only_and_compatible() -> None:
    """Expose upstream semantics without copying or aliasing the implementation."""
    entry = ACQUISITION_REGISTRY["qSimpleRegret"]
    assert entry.capabilities.purpose is AcquisitionPurpose.BAYESIAN_OPTIMIZATION
    assert entry.capabilities.posterior_requirement is PosteriorRequirement.POSTERIOR_SAMPLES
    assert entry.capabilities.max_q is None
    assert not entry.capabilities.supports_constraints
    assert not entry.capabilities.one_shot
    assert entry.implementation_strategy.startswith("BoTorch-native")
    assert qSimpleRegret.__module__.startswith("botorch.")
    assert {"model", "sampler", "objective"} <= set(inspect.signature(qSimpleRegret).parameters)
    assert check_model_acquisition_compatibility("SingleTaskGP", "qSimpleRegret").compatible
    assert check_acquisition_optimizer_compatibility(
        "qSimpleRegret", CandidateOptimizer.CONTINUOUS, q=2
    ).compatible


def test_qsimple_regret_is_recommended_only_for_unconstrained_single_objective() -> None:
    default = ProblemSpec(purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION)
    selected = {entry.acquisition_name for entry in recommend_compatible_workflows(default)}
    assert "qSimpleRegret" in selected
    constrained = ProblemSpec(purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION, constrained=True)
    selected_constrained = {
        entry.acquisition_name for entry in recommend_compatible_workflows(constrained)
    }
    assert "qSimpleRegret" not in selected_constrained


def test_native_qsimple_regret_scalar_objective_q_batch_and_dispatch() -> None:
    """Complete GP-posterior-sampler-objective-acquisition-optimizer execution."""
    X = torch.linspace(0.1, 0.9, 9, dtype=torch.double).unsqueeze(-1)
    Y = torch.cat((torch.sin(3 * X), torch.cos(3 * X)), dim=-1)
    model = SingleTaskGP(X, Y)
    model.eval()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=42)
    objective = LinearMCObjective(weights=torch.tensor([0.6, 0.4], dtype=X.dtype))
    acquisition = qSimpleRegret(model=model, sampler=sampler, objective=objective)
    candidates = X[:6].reshape(3, 2, 1)
    with torch.no_grad():
        values = acquisition(candidates)
    assert values.shape == (3,)
    assert torch.isfinite(values).all()

    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidate, score = optimize_acqf(
        acquisition, bounds, q=2, optimizer="sobol", raw_samples=32, seed=42
    )
    assert candidate.shape == (2, 1)
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(score).all()
    assert ((candidate >= bounds[0]) & (candidate <= bounds[1])).all()
