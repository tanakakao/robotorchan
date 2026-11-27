"""Regression tests for decoupled workflow requirements."""

import pytest

from robotorchan.workflow import ProblemSpec, check_compatibility


def test_decoupled_requires_multi_output():
    with pytest.raises(ValueError, match="multi-output"):
        ProblemSpec(purpose="bayesian_optimization", decoupled=True)


def test_pending_requires_decoupled():
    with pytest.raises(ValueError, match="requires decoupled"):
        ProblemSpec(purpose="active_learning", masked_pending=True)


def test_pending_rejects_unsupported_acquisition():
    spec = ProblemSpec(
        purpose="bayesian_optimization",
        output_type="multi",
        decoupled=True,
        masked_pending=True,
    )
    decision = check_compatibility(
        spec,
        model="HeterogeneousModel",
        acquisition="qDecoupledParticleKnowledgeGradient",
        optimizer="optimize_acqf_decoupled",
    )
    assert "requires masked pending evaluation support" in decision.reasons


def test_decoupled_rejects_scalar_optimizer():
    spec = ProblemSpec(
        purpose="bayesian_optimization",
        output_type="multi",
        decoupled=True,
    )
    decision = check_compatibility(
        spec,
        model="HeterogeneousModel",
        acquisition="qDecoupledParticleKnowledgeGradient",
        optimizer="botorch",
    )
    assert "requires the dedicated decoupled optimizer without a search strategy" in (
        decision.reasons
    )
