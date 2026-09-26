"""Regression guardrails for robust Kronecker design."""

from gpytorch.likelihoods import MultitaskGaussianLikelihood


def test_multitask_likelihood_has_no_single_task_noise_covar_contract() -> None:
    """Relevance-pursuit mixin cannot wrap the Kronecker likelihood directly."""
    likelihood = MultitaskGaussianLikelihood(num_tasks=2)

    assert not hasattr(likelihood, "noise_covar")
    assert hasattr(likelihood, "task_noises")
