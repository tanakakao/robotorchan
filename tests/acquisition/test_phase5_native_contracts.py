"""Phase 5 native BoTorch acquisition API and optimizer interoperability contracts."""

import inspect

import pytest
import torch
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models import SingleTaskGP


@pytest.mark.parametrize(
    ("name", "acquisition_type", "required"),
    [
        (
            "qLogExpectedImprovement",
            qLogExpectedImprovement,
            {"model", "best_f", "sampler", "constraints"},
        ),
        (
            "qLogNoisyExpectedImprovement",
            qLogNoisyExpectedImprovement,
            {"model", "X_baseline", "sampler", "constraints"},
        ),
        (
            "qUpperConfidenceBound",
            qUpperConfidenceBound,
            {"model", "beta", "sampler"},
        ),
        (
            "qKnowledgeGradient",
            qKnowledgeGradient,
            {"model", "num_fantasies"},
        ),
    ],
)
def test_native_acquisitions_have_expected_constructor_contracts(
    name: str,
    acquisition_type: type,
    required: set[str],
) -> None:
    """Registry names must resolve to native BoTorch implementations."""
    metadata = ACQUISITION_REGISTRY[name]
    assert metadata.implementation_strategy.startswith("BoTorch-native")
    assert required <= set(inspect.signature(acquisition_type).parameters)
    assert acquisition_type.__module__.startswith("botorch.")


def test_scalarized_posterior_transform_sampler_and_q_batch_optimizer() -> None:
    """An unwrapped native MC acquisition can optimize multi-output GP inputs."""
    X = torch.linspace(0.1, 0.9, 9, dtype=torch.double).unsqueeze(-1)
    Y = torch.cat((torch.sin(3 * X), torch.cos(3 * X)), dim=-1)
    model = SingleTaskGP(X, Y)
    model.eval()
    transform = ScalarizedPosteriorTransform(
        weights=torch.tensor([0.6, 0.4], dtype=torch.double),
    )
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=314)
    acquisition = qUpperConfidenceBound(
        model=model,
        beta=0.1,
        sampler=sampler,
        posterior_transform=transform,
    )
    candidate_batch = X[:6].reshape(3, 2, 1)
    with torch.no_grad():
        score = acquisition(candidate_batch)
    assert score.shape == (3,)
    assert torch.isfinite(score).all()

    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidates, value = optimize_acqf(
        acq_function=acquisition,
        bounds=bounds,
        q=2,
        num_restarts=2,
        raw_samples=16,
        options={"maxiter": 5},
    )
    assert candidates.shape == (2, 1)
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()
    assert ((candidates >= bounds[0]) & (candidates <= bounds[1])).all()
