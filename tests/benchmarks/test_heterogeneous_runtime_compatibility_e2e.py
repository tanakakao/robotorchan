"""Phase 17: dtype, device, batch, and fixed-feature compatibility."""

import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition.composition import make_qei_acquisition
from robotorchan.benchmarks.heterogeneous_synthetic import evaluate_truth, observe
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.semantics.objectives import RegressionObjective
from robotorchan.semantics.problem import ProblemSemantics


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("q", [1, 2])
def test_dtype_q_batch_and_fixed_feature(dtype: torch.dtype, q: int) -> None:
    """Preserve tensor contracts through fitted qEI and fixed-feature optimization."""
    seed = 1717
    generator = torch.Generator().manual_seed(seed)
    train_X = torch.rand(18, 3, generator=generator, dtype=dtype)
    train_Y = observe(train_X, seed=seed + 100).strength.unsqueeze(-1)
    regression = SingleTaskGP(
        train_X,
        train_Y,
        train_Yvar=torch.full_like(train_Y, 0.02**2),
    )
    fit_gpytorch_mll(regression.make_mll())
    regression.eval()
    model = HeterogeneousModel(regression, output_names=["strength"])
    semantics = ProblemSemantics(objectives=(RegressionObjective("strength"),))
    acquisition = make_qei_acquisition(
        model,
        semantics,
        best_f=train_Y.max().item(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=seed),
    )
    X = torch.rand(2, q, 3, generator=generator, dtype=dtype)
    with torch.no_grad():
        posterior = regression.posterior(X)
        value = acquisition(X)
    assert posterior.mean.shape == (2, q, 1)
    assert posterior.mean.dtype == dtype
    assert value.shape == (2,)
    assert torch.isfinite(value).all()

    bounds = torch.stack((torch.zeros(3, dtype=dtype), torch.ones(3, dtype=dtype)))
    candidate, score = optimize_acqf(
        acquisition,
        bounds,
        q=q,
        optimizer="sobol",
        raw_samples=32,
        fixed_features={2: 0.5},
        seed=seed + q,
    )
    assert candidate.shape == (q, 3)
    assert candidate.dtype == dtype
    assert torch.isfinite(score).all()
    assert ((candidate >= 0) & (candidate <= 1)).all()
    torch.testing.assert_close(candidate[:, 2], torch.full((q,), 0.5, dtype=dtype))
    with torch.no_grad():
        strength, _, _ = evaluate_truth(candidate)
    assert torch.isfinite(strength).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_native_posterior_and_qei() -> None:
    """Check CUDA model and acquisition evaluation without assuming GPU CI."""
    device = torch.device("cuda")
    X = torch.rand(16, 3, device=device, dtype=torch.double)
    Y = evaluate_truth(X)[0].unsqueeze(-1)
    regression = SingleTaskGP(
        X,
        Y,
        train_Yvar=torch.full_like(Y, 0.02**2),
    )
    regression.eval()
    acquisition = make_qei_acquisition(
        HeterogeneousModel(regression),
        ProblemSemantics(objectives=(RegressionObjective(0),)),
        best_f=Y.max().item(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8]), seed=1717),
    )
    with torch.no_grad():
        posterior = regression.posterior(X[:2])
        values = acquisition(X[:2].unsqueeze(0))
    assert posterior.mean.device.type == "cuda"
    assert values.device.type == "cuda"
    assert torch.isfinite(values).all()
