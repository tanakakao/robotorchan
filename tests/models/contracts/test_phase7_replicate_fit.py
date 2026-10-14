"""Phase 7 reviewed model-gap integration for exact replicate-noise surrogates."""

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import (
    MixedReplicateNoiseSingleTaskGP,
    ReplicateNoiseSingleTaskGP,
)
from robotorchan.optim.dispatch import optimize_acqf


def _replicate_observations(mixed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    continuous = torch.linspace(0.1, 0.9, 6, dtype=torch.double).unsqueeze(-1)
    if mixed:
        categories = torch.tensor([[0.0], [1.0]] * 3, dtype=torch.double)
        designs = torch.cat((continuous, categories), dim=-1)
    else:
        designs = continuous
    raw_X = designs.repeat_interleave(2, dim=0)
    truth = torch.sin(4.0 * continuous)
    if mixed:
        truth = truth + 0.1 * designs[:, 1:]
    noise = torch.tensor([-0.02, 0.02], dtype=torch.double).repeat(6).unsqueeze(-1)
    raw_Y = truth.repeat_interleave(2, dim=0) + noise
    return raw_X, raw_Y


@pytest.mark.parametrize("mixed", [False, True])
def test_replicate_noise_fit_posterior_and_candidate_optimization(mixed: bool) -> None:
    raw_X, raw_Y = _replicate_observations(mixed)
    if mixed:
        model = MixedReplicateNoiseSingleTaskGP.from_replicates(
            raw_X, raw_Y, cat_dims=[1]
        )
    else:
        model = ReplicateNoiseSingleTaskGP.from_replicates(raw_X, raw_Y)

    assert model.supports_fit is True
    assert model.make_mll().model is model
    assert model.raw_train_X.shape[0] == 6
    assert model.raw_train_Yvar is not None
    assert (model.raw_train_Yvar > 0).all()
    assert torch.equal(model.raw_replicate_X, raw_X)
    assert torch.equal(model.raw_replicate_Y, raw_Y)
    assert (model.replicate_counts == 2).all()

    assert model.fit(optimizer_kwargs={"options": {"maxiter": 3}}) is None
    assert not model.training
    with torch.no_grad():
        posterior = model.posterior(model.raw_train_X[:2])
    assert posterior.mean.shape == (2, 1)
    assert torch.isfinite(posterior.mean).all()

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=71)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=model.raw_train_Y.max().item(),
        sampler=sampler,
    )
    if mixed:
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        candidate, score = optimize_acqf(
            acquisition,
            bounds,
            q=1,
            optimizer="sobol",
            fixed_features={1: 0.0},
            raw_samples=32,
            seed=71,
        )
    else:
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        candidate, score = optimize_acqf(
            acquisition,
            bounds,
            q=2,
            optimizer="sobol",
            raw_samples=32,
            seed=71,
        )
    assert candidate.shape == (1 if mixed else 2, raw_X.shape[-1])
    assert torch.isfinite(candidate).all()
    if mixed:
        assert torch.all(candidate[:, 1] == 0)
    assert torch.isfinite(score).all()
    assert ((candidate >= bounds[0]) & (candidate <= bounds[1])).all()


def test_specialized_replicate_subclasses_do_not_inherit_fit_opt_in() -> None:
    class CustomReplicate(ReplicateNoiseSingleTaskGP):
        pass

    class CustomMixedReplicate(MixedReplicateNoiseSingleTaskGP):
        pass

    assert not CustomReplicate.supports_fit
    assert not CustomMixedReplicate.supports_fit
