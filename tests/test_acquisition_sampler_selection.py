"""Tests for capability-aware BoTorch sampler selection."""

import pytest
import torch
from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition.samplers import make_model_sampler
from robotorchan.models import EnsembleMapSaasSingleTaskGP, MixedEnsembleMapSaasSingleTaskGP


def test_make_model_sampler_selects_gaussian_sampler() -> None:
    sampler = make_model_sampler("SingleTaskGP", torch.Size([8]))
    assert isinstance(sampler, SobolQMCNormalSampler)


def test_make_model_sampler_selects_ensemble_sampler() -> None:
    sampler = make_model_sampler("RandomForestSurrogate", torch.Size([8]))
    assert isinstance(sampler, IndexSampler)


def test_make_model_sampler_rejects_unsupported_sampling() -> None:
    with pytest.raises(ValueError, match="does not advertise posterior sampling support"):
        make_model_sampler("PairwiseGP", torch.Size([8]))


def test_map_saas_ensemble_sampler_runs_against_gaussian_posterior() -> None:
    train_x = torch.rand(8, 3, dtype=torch.double)
    train_y = torch.sin(train_x[:, :1] * 3.0)
    model = EnsembleMapSaasSingleTaskGP(train_x, train_y, num_taus=2)
    sampler = make_model_sampler("EnsembleMapSaasSingleTaskGP", torch.Size([4]))
    samples = sampler(model.posterior(torch.rand(2, 3, dtype=torch.double)))
    assert isinstance(sampler, SobolQMCNormalSampler)
    assert torch.isfinite(samples).all()


def test_mixed_map_saas_ensemble_sampler_runs_against_gaussian_posterior() -> None:
    continuous = torch.linspace(0.0, 1.0, 8, dtype=torch.double)
    categorical = torch.tensor([0.0, 1.0] * 4, dtype=torch.double)
    train_x = torch.stack([continuous, categorical], dim=-1)
    train_y = torch.sin(continuous * 3.0).unsqueeze(-1)
    model = MixedEnsembleMapSaasSingleTaskGP(
        train_x,
        train_y,
        cat_dims=[1],
        num_taus=2,
    )
    sampler = make_model_sampler("MixedEnsembleMapSaasSingleTaskGP", torch.Size([4]))
    candidate = torch.tensor([[0.5, 1.0]], dtype=torch.double)
    samples = sampler(model.posterior(candidate))
    assert isinstance(sampler, SobolQMCNormalSampler)
    assert torch.isfinite(samples).all()


def test_ngboost_sampler_matches_distribution_posterior_contract() -> None:
    sampler = make_model_sampler("NGBoostSurrogate", torch.Size([8]))
    assert isinstance(sampler, SobolQMCNormalSampler)


def test_robust_multifidelity_sampler_uses_gaussian_path() -> None:
    for model_name in (
        "ReplicateNoiseMultiFidelityGP",
        "HeteroskedasticMultiFidelityGP",
    ):
        sampler = make_model_sampler(model_name, torch.Size([8]))
        assert isinstance(sampler, SobolQMCNormalSampler)
