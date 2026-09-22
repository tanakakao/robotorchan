"""Tests for capability-aware BoTorch sampler selection."""

import pytest
import torch
from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition.samplers import make_model_sampler


def test_make_model_sampler_selects_gaussian_sampler() -> None:
    sampler = make_model_sampler("SingleTaskGP", torch.Size([8]))
    assert isinstance(sampler, SobolQMCNormalSampler)


def test_make_model_sampler_selects_ensemble_sampler() -> None:
    sampler = make_model_sampler("RandomForestSurrogate", torch.Size([8]))
    assert isinstance(sampler, IndexSampler)


def test_make_model_sampler_rejects_unsupported_sampling() -> None:
    with pytest.raises(ValueError, match="does not advertise posterior sampling support"):
        make_model_sampler("PairwiseGP", torch.Size([8]))
