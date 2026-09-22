"""Sampler selection from model posterior capability metadata."""

from botorch.sampling.base import MCSampler
from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Size

from robotorchan.models.capabilities import PosteriorSamplingType
from robotorchan.models.registry import MODEL_REGISTRY


def make_model_sampler(model_name: str, sample_shape: Size) -> MCSampler:
    """Construct a BoTorch sampler compatible with a registered model posterior."""
    capabilities = MODEL_REGISTRY[model_name].capabilities
    sampling_type = capabilities.posterior_sampling_type
    if sampling_type is PosteriorSamplingType.GAUSSIAN:
        return SobolQMCNormalSampler(sample_shape=sample_shape)
    if sampling_type is PosteriorSamplingType.ENSEMBLE:
        return IndexSampler(sample_shape=sample_shape)
    raise ValueError(f"{model_name} does not advertise posterior sampling support")
