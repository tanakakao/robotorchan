"""Surrogate models that represent uncertain inputs internally.

These are model implementations, not candidate input-perturbation scenarios.
Scenario utilities for robust optimization live under ``robotorchan.uncertainty``.
"""

from robotorchan.models.uncertain.uncertain_categorical import (
    AugmentedUncertainCategoricalKernel,
    ExpectedCategoricalKernel,
    UncertainCategoricalSingleTaskGP,
)
from robotorchan.models.uncertain.uncertain_input import (
    GaussianUncertainInputKernel,
    MixedGaussianUncertainInputKernel,
    MixedUncertainInputSingleTaskGP,
    UncertainInputSingleTaskGP,
)

__all__ = [
    "AugmentedUncertainCategoricalKernel",
    "ExpectedCategoricalKernel",
    "GaussianUncertainInputKernel",
    "MixedGaussianUncertainInputKernel",
    "MixedUncertainInputSingleTaskGP",
    "UncertainCategoricalSingleTaskGP",
    "UncertainInputSingleTaskGP",
]
