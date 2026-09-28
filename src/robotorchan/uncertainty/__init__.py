"""Input-perturbation scenario utilities for robust Bayesian optimization.

This namespace describes perturbation distributions evaluated around candidate
inputs. Surrogate models that represent uncertain inputs internally live under
``robotorchan.models.uncertain``.
"""

from robotorchan.uncertainty.scenarios import (
    CorrelatedGaussianPerturbation,
    EmpiricalScenarios,
    GaussianPerturbation,
    UniformPerturbation,
)

__all__ = [
    "CorrelatedGaussianPerturbation",
    "EmpiricalScenarios",
    "GaussianPerturbation",
    "UniformPerturbation",
]
