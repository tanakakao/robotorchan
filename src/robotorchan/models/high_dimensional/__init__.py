"""High-dimensional surrogate models and dimension-reduction extensions."""

from robotorchan.models.high_dimensional.alebo import ALEBOGP
from robotorchan.models.high_dimensional.fully_bayesian import (
    MixedSaasFullyBayesianMultiTaskGP,
    MixedSaasFullyBayesianSingleTaskGP,
    SaasFullyBayesianMultiTaskGP,
    SaasFullyBayesianSingleTaskGP,
)
from robotorchan.models.high_dimensional.map_saas import (
    AdditiveMapSaasSingleTaskGP,
    EnsembleMapSaasSingleTaskGP,
    MixedAdditiveMapSaasSingleTaskGP,
    MixedEnsembleMapSaasSingleTaskGP,
)
from robotorchan.models.high_dimensional.reduced import *  # noqa: F403
from robotorchan.models.high_dimensional.reduced import __all__ as _reduced_all

__all__ = [
    "ALEBOGP",
    "AdditiveMapSaasSingleTaskGP",
    "EnsembleMapSaasSingleTaskGP",
    "MixedAdditiveMapSaasSingleTaskGP",
    "MixedEnsembleMapSaasSingleTaskGP",
    "MixedSaasFullyBayesianMultiTaskGP",
    "MixedSaasFullyBayesianSingleTaskGP",
    "SaasFullyBayesianMultiTaskGP",
    "SaasFullyBayesianSingleTaskGP",
    *_reduced_all,
]
