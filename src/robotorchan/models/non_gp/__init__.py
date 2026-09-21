"""Non-GP probabilistic surrogate models."""

from robotorchan.models.non_gp.posterior import make_ensemble_posterior
from robotorchan.models.non_gp.random_forest import RandomForestSurrogate

__all__ = ["RandomForestSurrogate", "make_ensemble_posterior"]
