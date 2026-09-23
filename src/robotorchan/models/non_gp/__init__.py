"""Non-GP probabilistic surrogate models."""

from robotorchan.models.non_gp.extra_trees import ExtraTreesSurrogate
from robotorchan.models.non_gp.gradient_boosting import GradientBoostingSurrogate
from robotorchan.models.non_gp.hist_gradient_boosting import HistGradientBoostingSurrogate
from robotorchan.models.non_gp.ngboost import NGBoostSurrogate
from robotorchan.models.non_gp.posterior import make_ensemble_posterior
from robotorchan.models.non_gp.random_forest import RandomForestSurrogate

__all__ = [
    "ExtraTreesSurrogate",
    "GradientBoostingSurrogate",
    "HistGradientBoostingSurrogate",
    "NGBoostSurrogate",
    "RandomForestSurrogate",
    "make_ensemble_posterior",
]
