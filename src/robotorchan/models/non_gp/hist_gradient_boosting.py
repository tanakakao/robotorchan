"""Histogram gradient-boosting surrogate for Bayesian optimization."""

from __future__ import annotations

from typing import Any

from sklearn.multioutput import MultiOutputRegressor

from torch import Tensor

from robotorchan.models.non_gp.bootstrap import BootstrapEnsembleSurrogate

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:  # pragma: no cover
    HistGradientBoostingRegressor = None


class HistGradientBoostingSurrogate(BootstrapEnsembleSurrogate):
    """Histogram gradient boosting with bootstrap model uncertainty."""

    model_name = "HistGradientBoostingSurrogate"

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        n_members: int = 16,
        random_state: int | None = None,
        **boosting_kwargs: Any,
    ) -> None:
        if HistGradientBoostingRegressor is None:
            raise ImportError(
                "HistGradientBoostingSurrogate requires scikit-learn. "
                "Install the robotorchan tree extra."
            )
        super().__init__(
            train_X,
            train_Y,
            n_members=n_members,
            random_state=random_state,
            **boosting_kwargs,
        )
        self.boosting_kwargs = self.estimator_kwargs

    def _make_estimator(self, member_index: int) -> Any:
        estimator = HistGradientBoostingRegressor(
            random_state=None if self.random_state is None else self.random_state + member_index,
            **self.estimator_kwargs,
        )
        if self.num_outputs == 1:
            return estimator
        return MultiOutputRegressor(estimator)
