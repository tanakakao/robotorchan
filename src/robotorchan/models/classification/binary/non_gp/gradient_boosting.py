"""Bootstrap gradient-boosting binary classifier."""

from __future__ import annotations

from typing import Any

from robotorchan.models.classification.binary.non_gp.bootstrap import (
    BootstrapBinaryClassificationEnsemble,
)

try:
    from sklearn.ensemble import GradientBoostingClassifier
except ImportError:  # pragma: no cover
    GradientBoostingClassifier = None


class BootstrapGradientBoostingBinaryClassifier(BootstrapBinaryClassificationEnsemble):
    """Gradient boosting with empirical bootstrap model uncertainty."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if GradientBoostingClassifier is None:
            raise ImportError("BootstrapGradientBoostingBinaryClassifier requires scikit-learn.")
        super().__init__(*args, **kwargs)

    def _make_estimator(self, member_index: int) -> Any:
        kwargs = dict(self.estimator_kwargs)
        if "random_state" not in kwargs and self.random_state is not None:
            kwargs["random_state"] = self.random_state + member_index
        return GradientBoostingClassifier(**kwargs)
