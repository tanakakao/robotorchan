"""Model-list GP wrapper with robotorchan model conventions."""

from __future__ import annotations

from typing import ClassVar

from botorch.models import ModelListGP as BoTorchModelListGP
from gpytorch.mlls import SumMarginalLogLikelihood
from torch import Tensor

from robotorchan.models.base import ModelTrainingMixin


class ModelListGP(ModelTrainingMixin, BoTorchModelListGP):
    """BoTorch ``ModelListGP`` with robotorchan convenience features.

    ``ModelListGP`` is a container of independent GP models rather than a
    single supervised GP with one training tensor pair. robotorchan therefore
    preserves raw training data per child model instead of inventing singular
    ``raw_train_X`` / ``raw_train_Y`` attributes at the container level.
    """

    supports_mll: ClassVar[bool] = True

    @property
    def raw_train_Xs(self) -> tuple[Tensor | None, ...]:
        """Return each child model's retained raw training inputs, if available."""
        return tuple(getattr(model, "raw_train_X", None) for model in self.models)

    @property
    def raw_train_Ys(self) -> tuple[Tensor | None, ...]:
        """Return each child model's retained raw training outcomes, if available."""
        return tuple(getattr(model, "raw_train_Y", None) for model in self.models)

    @property
    def raw_train_Yvars(self) -> tuple[Tensor | None, ...]:
        """Return each child model's retained observation variances, if available."""
        return tuple(getattr(model, "raw_train_Yvar", None) for model in self.models)

    def make_mll(self) -> SumMarginalLogLikelihood:
        """Construct the summed marginal log likelihood for the child models."""
        return SumMarginalLogLikelihood(self.likelihood, self)
