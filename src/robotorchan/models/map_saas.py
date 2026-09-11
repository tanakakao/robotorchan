"""MAP-SAAS wrappers with robotorchan model conventions."""

from __future__ import annotations

from botorch.models.map_saas import (
    AdditiveMapSaasSingleTaskGP as BoTorchAdditiveMapSaasSingleTaskGP,
)
from botorch.models.map_saas import (
    EnsembleMapSaasSingleTaskGP as BoTorchEnsembleMapSaasSingleTaskGP,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from torch import Tensor

from robotorchan.models.base import ExactGPModelMixin


class AdditiveMapSaasSingleTaskGP(
    ExactGPModelMixin,
    BoTorchAdditiveMapSaasSingleTaskGP,
):
    """BoTorch additive MAP-SAAS GP with robotorchan raw-data retention."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        num_taus: int = 4,
    ) -> None:
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            num_taus=num_taus,
        )
        self._store_supervised_training_data(raw_train_X, raw_train_Y, raw_train_Yvar)


class EnsembleMapSaasSingleTaskGP(
    ExactGPModelMixin,
    BoTorchEnsembleMapSaasSingleTaskGP,
):
    """BoTorch ensemble MAP-SAAS GP with robotorchan raw-data retention."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        num_taus: int = 4,
        taus: Tensor | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            num_taus=num_taus,
            taus=taus,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._store_supervised_training_data(raw_train_X, raw_train_Y, raw_train_Yvar)
