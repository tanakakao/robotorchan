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

from robotorchan.models.base import (
    ExactGPModelMixin,
    _CategoricalOneHotInputTransform,
)


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


class MixedAdditiveMapSaasSingleTaskGP(AdditiveMapSaasSingleTaskGP):
    """Additive MAP-SAAS GP for mixed inputs via internal one-hot encoding.

    MAP-SAAS constructs a specialized additive SAAS covariance internally and
    does not expose a safe native mixed-kernel injection point. The public API
    therefore remains in the original mixed space while a private BoTorch input
    transform one-hot encodes categorical columns consistently for fitting,
    posterior evaluation, conditioning, and fantasization.

    This one-hot implementation is a compatibility fallback, not the preferred
    long-term categorical representation. If a mathematically faithful native
    categorical-kernel path becomes available without changing MAP-SAAS
    semantics, this Mixed wrapper should migrate to that path while preserving
    its public ``cat_dims`` API.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        num_taus: int = 4,
    ) -> None:
        if input_transform is not None:
            raise ValueError(
                "MixedAdditiveMapSaasSingleTaskGP manages input_transform internally "
                "for categorical encoding; a custom input_transform is not currently "
                "supported."
            )

        mixed_transform = _CategoricalOneHotInputTransform(
            train_X=train_X,
            cat_dims=cat_dims,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            outcome_transform=outcome_transform,
            input_transform=mixed_transform,
            num_taus=num_taus,
        )
        self.cat_dims = mixed_transform.cat_dims
        self.raw_input_dim = mixed_transform.raw_input_dim
        self.encoded_input_dim = mixed_transform.encoded_input_dim

    @property
    def category_values(self) -> tuple[Tensor, ...]:
        """Observed category values in normalized ``cat_dims`` order."""
        return self.input_transform.category_values


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


class MixedEnsembleMapSaasSingleTaskGP(EnsembleMapSaasSingleTaskGP):
    """Ensemble MAP-SAAS GP for mixed inputs via internal one-hot encoding.

    The same internal input-transform fallback used by
    ``MixedAdditiveMapSaasSingleTaskGP`` keeps the public feature space mixed and
    raw while the MAP-SAAS ensemble operates on encoded numeric features.
    Native categorical covariance remains the preferred future implementation
    whenever the upstream architecture can support it without semantic changes.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        num_taus: int = 4,
        taus: Tensor | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        if input_transform is not None:
            raise ValueError(
                "MixedEnsembleMapSaasSingleTaskGP manages input_transform internally "
                "for categorical encoding; a custom input_transform is not currently "
                "supported."
            )

        mixed_transform = _CategoricalOneHotInputTransform(
            train_X=train_X,
            cat_dims=cat_dims,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            num_taus=num_taus,
            taus=taus,
            outcome_transform=outcome_transform,
            input_transform=mixed_transform,
        )
        self.cat_dims = mixed_transform.cat_dims
        self.raw_input_dim = mixed_transform.raw_input_dim
        self.encoded_input_dim = mixed_transform.encoded_input_dim

    @property
    def category_values(self) -> tuple[Tensor, ...]:
        """Observed category values in normalized ``cat_dims`` order."""
        return self.input_transform.category_values
