"""Mixed-input multi-fidelity GP models."""

from __future__ import annotations

from collections.abc import Sequence

from botorch.models import MixedSingleTaskGP as BoTorchMixedSingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from torch import Tensor

from robotorchan.models.base import ExactGPModelMixin


class MixedSingleTaskMultiFidelityGP(ExactGPModelMixin, BoTorchMixedSingleTaskGP):
    """Mixed-space GP with explicit fidelity dimensions.

    This model keeps categorical design variables under the BoTorch mixed kernel
    while treating fidelity variables as ordinary ordered continuous inputs.
    It is intended for discrete or continuous fidelity levels optimized with
    BoTorch multi-fidelity acquisition utilities. Unlike
    :class:`SingleTaskMultiFidelityGP`, it does not apply the specialized
    linear-truncated fidelity kernel.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        *,
        fidelity_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        cont_kernel_factory: callable | None = None,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        input_dim = train_X.shape[-1]
        normalized_cat_dims = self._validate_dims(cat_dims, input_dim, "cat_dims")
        normalized_fidelity_dims = self._validate_dims(
            fidelity_dims, input_dim, "fidelity_dims"
        )
        overlap = set(normalized_cat_dims).intersection(normalized_fidelity_dims)
        if overlap:
            raise ValueError(
                "Categorical and fidelity dimensions must be disjoint; "
                f"overlap={sorted(overlap)}."
            )

        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        kwargs: dict[str, object] = {}
        if cont_kernel_factory is not None:
            kwargs["cont_kernel_factory"] = cont_kernel_factory

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=normalized_cat_dims,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            **kwargs,
        )
        self._fidelity_dims = normalized_fidelity_dims
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=raw_train_Yvar,
        )

    @staticmethod
    def _validate_dims(dims: Sequence[int], input_dim: int, name: str) -> list[int]:
        normalized = [int(dim) for dim in dims]
        if not normalized:
            raise ValueError(f"{name} must contain at least one dimension.")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name} must not contain duplicate dimensions.")
        if any(dim < 0 or dim >= input_dim for dim in normalized):
            raise ValueError(f"{name} must be within [0, {input_dim}).")
        return normalized

    @property
    def fidelity_dims(self) -> list[int]:
        """Return original-space fidelity dimensions."""
        return list(self._fidelity_dims)

    @property
    def cat_dims(self) -> list[int]:
        """Return original-space categorical dimensions."""
        return list(self._ignore_X_dims_scaling_check)
