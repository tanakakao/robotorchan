"""Multi-fidelity GP wrapper with robotorchan model conventions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.models import SingleTaskMultiFidelityGP as BoTorchSingleTaskMultiFidelityGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import Likelihood
from gpytorch.module import Module
from torch import Tensor

from robotorchan.models.base import (
    ContinuousKernelFactory,
    ExactGPModelMixin,
    make_mixed_covar_module,
    normalize_feature_dims,
)


class SingleTaskMultiFidelityGP(ExactGPModelMixin, BoTorchSingleTaskMultiFidelityGP):
    """BoTorch ``SingleTaskMultiFidelityGP`` with robotorchan conveniences."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        linear_truncated: bool = True,
        nu: float = 2.5,
        covar_module: Module | None = None,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize the multi-fidelity GP while retaining raw training tensors."""
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            iteration_fidelity=iteration_fidelity,
            data_fidelities=data_fidelities,
            linear_truncated=linear_truncated,
            nu=nu,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=raw_train_Yvar,
        )


class MixedSingleTaskMultiFidelityGP(SingleTaskMultiFidelityGP):
    """Multi-fidelity GP with native categorical design covariance.

    Fidelity columns remain structural and are modeled by BoTorch's native
    fidelity kernels. The mixed covariance is restricted to non-fidelity
    design columns. The linear-truncated path is intentionally unsupported
    because it owns the complete input covariance and cannot preserve a
    separate mixed design covariance.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        linear_truncated: bool = False,
        nu: float = 2.5,
        cont_kernel_factory: ContinuousKernelFactory | None = None,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize a mixed-input multi-fidelity GP."""
        if linear_truncated:
            raise ValueError(
                "MixedSingleTaskMultiFidelityGP requires linear_truncated=False "
                "because the linear-truncated fidelity kernel owns the full "
                "input covariance."
            )

        input_dim = train_X.shape[-1]
        fidelity_dims: list[int] = []
        if iteration_fidelity is not None:
            fidelity_dims.extend(
                normalize_feature_dims(
                    [iteration_fidelity], input_dim, name="iteration_fidelity"
                )
            )
        if data_fidelities is not None:
            fidelity_dims.extend(
                normalize_feature_dims(
                    data_fidelities, input_dim, name="data_fidelities"
                )
            )
        if len(set(fidelity_dims)) != len(fidelity_dims):
            raise ValueError("Fidelity dimensions must not contain duplicates.")

        normalized_cat_dims = normalize_feature_dims(
            cat_dims,
            input_dim,
            name="cat_dims",
            excluded_dims=fidelity_dims,
        )
        batch_shape = train_X.shape[:-2]
        if train_Y.shape[-1] > 1:
            batch_shape = torch.Size((*batch_shape, train_Y.shape[-1]))
        data_covar_module = make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=normalized_cat_dims,
            excluded_dims=fidelity_dims,
            batch_shape=batch_shape,
            cont_kernel_factory=cont_kernel_factory,
        )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            iteration_fidelity=iteration_fidelity,
            data_fidelities=data_fidelities,
            linear_truncated=False,
            nu=nu,
            covar_module=data_covar_module,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.cat_dims = tuple(normalized_cat_dims)
        self.fidelity_dims = tuple(sorted(fidelity_dims))
