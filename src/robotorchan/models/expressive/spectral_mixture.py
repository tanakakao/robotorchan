"""Spectral-mixture Gaussian process models."""

from __future__ import annotations

from typing import Literal

import torch
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.kernels import ScaleKernel, SpectralMixtureKernel
from gpytorch.likelihoods import Likelihood, MultitaskGaussianLikelihood
from torch import Tensor

from robotorchan.models.base import make_mixed_covar_module, normalize_feature_dims
from robotorchan.models.standard.multitask import KroneckerMultiTaskGP, MultiTaskGP
from robotorchan.models.standard.single_task import SingleTaskGP


class SpectralMixtureGP(SingleTaskGP):
    """Exact GP with a spectral-mixture covariance kernel.

    The spectral-mixture kernel represents stationary covariance through a
    Gaussian-mixture approximation to its spectral density. It is useful for
    periodic, quasi-periodic, and multi-scale stationary structure.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        num_mixtures: int = 4,
        initialization: Literal["data", "empspect"] = "data",
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize the spectral-mixture GP and its spectral parameters."""
        if num_mixtures <= 0:
            raise ValueError("num_mixtures must be positive.")
        if initialization not in {"data", "empspect"}:
            raise ValueError("initialization must be 'data' or 'empspect'.")
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape n x d.")
        if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
            raise ValueError("train_Y must have shape n x 1.")

        spectral_kernel = SpectralMixtureKernel(
            num_mixtures=num_mixtures,
            ard_num_dims=train_X.shape[-1],
        ).to(train_X)
        if initialization == "data":
            spectral_kernel.initialize_from_data(train_X, train_Y.squeeze(-1))
        else:
            spectral_kernel.initialize_from_data_empspect(train_X, train_Y.squeeze(-1))

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=ScaleKernel(spectral_kernel).to(train_X),
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.num_mixtures = num_mixtures
        self.initialization = initialization


class SpectralMixtureMultiTaskGP(MultiTaskGP):
    """Long-format multi-task GP with a spectral-mixture data kernel."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        train_Yvar: Tensor | None = None,
        *,
        num_mixtures: int = 4,
        initialization: Literal["data", "empspect"] = "data",
        rank: int | None = None,
    ) -> None:
        if num_mixtures <= 0:
            raise ValueError("num_mixtures must be positive.")
        if initialization not in {"data", "empspect"}:
            raise ValueError("initialization must be 'data' or 'empspect'.")
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape n x d.")
        if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
            raise ValueError("train_Y must have shape n x 1.")

        input_dim = train_X.shape[-1]
        resolved_task_feature = normalize_feature_dims(
            [task_feature], input_dim, name="task_feature"
        )[0]
        data_dims = [i for i in range(input_dim) if i != resolved_task_feature]
        data_X = train_X[..., data_dims]
        spectral_kernel = SpectralMixtureKernel(
            num_mixtures=num_mixtures,
            ard_num_dims=len(data_dims),
            active_dims=data_dims,
        ).to(train_X)
        if initialization == "data":
            spectral_kernel.initialize_from_data(data_X, train_Y.squeeze(-1))
        else:
            spectral_kernel.initialize_from_data_empspect(data_X, train_Y.squeeze(-1))

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            train_Yvar=train_Yvar,
            covar_module=ScaleKernel(spectral_kernel).to(train_X),
            rank=rank,
        )
        self.num_mixtures = num_mixtures
        self.initialization = initialization


class SpectralMixtureKroneckerMultiTaskGP(KroneckerMultiTaskGP):
    """Block-design Kronecker multi-task GP with spectral data covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_mixtures: int = 4,
        initialization: Literal["data", "empspect"] = "data",
        likelihood: MultitaskGaussianLikelihood | None = None,
        rank: int | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        if num_mixtures <= 0:
            raise ValueError("num_mixtures must be positive.")
        if initialization not in {"data", "empspect"}:
            raise ValueError("initialization must be 'data' or 'empspect'.")
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape n x d.")
        if train_Y.ndim != 2 or train_Y.shape[0] != train_X.shape[0]:
            raise ValueError("train_Y must have shape n x m.")

        spectral_kernel = SpectralMixtureKernel(
            num_mixtures=num_mixtures,
            ard_num_dims=train_X.shape[-1],
        ).to(train_X)
        initialization_target = train_Y.mean(dim=-1)
        if initialization == "data":
            spectral_kernel.initialize_from_data(train_X, initialization_target)
        else:
            spectral_kernel.initialize_from_data_empspect(train_X, initialization_target)

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            data_covar_module=ScaleKernel(spectral_kernel).to(train_X),
            rank=rank,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.num_mixtures = num_mixtures
        self.initialization = initialization


class MixedSpectralMixtureGP(SingleTaskGP):
    """Mixed-input exact GP with a spectral-mixture continuous kernel."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        *,
        num_mixtures: int = 4,
        initialization: Literal["data", "empspect"] = "data",
    ) -> None:
        if num_mixtures <= 0:
            raise ValueError("num_mixtures must be positive.")
        if initialization not in {"data", "empspect"}:
            raise ValueError("initialization must be 'data' or 'empspect'.")
        input_dim = train_X.shape[-1]
        normalized_cat_dims = normalize_feature_dims(cat_dims, input_dim, name="cat_dims")
        continuous_dims = [dim for dim in range(input_dim) if dim not in normalized_cat_dims]
        if not continuous_dims:
            raise ValueError("MixedSpectralMixtureGP requires at least one continuous feature.")
        data_X = train_X[..., continuous_dims]

        def continuous_kernel_factory(batch_shape, num_dims, active_dims):
            kernel = SpectralMixtureKernel(
                num_mixtures=num_mixtures,
                ard_num_dims=num_dims,
                active_dims=active_dims,
                batch_shape=batch_shape,
            ).to(train_X)
            if initialization == "data":
                kernel.initialize_from_data(data_X, train_Y.squeeze(-1))
            else:
                kernel.initialize_from_data_empspect(data_X, train_Y.squeeze(-1))
            return ScaleKernel(kernel, batch_shape=batch_shape).to(train_X)

        covar_module = make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=normalized_cat_dims,
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=continuous_kernel_factory,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
        )
        self.cat_dims = tuple(normalized_cat_dims)
        self.num_mixtures = num_mixtures
        self.initialization = initialization


class MixedSpectralMixtureMultiTaskGP(MultiTaskGP):
    """Mixed long-format multi-task GP with a spectral-mixture data kernel."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        *,
        num_mixtures: int = 4,
        initialization: Literal["data", "empspect"] = "data",
        rank: int | None = None,
    ) -> None:
        if num_mixtures <= 0:
            raise ValueError("num_mixtures must be positive.")
        if initialization not in {"data", "empspect"}:
            raise ValueError("initialization must be 'data' or 'empspect'.")
        input_dim = train_X.shape[-1]
        task_dim = normalize_feature_dims([task_feature], input_dim, name="task_feature")[0]
        cats = normalize_feature_dims(
            cat_dims,
            input_dim,
            name="cat_dims",
            excluded_dims=[task_dim],
        )
        continuous_dims = [dim for dim in range(input_dim) if dim != task_dim and dim not in cats]
        if not continuous_dims:
            raise ValueError(
                "MixedSpectralMixtureMultiTaskGP requires at least one continuous feature."
            )
        data_X = train_X[..., continuous_dims]

        def continuous_kernel_factory(batch_shape, num_dims, active_dims):
            kernel = SpectralMixtureKernel(
                num_mixtures=num_mixtures,
                ard_num_dims=num_dims,
                active_dims=active_dims,
                batch_shape=batch_shape,
            ).to(train_X)
            if initialization == "data":
                kernel.initialize_from_data(data_X, train_Y.squeeze(-1))
            else:
                kernel.initialize_from_data_empspect(data_X, train_Y.squeeze(-1))
            return ScaleKernel(kernel, batch_shape=batch_shape).to(train_X)

        covar_module = make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=cats,
            excluded_dims=[task_dim],
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=continuous_kernel_factory,
        )
        covar_module.active_dims = torch.arange(input_dim, device=train_X.device)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
            rank=rank,
        )
        self.cat_dims = tuple(cats)
        self.num_mixtures = num_mixtures
        self.initialization = initialization
