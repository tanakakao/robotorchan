"""Spectral-mixture Gaussian process models."""

from __future__ import annotations

from typing import Literal

from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.kernels import ScaleKernel, SpectralMixtureKernel
from gpytorch.likelihoods import Likelihood
from torch import Tensor

from robotorchan.models.base import normalize_feature_dims
from robotorchan.models.multitask import MultiTaskGP
from robotorchan.models.single_task import SingleTaskGP


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
