"""Heterogeneous multitask GP wrappers with robotorchan conventions."""

from __future__ import annotations

import torch
from botorch.models.heterogeneous_mtgp import HeterogeneousMTGP as BoTorchHeterogeneousMTGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.kernels.heterogeneous_multitask import (
    LOG_OUTPUTSCALE_CONSTRAINT,
    MultiTaskConditionalKernel,
)
from botorch.models.map_saas import add_saas_prior
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.kernels import AdditiveKernel, MaternKernel, ProductKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.nn import ModuleList

from robotorchan.models.base import ModelTrainingMixin, RawDataMixin, normalize_feature_dims


class HeterogeneousMTGP(RawDataMixin, ModelTrainingMixin, BoTorchHeterogeneousMTGP):
    """BoTorch heterogeneous MTGP with grouped raw-data retention."""

    supports_mll = True

    def __init__(
        self,
        train_Xs: list[Tensor],
        train_Ys: list[Tensor],
        train_Yvars: list[Tensor] | None,
        feature_indices: list[list[int]],
        full_feature_dim: int,
        rank: int | None = None,
        use_saas_prior: bool = True,
        use_combinatorial_kernel: bool = True,
        all_tasks: list[int] | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | None = None,
        validate_task_values: bool = True,
    ) -> None:
        raw_Xs = [X.detach().clone() for X in train_Xs]
        raw_Ys = [Y.detach().clone() for Y in train_Ys]
        raw_Yvars = None if train_Yvars is None else [Yvar.detach().clone() for Yvar in train_Yvars]

        super().__init__(
            train_Xs=train_Xs,
            train_Ys=train_Ys,
            train_Yvars=train_Yvars,
            feature_indices=feature_indices,
            full_feature_dim=full_feature_dim,
            rank=rank,
            use_saas_prior=use_saas_prior,
            use_combinatorial_kernel=use_combinatorial_kernel,
            all_tasks=all_tasks,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            validate_task_values=validate_task_values,
        )

        self._raw_task_count = len(raw_Xs)
        for index, tensor in enumerate(raw_Xs):
            self._store_raw_tensor(f"train_X_{index}", tensor)
        for index, tensor in enumerate(raw_Ys):
            self._store_raw_tensor(f"train_Y_{index}", tensor)
        if raw_Yvars is not None:
            for index, tensor in enumerate(raw_Yvars):
                self._store_raw_tensor(f"train_Yvar_{index}", tensor)

    @property
    def raw_train_Xs(self) -> tuple[Tensor, ...]:
        return tuple(
            self._get_raw_tensor(f"train_X_{index}") for index in range(self._raw_task_count)
        )  # type: ignore[return-value]

    @property
    def raw_train_Ys(self) -> tuple[Tensor, ...]:
        return tuple(
            self._get_raw_tensor(f"train_Y_{index}") for index in range(self._raw_task_count)
        )  # type: ignore[return-value]

    @property
    def raw_train_Yvars(self) -> tuple[Tensor, ...] | None:
        if not any(name.startswith("train_Yvar_") for name in self._raw_data_names):
            return None
        return tuple(
            self._get_raw_tensor(f"train_Yvar_{index}") for index in range(self._raw_task_count)
        )  # type: ignore[return-value]

    @property
    def raw_data(self) -> dict[str, tuple[Tensor, ...] | None]:
        return {
            "train_Xs": self.raw_train_Xs,
            "train_Ys": self.raw_train_Ys,
            "train_Yvars": self.raw_train_Yvars,
        }

    def make_mll(self) -> ExactMarginalLogLikelihood:
        return ExactMarginalLogLikelihood(self.likelihood, self)


class _MixedMultiTaskConditionalKernel(MultiTaskConditionalKernel):
    """Conditional heterogeneous kernel with native categorical subset kernels."""

    def __init__(
        self,
        feature_indices: list[list[int]],
        cat_dims: tuple[int, ...],
        use_saas_prior: bool,
        use_combinatorial_kernel: bool,
    ) -> None:
        super().__init__(
            feature_indices=feature_indices,
            use_saas_prior=use_saas_prior,
            use_combinatorial_kernel=use_combinatorial_kernel,
        )
        self.cat_dims = cat_dims
        self.kernels = self._construct_mixed_individual_kernels()

    def _construct_continuous_kernel(self, active_dims: list[int]) -> MaternKernel:
        kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=len(active_dims),
            active_dims=active_dims,
        )
        if self.use_saas_prior:
            add_saas_prior(kernel)
        return kernel

    def _scale(self, kernel: torch.nn.Module) -> ScaleKernel:
        return ScaleKernel(
            kernel,
            outputscale_constraint=LOG_OUTPUTSCALE_CONSTRAINT,
        )

    def _construct_mixed_individual_kernels(self) -> ModuleList:
        kernels = ModuleList()
        for active_indices_tuple in self.active_index_map:
            active_dims = list(active_indices_tuple)
            cat_dims = [dim for dim in active_dims if dim in self.cat_dims]
            cont_dims = [dim for dim in active_dims if dim not in self.cat_dims]

            if not cat_dims:
                kernels.append(self._scale(self._construct_continuous_kernel(cont_dims)))
                continue

            if not cont_dims:
                kernels.append(
                    self._scale(
                        CategoricalKernel(
                            ard_num_dims=len(cat_dims),
                            active_dims=cat_dims,
                        )
                    )
                )
                continue

            continuous_main = self._construct_continuous_kernel(cont_dims)
            continuous_interaction = self._construct_continuous_kernel(cont_dims)
            categorical_main = CategoricalKernel(
                ard_num_dims=len(cat_dims),
                active_dims=cat_dims,
            )
            categorical_interaction = CategoricalKernel(
                ard_num_dims=len(cat_dims),
                active_dims=cat_dims,
            )
            kernels.append(
                AdditiveKernel(
                    self._scale(continuous_main),
                    self._scale(categorical_main),
                    self._scale(
                        ProductKernel(
                            continuous_interaction,
                            categorical_interaction,
                        )
                    ),
                )
            )
        return kernels


class MixedHeterogeneousMTGP(HeterogeneousMTGP):
    """Heterogeneous MTGP with native categorical kernels in feature subsets.

    ``cat_dims`` uses the global feature numbering defined by ``feature_indices``
    and ``full_feature_dim``. The task feature appended internally by BoTorch is
    structural and is never part of ``cat_dims``.

    BoTorch's heterogeneous model owns a ``MultiTaskConditionalKernel`` that
    conditionally activates kernels for feature subsets shared by different
    tasks. This wrapper preserves that structure and replaces only those subset
    kernels that contain categorical dimensions with native categorical/mixed
    covariance. Continuous-only subsets keep the upstream Matern + optional
    SAAS-prior construction.
    """

    def __init__(
        self,
        train_Xs: list[Tensor],
        train_Ys: list[Tensor],
        train_Yvars: list[Tensor] | None,
        feature_indices: list[list[int]],
        full_feature_dim: int,
        cat_dims: list[int],
        rank: int | None = None,
        use_saas_prior: bool = True,
        use_combinatorial_kernel: bool = True,
        all_tasks: list[int] | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | None = None,
        validate_task_values: bool = True,
    ) -> None:
        normalized_cat_dims = tuple(
            normalize_feature_dims(cat_dims, input_dim=full_feature_dim)
        )
        super().__init__(
            train_Xs=train_Xs,
            train_Ys=train_Ys,
            train_Yvars=train_Yvars,
            feature_indices=feature_indices,
            full_feature_dim=full_feature_dim,
            rank=rank,
            use_saas_prior=use_saas_prior,
            use_combinatorial_kernel=use_combinatorial_kernel,
            all_tasks=all_tasks,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            validate_task_values=validate_task_values,
        )

        mixed_covar_module = _MixedMultiTaskConditionalKernel(
            feature_indices=feature_indices,
            cat_dims=normalized_cat_dims,
            use_saas_prior=use_saas_prior,
            use_combinatorial_kernel=use_combinatorial_kernel,
        )
        mixed_covar_module.active_dims = torch.arange(
            full_feature_dim + 1,
            device=train_Xs[0].device,
        )
        self.covar_module = mixed_covar_module.to(train_Xs[0])
        self.cat_dims = normalized_cat_dims
