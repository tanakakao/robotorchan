"""Heterogeneous multitask GP wrapper with robotorchan conventions."""

from __future__ import annotations

from botorch.models.heterogeneous_mtgp import HeterogeneousMTGP as BoTorchHeterogeneousMTGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from robotorchan.models.base import ModelTrainingMixin, RawDataMixin


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
        raw_Yvars = (
            None
            if train_Yvars is None
            else [Yvar.detach().clone() for Yvar in train_Yvars]
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
            self._get_raw_tensor(f"train_X_{index}")
            for index in range(self._raw_task_count)
        )  # type: ignore[return-value]

    @property
    def raw_train_Ys(self) -> tuple[Tensor, ...]:
        return tuple(
            self._get_raw_tensor(f"train_Y_{index}")
            for index in range(self._raw_task_count)
        )  # type: ignore[return-value]

    @property
    def raw_train_Yvars(self) -> tuple[Tensor, ...] | None:
        if not any(name.startswith("train_Yvar_") for name in self._raw_data_names):
            return None
        return tuple(
            self._get_raw_tensor(f"train_Yvar_{index}")
            for index in range(self._raw_task_count)
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
