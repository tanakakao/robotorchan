"""Variational GP wrapper with robotorchan model conventions."""

from __future__ import annotations

from botorch.models import SingleTaskVariationalGP as BoTorchSingleTaskVariationalGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import (
    VariationalStrategy,
    _VariationalDistribution,
    _VariationalStrategy,
)
from torch import Tensor

from robotorchan.models.base import ModelTrainingMixin, RawDataMixin


class SingleTaskVariationalGP(
    RawDataMixin,
    ModelTrainingMixin,
    BoTorchSingleTaskVariationalGP,
):
    """BoTorch ``SingleTaskVariationalGP`` with robotorchan conveniences.

    Predictive behavior, inducing-point allocation, variational strategy, and
    transforms are delegated to BoTorch. robotorchan adds raw-data retention
    and a ``VariationalELBO`` factory.
    """

    supports_mll = True

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor | None = None,
        likelihood: Likelihood | None = None,
        num_outputs: int = 1,
        learn_inducing_points: bool = True,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize the variational GP while retaining caller raw tensors."""
        raw_train_X = train_X.detach().clone()
        raw_train_Y = None if train_Y is None else train_Y.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            num_outputs=num_outputs,
            learn_inducing_points=learn_inducing_points,
            covar_module=covar_module,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=inducing_point_allocator,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._store_raw_tensor("train_X", raw_train_X)
        self._store_raw_tensor("train_Y", raw_train_Y)
        self._store_raw_tensor("train_Yvar", None)

    @property
    def raw_train_X(self) -> Tensor:
        """Caller-supplied training inputs before input transforms."""
        value = self._get_raw_tensor("train_X")
        if value is None:  # pragma: no cover - guarded by constructor contract
            raise RuntimeError("raw_train_X was unexpectedly stored as None.")
        return value

    @property
    def raw_train_Y(self) -> Tensor | None:
        """Caller-supplied training outcomes, if provided."""
        return self._get_raw_tensor("train_Y")

    @property
    def raw_train_Yvar(self) -> None:
        """Return ``None`` because the upstream constructor has no train_Yvar."""
        return None

    def make_mll(self, num_data: int | None = None) -> VariationalELBO:
        """Construct the variational evidence lower bound.

        Args:
            num_data: Total number of observations represented by the ELBO. If
                omitted, use the number of rows in ``raw_train_X``. Supply this
                explicitly when ``train_X`` is only a minibatch or subset of a
                larger training data set.

        Raises:
            ValueError: If ``num_data`` is not positive.
        """
        if num_data is None:
            num_data = self.raw_train_X.shape[-2]
        if num_data <= 0:
            raise ValueError("num_data must be positive.")
        return VariationalELBO(
            likelihood=self.likelihood,
            model=self.model,
            num_data=num_data,
        )
