"""Variational GP surrogate with Student-t observation noise."""

from __future__ import annotations

from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.kernels import IndexKernel, MaternKernel, ProductKernel
from gpytorch.likelihoods import StudentTLikelihood
from torch import Tensor, nn

from robotorchan.models.base import RawDataMixin, make_mixed_covar_module, normalize_feature_dims
from robotorchan.models.variational import MixedSingleTaskVariationalGP, SingleTaskVariationalGP


class _StudentTGPBase(RawDataMixin, nn.Module):
    """Shared Student-t variational training and posterior contract."""

    supports_mll = False

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def likelihood(self) -> StudentTLikelihood:
        return self.response_model.likelihood

    @property
    def raw_train_X(self) -> Tensor:
        value = self._get_raw_tensor("train_X")
        if value is None:
            raise RuntimeError("raw_train_X was unexpectedly stored as None.")
        return value

    @property
    def raw_train_Y(self) -> Tensor:
        value = self._get_raw_tensor("train_Y")
        if value is None:
            raise RuntimeError("raw_train_Y was unexpectedly stored as None.")
        return value

    @property
    def raw_train_Yvar(self) -> None:
        return None

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.response_model.model.variational_strategy._clear_cache()
        return result

    def make_mll(self):
        raise RuntimeError("Use training_loss() for Student-t variational inference.")

    def posterior(self, X: Tensor, **kwargs) -> GPyTorchPosterior:
        self.response_model.model.variational_strategy._clear_cache()
        return self.response_model.posterior(X, **kwargs)

    def training_loss(self) -> Tensor:
        X = self.raw_train_X
        Y = self.raw_train_Y.squeeze(-1)
        output = self.response_model.model(X)
        expected_log_prob = self.likelihood.expected_log_prob(Y, output).sum()
        kl = self.response_model.model.variational_strategy.kl_divergence().sum()
        return (-expected_log_prob + kl) / X.shape[-2]


class StudentTSingleTaskGP(_StudentTGPBase):
    """Scalar variational GP with a heavy-tailed Student-t likelihood."""

    supports_mll = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_inducing: int = 32,
        df: float = 4.0,
    ) -> None:
        super().__init__()
        if train_Y.shape[-1] != 1:
            raise ValueError("train_Y must have a single output.")
        if num_inducing < 1:
            raise ValueError("num_inducing must be positive.")
        if df <= 2:
            raise ValueError("df must be greater than 2 for finite observation variance.")

        likelihood = StudentTLikelihood()
        likelihood.initialize(deg_free=df)
        inducing = min(int(num_inducing), train_X.shape[-2])
        self.response_model = SingleTaskVariationalGP(
            train_X,
            train_Y,
            likelihood=likelihood,
            inducing_points=inducing,
        )
        self._store_raw_tensor("train_X", train_X.detach().clone())
        self._store_raw_tensor("train_Y", train_Y.detach().clone())
        self._store_raw_tensor("train_Yvar", None)


class MixedStudentTSingleTaskGP(_StudentTGPBase):
    """Student-t variational GP with native mixed continuous/categorical covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: list[int],
        num_inducing: int = 32,
        df: float = 4.0,
    ) -> None:
        super().__init__()
        if train_Y.shape[-1] != 1:
            raise ValueError("train_Y must have a single output.")
        if num_inducing < 1:
            raise ValueError("num_inducing must be positive.")
        if df <= 2:
            raise ValueError("df must be greater than 2 for finite observation variance.")
        likelihood = StudentTLikelihood()
        likelihood.initialize(deg_free=df)
        inducing = min(int(num_inducing), train_X.shape[-2])
        self.response_model = MixedSingleTaskVariationalGP(
            train_X,
            train_Y,
            cat_dims=cat_dims,
            likelihood=likelihood,
            inducing_points=inducing,
        )
        self.cat_dims = self.response_model.cat_dims
        self._store_raw_tensor("train_X", train_X)
        self._store_raw_tensor("train_Y", train_Y)
        self._store_raw_tensor("train_Yvar", None)


def _multitask_covar_module(
    train_X: Tensor,
    task_feature: int,
    *,
    cat_dims: list[int] | None = None,
):
    """Build an ICM-style covariance for long-format variational models."""
    input_dim = train_X.shape[-1]
    task_dim = normalize_feature_dims([task_feature], input_dim, name="task_feature")[0]
    data_dims = [dim for dim in range(input_dim) if dim != task_dim]
    if cat_dims is None:
        data_covar = MaternKernel(nu=2.5, active_dims=data_dims)
    else:
        normalized_cat_dims = normalize_feature_dims(
            cat_dims, input_dim, name="cat_dims", excluded_dims=[task_dim]
        )
        data_covar = make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=normalized_cat_dims,
            excluded_dims=[task_dim],
            batch_shape=train_X.shape[:-2],
        )
    task_values = train_X[..., task_dim].long()
    num_tasks = int(task_values.max().item()) + 1
    task_covar = IndexKernel(num_tasks=num_tasks, active_dims=[task_dim])
    return ProductKernel(data_covar, task_covar), task_dim


class StudentTMultiTaskGP(_StudentTGPBase):
    """Long-format multi-task variational GP with Student-t observations."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        *,
        num_inducing: int = 32,
        df: float = 4.0,
    ) -> None:
        super().__init__()
        if train_Y.shape[-1] != 1:
            raise ValueError("train_Y must have a single output.")
        if num_inducing < 1:
            raise ValueError("num_inducing must be positive.")
        if df <= 2:
            raise ValueError("df must be greater than 2 for finite observation variance.")
        covar_module, task_dim = _multitask_covar_module(train_X, task_feature)
        likelihood = StudentTLikelihood()
        likelihood.initialize(deg_free=df)
        self.response_model = SingleTaskVariationalGP(
            train_X,
            train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            inducing_points=min(int(num_inducing), train_X.shape[-2]),
        )
        self.task_feature = task_dim
        self._store_raw_tensor("train_X", train_X.detach().clone())
        self._store_raw_tensor("train_Y", train_Y.detach().clone())
        self._store_raw_tensor("train_Yvar", None)


class MixedStudentTMultiTaskGP(StudentTMultiTaskGP):
    """Student-t multi-task GP with mixed data covariance and task covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        *,
        cat_dims: list[int],
        num_inducing: int = 32,
        df: float = 4.0,
    ) -> None:
        _StudentTGPBase.__init__(self)
        if train_Y.shape[-1] != 1:
            raise ValueError("train_Y must have a single output.")
        if num_inducing < 1:
            raise ValueError("num_inducing must be positive.")
        if df <= 2:
            raise ValueError("df must be greater than 2 for finite observation variance.")
        covar_module, task_dim = _multitask_covar_module(train_X, task_feature, cat_dims=cat_dims)
        likelihood = StudentTLikelihood()
        likelihood.initialize(deg_free=df)
        self.response_model = SingleTaskVariationalGP(
            train_X,
            train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            inducing_points=min(int(num_inducing), train_X.shape[-2]),
        )
        self.task_feature = task_dim
        self.cat_dims = tuple(
            normalize_feature_dims(
                cat_dims,
                train_X.shape[-1],
                name="cat_dims",
                excluded_dims=[task_dim],
            )
        )
        self._store_raw_tensor("train_X", train_X.detach().clone())
        self._store_raw_tensor("train_Y", train_Y.detach().clone())
        self._store_raw_tensor("train_Yvar", None)
