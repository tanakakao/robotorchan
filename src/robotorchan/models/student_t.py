"""Variational GP surrogate with Student-t observation noise."""

from __future__ import annotations

from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.likelihoods import StudentTLikelihood
from torch import Tensor, nn

from robotorchan.models.base import RawDataMixin
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
