"""Single-output deep Gaussian process foundation."""

from __future__ import annotations

from collections.abc import Sequence

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import Tensor

from robotorchan.models.base import ModelTrainingMixin, SupervisedTrainingDataMixin


class _DeepGPLayer(DeepGPLayer):
    """One variational GP mapping used inside the single-task DeepGP."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int | None,
        num_inducing: int,
        linear_mean: bool,
        reference: Tensor,
    ) -> None:
        batch_shape = torch.Size() if output_dim is None else torch.Size([output_dim])
        inducing_shape = (
            (num_inducing, input_dim)
            if output_dim is None
            else (output_dim, num_inducing, input_dim)
        )
        inducing_points = torch.randn(
            inducing_shape,
            device=reference.device,
            dtype=reference.dtype,
        )
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape,
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(
            variational_strategy=variational_strategy,
            input_dims=input_dim,
            output_dims=output_dim,
        )
        self.mean_module = (
            LinearMean(input_dim, batch_shape=batch_shape)
            if linear_mean
            else ConstantMean(batch_shape=batch_shape)
        )
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dim),
            batch_shape=batch_shape,
        )

    def forward(self, X: Tensor) -> MultivariateNormal:
        """Evaluate this stochastic GP mapping."""
        return MultivariateNormal(self.mean_module(X), self.covar_module(X))


class SingleTaskDeepGP(
    SupervisedTrainingDataMixin,
    ModelTrainingMixin,
    DeepGP,
):
    """Single-output variational deep Gaussian process.

    This phase establishes the stochastic DeepGP hierarchy and training
    objective. BoTorch posterior and acquisition integration is added in the
    following phase rather than approximating the DeepGP as an exact GP.
    """

    supports_mll = True

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        hidden_dims: Sequence[int] = (4,),
        num_inducing: int = 16,
        standardize_inputs: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape n x d.")
        if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
            raise ValueError("train_Y must have shape n x 1.")
        if train_X.shape[-2] != train_Y.shape[-2]:
            raise ValueError("train_X and train_Y must contain the same number of rows.")
        hidden_dims = tuple(int(width) for width in hidden_dims)
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain at least one positive integer.")
        if num_inducing <= 0:
            raise ValueError("num_inducing must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        super().__init__()
        self._store_supervised_training_data(train_X, train_Y)
        self.hidden_dims = hidden_dims
        self.num_inducing = int(num_inducing)
        self.standardize_inputs = bool(standardize_inputs)
        self.eps = float(eps)

        x_mean = train_X.mean(dim=-2, keepdim=True)
        x_scale = train_X.std(dim=-2, keepdim=True).clamp_min(eps)
        if not standardize_inputs:
            x_mean = torch.zeros_like(x_mean)
            x_scale = torch.ones_like(x_scale)
        self.register_buffer("input_mean", x_mean)
        self.register_buffer("input_scale", x_scale)

        with torch.random.fork_rng():
            torch.manual_seed(random_state)
            layers: list[_DeepGPLayer] = []
            input_dim = train_X.shape[-1]
            for width in hidden_dims:
                layers.append(
                    _DeepGPLayer(
                        input_dim=input_dim,
                        output_dim=width,
                        num_inducing=num_inducing,
                        linear_mean=True,
                        reference=train_X,
                    )
                )
                input_dim = width
            self.hidden_layers = torch.nn.ModuleList(layers)
            self.output_layer = _DeepGPLayer(
                input_dim=input_dim,
                output_dim=None,
                num_inducing=num_inducing,
                linear_mean=False,
                reference=train_X,
            )
        self.likelihood = GaussianLikelihood().to(train_X)

    def transform_inputs(self, X: Tensor) -> Tensor:
        """Apply the model-owned input standardization."""
        if X.shape[-1] != self.raw_train_X.shape[-1]:
            raise ValueError(
                f"Expected {self.raw_train_X.shape[-1]} input features, got {X.shape[-1]}."
            )
        return (X - self.input_mean) / self.input_scale

    def forward(self, X: Tensor) -> MultivariateNormal:
        """Propagate inputs through all stochastic GP layers."""
        hidden = self.transform_inputs(X)
        for layer in self.hidden_layers:
            hidden = layer(hidden)
        return self.output_layer(hidden)

    def make_mll(self, num_data: int | None = None) -> DeepApproximateMLL:
        """Construct the DeepGP variational ELBO objective."""
        if num_data is None:
            num_data = self.raw_train_X.shape[-2]
        if num_data <= 0:
            raise ValueError("num_data must be positive.")
        return DeepApproximateMLL(
            VariationalELBO(
                likelihood=self.likelihood,
                model=self,
                num_data=num_data,
            )
        )

    def training_loss(
        self,
        X: Tensor | None = None,
        Y: Tensor | None = None,
        *,
        num_data: int | None = None,
        num_likelihood_samples: int = 8,
    ) -> Tensor:
        """Return the negative DeepGP ELBO for a full batch or minibatch."""
        X = self.raw_train_X if X is None else X
        Y = self.raw_train_Y if Y is None else Y
        if Y.ndim != 2 or Y.shape[-1] != 1:
            raise ValueError("Y must have shape n x 1.")
        if X.shape[-2] != Y.shape[-2]:
            raise ValueError("X and Y must contain the same number of rows.")
        if num_likelihood_samples <= 0:
            raise ValueError("num_likelihood_samples must be positive.")

        mll = self.make_mll(num_data=num_data)
        with gpytorch.settings.num_likelihood_samples(num_likelihood_samples):
            output = self(X)
            return -mll(output, Y.squeeze(-1))
