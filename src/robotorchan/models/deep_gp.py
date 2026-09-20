"""Deep Gaussian process surrogates for single-task and long-format multi-task data."""

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
from robotorchan.models.deep_gp_posterior import DeepGPPosterior


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
        self.to(device=reference.device, dtype=reference.dtype)

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
        posterior_samples: int = 64,
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
        if posterior_samples <= 1:
            raise ValueError("posterior_samples must be greater than one.")

        super().__init__()
        self._store_supervised_training_data(train_X, train_Y)
        self.hidden_dims = hidden_dims
        self.num_inducing = int(num_inducing)
        self.standardize_inputs = bool(standardize_inputs)
        self.eps = float(eps)
        self.posterior_samples = int(posterior_samples)

        x_mean = train_X.mean(dim=-2, keepdim=True)
        x_scale = train_X.std(dim=-2, keepdim=True, correction=0).clamp_min(eps)
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

    @property
    def num_outputs(self) -> int:
        """Return the number of modeled outputs."""
        return 1

    def posterior(
        self,
        X: Tensor,
        *,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        num_samples: int | None = None,
    ) -> DeepGPPosterior:
        """Return a BoTorch-compatible Monte Carlo DeepGP posterior."""
        if isinstance(observation_noise, Tensor):
            raise NotImplementedError("Tensor-valued observation_noise is not supported.")
        num_samples = self.posterior_samples if num_samples is None else int(num_samples)
        if num_samples <= 1:
            raise ValueError("num_samples must be greater than one.")

        with gpytorch.settings.num_likelihood_samples(num_samples):
            distribution = self(X)
            if observation_noise:
                distribution = self.likelihood(distribution)
            samples = distribution.rsample()
        if samples.shape[0] != num_samples:
            samples = samples.unsqueeze(0).expand(num_samples, *samples.shape)
        posterior = DeepGPPosterior(samples.unsqueeze(-1))
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

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


class MultiTaskDeepGP(SingleTaskDeepGP):
    """Long-format DeepGP with an explicit task feature.

    The task feature is represented by a learned embedding and is not
    standardized as a continuous data feature. The public posterior accepts
    the original long-format input including the task column.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        task_feature: int,
        task_embedding_dim: int = 2,
        hidden_dims: Sequence[int] = (4,),
        num_inducing: int = 16,
        standardize_inputs: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        posterior_samples: int = 64,
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape n x d.")
        input_dim = train_X.shape[-1]
        task_dim = int(task_feature)
        if task_dim < 0:
            task_dim += input_dim
        if task_dim < 0 or task_dim >= input_dim:
            raise ValueError("task_feature is out of range.")
        if task_embedding_dim <= 0:
            raise ValueError("task_embedding_dim must be positive.")

        task_values = train_X[..., task_dim]
        if not torch.allclose(task_values, task_values.round()):
            raise ValueError("task feature values must be integer-valued.")
        task_indices = task_values.long()
        if torch.any(task_indices < 0):
            raise ValueError("task feature values must be non-negative.")
        unique_tasks = torch.unique(task_indices, sorted=True)
        expected = torch.arange(unique_tasks.numel(), device=train_X.device)
        if not torch.equal(unique_tasks, expected):
            raise ValueError("task feature values must be contiguous and zero-based.")

        data_dims = tuple(i for i in range(input_dim) if i != task_dim)
        data_X = train_X[..., list(data_dims)]
        num_tasks = int(unique_tasks.numel())
        with torch.random.fork_rng():
            torch.manual_seed(random_state)
            task_embedding = torch.nn.Embedding(num_tasks, int(task_embedding_dim)).to(train_X)
            embedded_tasks = task_embedding(task_indices)
        deep_X = torch.cat((data_X, embedded_tasks.detach()), dim=-1)

        super().__init__(
            deep_X,
            train_Y,
            hidden_dims=hidden_dims,
            num_inducing=num_inducing,
            standardize_inputs=False,
            eps=eps,
            random_state=random_state,
            posterior_samples=posterior_samples,
        )
        self.task_embedding = task_embedding
        self.task_feature = task_dim
        self.task_embedding_dim = int(task_embedding_dim)
        self.num_tasks = num_tasks
        self._data_dims = data_dims
        self.standardize_inputs = bool(standardize_inputs)

        data_mean = data_X.mean(dim=-2, keepdim=True)
        data_scale = data_X.std(dim=-2, keepdim=True, correction=0).clamp_min(eps)
        if not standardize_inputs:
            data_mean = torch.zeros_like(data_mean)
            data_scale = torch.ones_like(data_scale)
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_scale", data_scale)
        self._store_supervised_training_data(train_X, train_Y)

    def transform_inputs(self, X: Tensor) -> Tensor:
        """Encode original long-format inputs for the DeepGP hierarchy."""
        if X.shape[-1] != self.raw_train_X.shape[-1]:
            raise ValueError(
                f"Expected {self.raw_train_X.shape[-1]} input features, got {X.shape[-1]}."
            )
        data = X[..., list(self._data_dims)]
        data = (data - self.data_mean) / self.data_scale
        task_values = X[..., self.task_feature]
        if not torch.allclose(task_values, task_values.round()):
            raise ValueError("task feature values must be integer-valued.")
        task_indices = task_values.long()
        if torch.any(task_indices < 0) or torch.any(task_indices >= self.num_tasks):
            raise ValueError("task feature contains an unknown task index.")
        return torch.cat((data, self.task_embedding(task_indices)), dim=-1)

    def forward(self, X: Tensor) -> MultivariateNormal:
        """Propagate original long-format inputs through the DeepGP."""
        hidden = self.transform_inputs(X)
        for layer in self.hidden_layers:
            hidden = layer(hidden)
        return self.output_layer(hidden)


class MixedSingleTaskDeepGP(SingleTaskDeepGP):
    """DeepGP for mixed continuous and categorical single-task inputs.

    Continuous features are standardized while categorical features are
    represented by learned embeddings. The public API always uses the original
    mixed input space.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        categorical_embedding_dim: int = 2,
        hidden_dims: Sequence[int] = (4,),
        num_inducing: int = 16,
        standardize_inputs: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        posterior_samples: int = 64,
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape n x d.")
        input_dim = train_X.shape[-1]
        cats = tuple(sorted({dim + input_dim if dim < 0 else dim for dim in cat_dims}))
        if not cats or any(dim < 0 or dim >= input_dim for dim in cats):
            raise ValueError("cat_dims must contain valid categorical feature indices.")
        if categorical_embedding_dim <= 0:
            raise ValueError("categorical_embedding_dim must be positive.")
        continuous_dims = tuple(i for i in range(input_dim) if i not in set(cats))
        if not continuous_dims:
            raise ValueError("MixedSingleTaskDeepGP requires at least one continuous feature.")

        continuous_X = train_X[..., list(continuous_dims)]
        category_sizes = []
        category_indices = []
        for dim in cats:
            values = train_X[..., dim]
            if not torch.allclose(values, values.round()):
                raise ValueError("categorical feature values must be integer-valued.")
            indices = values.long()
            if torch.any(indices < 0):
                raise ValueError("categorical feature values must be non-negative.")
            unique = torch.unique(indices, sorted=True)
            expected = torch.arange(unique.numel(), device=train_X.device)
            if not torch.equal(unique, expected):
                raise ValueError("categorical values must be contiguous and zero-based.")
            category_sizes.append(int(unique.numel()))
            category_indices.append(indices)

        with torch.random.fork_rng():
            torch.manual_seed(random_state)
            embeddings = torch.nn.ModuleList(
                [
                    torch.nn.Embedding(size, int(categorical_embedding_dim)).to(train_X)
                    for size in category_sizes
                ]
            )
            embedded = [
                embedding(indices)
                for embedding, indices in zip(embeddings, category_indices, strict=True)
            ]
        deep_X = torch.cat((continuous_X, *(item.detach() for item in embedded)), dim=-1)

        super().__init__(
            deep_X,
            train_Y,
            hidden_dims=hidden_dims,
            num_inducing=num_inducing,
            standardize_inputs=False,
            eps=eps,
            random_state=random_state,
            posterior_samples=posterior_samples,
        )
        self.category_embeddings = embeddings
        self.cat_dims = cats
        self.categorical_embedding_dim = int(categorical_embedding_dim)
        self.category_sizes = tuple(category_sizes)
        self._continuous_dims = continuous_dims
        self.standardize_inputs = bool(standardize_inputs)
        continuous_mean = continuous_X.mean(dim=-2, keepdim=True)
        continuous_scale = continuous_X.std(dim=-2, keepdim=True, correction=0).clamp_min(eps)
        if not standardize_inputs:
            continuous_mean = torch.zeros_like(continuous_mean)
            continuous_scale = torch.ones_like(continuous_scale)
        self.register_buffer("continuous_mean", continuous_mean)
        self.register_buffer("continuous_scale", continuous_scale)
        self._store_supervised_training_data(train_X, train_Y)

    def transform_inputs(self, X: Tensor) -> Tensor:
        """Encode original mixed inputs for the DeepGP hierarchy."""
        if X.shape[-1] != self.raw_train_X.shape[-1]:
            raise ValueError(
                f"Expected {self.raw_train_X.shape[-1]} input features, got {X.shape[-1]}."
            )
        continuous = X[..., list(self._continuous_dims)]
        continuous = (continuous - self.continuous_mean) / self.continuous_scale
        embedded = []
        for dim, size, embedding in zip(
            self.cat_dims,
            self.category_sizes,
            self.category_embeddings,
            strict=True,
        ):
            values = X[..., dim]
            if not torch.allclose(values, values.round()):
                raise ValueError("categorical feature values must be integer-valued.")
            indices = values.long()
            if torch.any(indices < 0) or torch.any(indices >= size):
                raise ValueError("categorical feature contains an unknown category index.")
            embedded.append(embedding(indices))
        return torch.cat((continuous, *embedded), dim=-1)

    def forward(self, X: Tensor) -> MultivariateNormal:
        """Propagate original mixed inputs through the DeepGP."""
        hidden = self.transform_inputs(X)
        for layer in self.hidden_layers:
            hidden = layer(hidden)
        return self.output_layer(hidden)


class MixedMultiTaskDeepGP(SingleTaskDeepGP):
    """DeepGP for mixed continuous/categorical long-format multi-task inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        task_feature: int,
        cat_dims: Sequence[int],
        task_embedding_dim: int = 2,
        categorical_embedding_dim: int = 2,
        hidden_dims: Sequence[int] = (4,),
        num_inducing: int = 16,
        standardize_inputs: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        posterior_samples: int = 64,
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape n x d.")
        input_dim = train_X.shape[-1]
        task_dim = task_feature + input_dim if task_feature < 0 else task_feature
        if task_dim < 0 or task_dim >= input_dim:
            raise ValueError("task_feature is out of range.")
        cats = tuple(sorted({dim + input_dim if dim < 0 else dim for dim in cat_dims}))
        if not cats or any(dim < 0 or dim >= input_dim for dim in cats):
            raise ValueError("cat_dims must contain valid categorical feature indices.")
        if task_dim in cats:
            raise ValueError("cat_dims must not contain task_feature.")
        if task_embedding_dim <= 0 or categorical_embedding_dim <= 0:
            raise ValueError("embedding dimensions must be positive.")
        continuous_dims = tuple(i for i in range(input_dim) if i != task_dim and i not in cats)
        if not continuous_dims:
            raise ValueError("MixedMultiTaskDeepGP requires at least one continuous feature.")

        task_indices = self._validated_indices(train_X[..., task_dim], "task feature")
        unique_tasks = torch.unique(task_indices, sorted=True)
        expected_tasks = torch.arange(unique_tasks.numel(), device=train_X.device)
        if not torch.equal(unique_tasks, expected_tasks):
            raise ValueError("task feature values must be contiguous and zero-based.")
        category_indices = []
        category_sizes = []
        for dim in cats:
            indices = self._validated_indices(train_X[..., dim], "categorical feature")
            unique = torch.unique(indices, sorted=True)
            expected = torch.arange(unique.numel(), device=train_X.device)
            if not torch.equal(unique, expected):
                raise ValueError("categorical values must be contiguous and zero-based.")
            category_indices.append(indices)
            category_sizes.append(int(unique.numel()))

        continuous_X = train_X[..., list(continuous_dims)]
        with torch.random.fork_rng():
            torch.manual_seed(random_state)
            task_embedding = torch.nn.Embedding(
                int(unique_tasks.numel()), int(task_embedding_dim)
            ).to(train_X)
            category_embeddings = torch.nn.ModuleList(
                [
                    torch.nn.Embedding(size, int(categorical_embedding_dim)).to(train_X)
                    for size in category_sizes
                ]
            )
            embedded_categories = [
                embedding(indices)
                for embedding, indices in zip(
                    category_embeddings, category_indices, strict=True
                )
            ]
            deep_X = torch.cat(
                (
                    continuous_X,
                    *(item.detach() for item in embedded_categories),
                    task_embedding(task_indices).detach(),
                ),
                dim=-1,
            )

        super().__init__(
            deep_X,
            train_Y,
            hidden_dims=hidden_dims,
            num_inducing=num_inducing,
            standardize_inputs=False,
            eps=eps,
            random_state=random_state,
            posterior_samples=posterior_samples,
        )
        self.task_embedding = task_embedding
        self.category_embeddings = category_embeddings
        self.task_feature = task_dim
        self.cat_dims = cats
        self.num_tasks = int(unique_tasks.numel())
        self.task_embedding_dim = int(task_embedding_dim)
        self.categorical_embedding_dim = int(categorical_embedding_dim)
        self.category_sizes = tuple(category_sizes)
        self._continuous_dims = continuous_dims
        self.standardize_inputs = bool(standardize_inputs)
        mean = continuous_X.mean(dim=-2, keepdim=True)
        scale = continuous_X.std(dim=-2, keepdim=True, correction=0).clamp_min(eps)
        if not standardize_inputs:
            mean = torch.zeros_like(mean)
            scale = torch.ones_like(scale)
        self.register_buffer("continuous_mean", mean)
        self.register_buffer("continuous_scale", scale)
        self._store_supervised_training_data(train_X, train_Y)

    @staticmethod
    def _validated_indices(values: Tensor, name: str) -> Tensor:
        if not torch.allclose(values, values.round()):
            raise ValueError(f"{name} values must be integer-valued.")
        indices = values.long()
        if torch.any(indices < 0):
            raise ValueError(f"{name} values must be non-negative.")
        return indices

    def transform_inputs(self, X: Tensor) -> Tensor:
        """Encode continuous, categorical, and task structure independently."""
        if X.shape[-1] != self.raw_train_X.shape[-1]:
            raise ValueError(
                f"Expected {self.raw_train_X.shape[-1]} input features, got {X.shape[-1]}."
            )
        continuous = X[..., list(self._continuous_dims)]
        continuous = (continuous - self.continuous_mean) / self.continuous_scale
        parts = [continuous]
        for dim, size, embedding in zip(
            self.cat_dims, self.category_sizes, self.category_embeddings, strict=True
        ):
            indices = self._validated_indices(X[..., dim], "categorical feature")
            if torch.any(indices >= size):
                raise ValueError("categorical feature contains an unknown category index.")
            parts.append(embedding(indices))
        task_indices = self._validated_indices(X[..., self.task_feature], "task feature")
        if torch.any(task_indices >= self.num_tasks):
            raise ValueError("task feature contains an unknown task index.")
        parts.append(self.task_embedding(task_indices))
        return torch.cat(parts, dim=-1)

    def forward(self, X: Tensor) -> MultivariateNormal:
        """Propagate original mixed long-format inputs through the DeepGP."""
        hidden = self.transform_inputs(X)
        for layer in self.hidden_layers:
            hidden = layer(hidden)
        return self.output_layer(hidden)
