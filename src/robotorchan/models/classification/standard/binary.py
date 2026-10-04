"""Binary variational GP classifier."""

from __future__ import annotations

import torch
from botorch.models.transforms.input import InputTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from gpytorch.kernels import IndexKernel, Kernel, ProductKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import Mean
from gpytorch.mlls import VariationalELBO
from gpytorch.utils.memoize import clear_cache_hook
from gpytorch.variational import VariationalStrategy, _VariationalDistribution, _VariationalStrategy
from torch import Tensor, nn
from torch.distributions import Bernoulli

from robotorchan.models.base import (
    ContinuousKernelFactory,
    make_mixed_covar_module,
    normalize_feature_dims,
)
from robotorchan.models.classification.base import BinaryClassificationMixin
from robotorchan.models.classification.validation import validate_binary_labels
from robotorchan.models.standard.variational import SingleTaskVariationalGP


class BinarySingleTaskGPClassifier(BinaryClassificationMixin, SingleTaskVariationalGP):
    """Binary GP classifier backed by BoTorch variational GP infrastructure."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        learn_inducing_points: bool = True,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize a binary variational GP with Bernoulli likelihood."""
        validate_binary_labels(train_Y)
        if train_X.shape[-2] != train_Y.shape[0]:
            raise ValueError("train_X and train_Y must contain the same number of rows.")
        raw_train_Y = train_Y.detach().clone()
        model_train_Y = train_Y.unsqueeze(-1) if train_Y.ndim == train_X.ndim - 1 else train_Y
        model_train_Y = model_train_Y.to(dtype=train_X.dtype, device=train_X.device)
        super().__init__(
            train_X=train_X,
            train_Y=model_train_Y,
            likelihood=BernoulliLikelihood(),
            num_outputs=1,
            learn_inducing_points=learn_inducing_points,
            covar_module=covar_module,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=inducing_point_allocator,
            outcome_transform=None,
            input_transform=input_transform,
        )
        self._store_raw_tensor("train_Y", raw_train_Y)

    def load_state_dict(
        self,
        state_dict: dict[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load variational classifier state without Gaussian target extraction."""
        result = nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)
        self.model.variational_strategy.variational_params_initialized.fill_(1)
        self.apply(clear_cache_hook)
        return result

    def make_mll(self, num_data: int | None = None) -> VariationalELBO:
        """Construct the Bernoulli variational evidence lower bound.

        Args:
            num_data: Total number of binary observations represented by the
                ELBO. Defaults to the number of caller-supplied training rows.

        Returns:
            Variational ELBO bound to this classifier likelihood and latent GP.
        """
        return super().make_mll(num_data=num_data)

    def predictive_distribution(self, X: Tensor, **kwargs: object) -> Bernoulli:
        """Return Bernoulli predictions after integrating latent uncertainty."""
        latent = self.latent_posterior(X, **kwargs)
        return self.likelihood(latent.distribution)

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return ``[P(y=0), P(y=1)]`` along the final dimension."""
        positive = self.predictive_distribution(X, **kwargs).probs
        if positive.shape[-1:] == (1,):
            positive = positive.squeeze(-1)
        return torch.stack((1.0 - positive, positive), dim=-1)

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Draw class probabilities by mapping latent posterior samples."""
        latent_samples = self.sample_latent(X, sample_shape=sample_shape, **kwargs)
        positive = self.likelihood.forward(latent_samples).probs
        if positive.shape[-1:] == (1,):
            positive = positive.squeeze(-1)
        return torch.stack((1.0 - positive, positive), dim=-1)

    def predictive_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return Bernoulli variance for each class probability."""
        probabilities = self.predict_proba(X, **kwargs)
        return probabilities * (1.0 - probabilities)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return Bernoulli predictive entropy for each input point."""
        probabilities = self.predict_proba(X, **kwargs)
        terms = torch.special.xlogy(probabilities, probabilities)
        return -terms.sum(dim=-1)

    def predict_class(
        self,
        X: Tensor,
        *,
        threshold: float = 0.5,
        **kwargs: object,
    ) -> Tensor:
        """Return binary class labels using the requested probability threshold."""
        if not isinstance(threshold, int | float):
            raise TypeError("threshold must be a real number.")
        if not torch.isfinite(torch.tensor(threshold)):
            raise ValueError("threshold must be finite.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        positive = self.predict_proba(X, **kwargs)[..., 1]
        return torch.where(
            positive >= threshold,
            torch.ones_like(positive, dtype=torch.long),
            torch.zeros_like(positive, dtype=torch.long),
        )


class MixedBinarySingleTaskGPClassifier(BinarySingleTaskGPClassifier):
    """Binary variational GP classifier for native mixed continuous/categorical inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: list[int],
        learn_inducing_points: bool = True,
        cont_kernel_factory: ContinuousKernelFactory | None = None,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize a binary classifier with native categorical covariance."""
        covar_module = make_mixed_covar_module(
            input_dim=train_X.shape[-1],
            cat_dims=cat_dims,
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=cont_kernel_factory,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            learn_inducing_points=learn_inducing_points,
            covar_module=covar_module,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=inducing_point_allocator,
            input_transform=input_transform,
        )
        self.cat_dims = normalize_feature_dims(
            cat_dims,
            train_X.shape[-1],
            name="cat_dims",
        )


class MultiTaskBinaryGPClassifier(BinarySingleTaskGPClassifier):
    """Long-format binary variational GP classifier with a structural task feature."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        *,
        rank: int | None = None,
        learn_inducing_points: bool = True,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize an intrinsic-coregionalization binary classifier."""
        input_dim = train_X.shape[-1]
        resolved_task_feature = normalize_feature_dims(
            [task_feature],
            input_dim,
            name="task_feature",
        )[0]
        task_values = torch.unique(train_X[..., resolved_task_feature])
        if task_values.numel() < 1:
            raise ValueError("task_feature must contain at least one task value.")
        rounded = task_values.round()
        if not torch.allclose(task_values, rounded):
            raise ValueError("task_feature values must be integer task identifiers.")
        task_ids = rounded.to(dtype=torch.long)
        if int(task_ids.min()) < 0:
            raise ValueError("task_feature values must be non-negative.")
        num_tasks = int(task_ids.max()) + 1
        resolved_rank = min(num_tasks, 1) if rank is None else rank
        if resolved_rank < 1 or resolved_rank > num_tasks:
            raise ValueError("rank must be between 1 and the number of indexed tasks.")

        data_dims = [dim for dim in range(input_dim) if dim != resolved_task_feature]
        if not data_dims:
            raise ValueError("train_X must contain at least one non-task feature.")
        data_kernel = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=len(data_dims),
            batch_shape=train_X.shape[:-2],
            active_dims=data_dims,
        )
        task_kernel = IndexKernel(
            num_tasks=num_tasks,
            rank=resolved_rank,
            active_dims=[resolved_task_feature],
        )
        covar_module = ProductKernel(data_kernel, task_kernel)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            learn_inducing_points=learn_inducing_points,
            covar_module=covar_module,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=inducing_point_allocator,
            input_transform=input_transform,
        )
        self.task_feature = resolved_task_feature
        self.num_tasks = num_tasks
        self.rank = resolved_rank


class KroneckerMultiTaskBinaryGPClassifier(MultiTaskBinaryGPClassifier):
    """Block-design binary classifier using the variational ICM backend."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, *, rank: int | None = None) -> None:
        """Initialize from inputs shared by every binary classification task."""
        validate_binary_labels(train_Y)
        if train_X.ndim != 2 or train_Y.ndim != 2:
            raise ValueError("train_X and train_Y must have shapes n x d and n x m.")
        if train_X.shape[0] != train_Y.shape[0]:
            raise ValueError("train_X and train_Y must contain the same number of rows.")
        if train_Y.shape[1] < 1:
            raise ValueError("train_Y must contain at least one task.")
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        num_rows, num_tasks = train_Y.shape
        expanded_X = train_X.unsqueeze(1).expand(num_rows, num_tasks, train_X.shape[-1])
        task_ids = torch.arange(num_tasks, device=train_X.device, dtype=train_X.dtype)
        tasks = task_ids.view(1, num_tasks, 1).expand(num_rows, num_tasks, 1)
        long_X = torch.cat((expanded_X, tasks), dim=-1).reshape(num_rows * num_tasks, -1)
        long_Y = train_Y.reshape(num_rows * num_tasks)
        super().__init__(long_X, long_Y, task_feature=-1, rank=rank)
        self._store_raw_tensor("train_X", raw_train_X)
        self._store_raw_tensor("train_Y", raw_train_Y)
        self.block_input_dim = train_X.shape[-1]

    def _expand_block_X(self, X: Tensor) -> Tensor:
        """Expand shared task inputs to the internal long-format representation."""
        if X.shape[-1] != self.block_input_dim:
            raise ValueError(f"Expected X with {self.block_input_dim} features.")
        task_ids = torch.arange(self.num_tasks, device=X.device, dtype=X.dtype)
        expanded_X = X.unsqueeze(-2).expand(*X.shape[:-1], self.num_tasks, X.shape[-1])
        shape = (1,) * (X.ndim - 1) + (self.num_tasks, 1)
        tasks = task_ids.reshape(shape).expand(*X.shape[:-1], self.num_tasks, 1)
        return torch.cat((expanded_X, tasks), dim=-1)

    def latent_posterior(self, X: Tensor, **kwargs: object):
        """Return task-aware latent posterior for block-design inputs."""
        return super().latent_posterior(self._expand_block_X(X), **kwargs)

    def sample_latent(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Draw task-aware latent samples for block-design inputs."""
        return super().sample_latent(
            self._expand_block_X(X),
            sample_shape=sample_shape,
            **kwargs,
        )

    def predictive_distribution(self, X: Tensor, **kwargs: object) -> Bernoulli:
        """Return task-aware Bernoulli predictions for block-design inputs."""
        latent = self.latent_posterior(X, **kwargs)
        return self.likelihood(latent.distribution)

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Draw task-aware class-probability samples for block-design inputs."""
        latent_samples = self.sample_latent(X, sample_shape=sample_shape, **kwargs)
        positive = self.likelihood.forward(latent_samples).probs
        if positive.shape[-1:] == (1,):
            positive = positive.squeeze(-1)
        return torch.stack((1.0 - positive, positive), dim=-1)

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return probabilities with task and class dimensions."""
        positive = self.predictive_distribution(X, **kwargs).probs
        if positive.shape[-1:] == (1,):
            positive = positive.squeeze(-1)
        return torch.stack((1.0 - positive, positive), dim=-1)

    def predict_class(
        self,
        X: Tensor,
        *,
        threshold: float = 0.5,
        **kwargs: object,
    ) -> Tensor:
        """Return one binary class label for every input-task pair."""
        if not isinstance(threshold, int | float):
            raise TypeError("threshold must be a real number.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        positive = self.predict_proba(X, **kwargs)[..., 1]
        return torch.where(
            positive >= threshold,
            torch.ones_like(positive, dtype=torch.long),
            torch.zeros_like(positive, dtype=torch.long),
        )
