"""Latent Kronecker GP wrappers with robotorchan model conventions."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.latent_kronecker_gp import (
    LatentKroneckerGP as BoTorchLatentKroneckerGP,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.posteriors.latent_kronecker import LatentKroneckerGPPosterior
from botorch.posteriors.posterior import Posterior
from botorch.posteriors.transformed import TransformedPosterior
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.module import Module
from torch import Tensor

from robotorchan.models.base import (
    ContinuousKernelFactory,
    ExactGPModelMixin,
    make_mixed_covar_module,
    normalize_feature_dims,
)

# Track explicit contexts without storing transient state on the torch module.
_ACTIVE_ITERATIVE_MODELS: ContextVar[tuple[int, ...]] = ContextVar(
    "_ACTIVE_ITERATIVE_MODELS", default=()
)


class _IterativeLatentKroneckerPosterior(LatentKroneckerGPPosterior):
    """Keep upstream pathwise sampling and lazy moments inside solver settings."""

    @property
    def mean(self) -> Tensor:
        with self.model._iterative_context():
            return super().mean

    @property
    def variance(self) -> Tensor:
        with self.model._iterative_context():
            return super().variance

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        with self.model._iterative_context():
            return super().rsample(sample_shape=sample_shape)

    def rsample_from_base_samples(self, sample_shape: torch.Size, base_samples: Tensor) -> Tensor:
        with self.model._iterative_context():
            return super().rsample_from_base_samples(
                sample_shape=sample_shape, base_samples=base_samples
            )


class LatentKroneckerGP(ExactGPModelMixin, BoTorchLatentKroneckerGP):
    """BoTorch ``LatentKroneckerGP`` with robotorchan conveniences.

    Missing-output handling, product-space covariance structure, iterative
    inference, pathwise posterior sampling, and transforms remain delegated to
    BoTorch. robotorchan retains caller-supplied ``train_X``, ``train_T``, and
    ``train_Y`` before upstream broadcasting, masking, or transformations.
    """

    supports_fit = True

    @contextmanager
    @wraps(BoTorchLatentKroneckerGP.use_iterative_methods)
    def use_iterative_methods(self, *args, **kwargs):
        """Preserve explicit solver settings across upstream default-only calls."""
        if not args and not kwargs and id(self) in _ACTIVE_ITERATIVE_MODELS.get():
            yield
            return
        with super().use_iterative_methods(*args, **kwargs):
            token = _ACTIVE_ITERATIVE_MODELS.set((*_ACTIVE_ITERATIVE_MODELS.get(), id(self)))
            try:
                yield
            finally:
                _ACTIVE_ITERATIVE_MODELS.reset(token)

    @contextmanager
    def _iterative_context(self):
        """Use defaults unless this model already has explicit solver settings."""
        if id(self) in _ACTIVE_ITERATIVE_MODELS.get():
            yield
        else:
            with self.use_iterative_methods():
                yield

    def fit(self, *, optimizer_kwargs: dict | None = None) -> None:
        """Fit the native MLL in BoTorch's iterative-methods context."""
        with self._iterative_context():
            fit_gpytorch_mll(self.make_mll(), optimizer_kwargs=optimizer_kwargs)

    def posterior(self, X: Tensor, T: Tensor | None = None, **kwargs) -> Posterior:
        """Return original-scale moments without double-transforming pathwise samples.

        BoTorch's LatentKronecker posterior samples are already unstandardized,
        while its distribution moments are not. A TransformedPosterior with an
        identity sample transform preserves its specialized pathwise sampler.
        """
        with self._iterative_context():
            native = super().posterior(X, T=T, **kwargs)
            posterior = _IterativeLatentKroneckerPosterior(
                model=self, distribution=native.distribution, X=native.X, T=native.T
            )
        transform = getattr(self, "outcome_transform", None)
        if transform is None:
            return posterior
        if type(transform) is not Standardize:
            raise NotImplementedError(
                "LatentKroneckerGP original-scale posterior moments currently "
                "support only the default Standardize outcome transform or None."
            )

        def original_moments(mean: Tensor, variance: Tensor) -> tuple[Tensor, Tensor]:
            untransformed_mean, untransformed_variance = transform.untransform(
                mean.unsqueeze(-1), variance.unsqueeze(-1)
            )
            return (
                untransformed_mean.squeeze(-1),
                untransformed_variance.squeeze(-1),
            )

        return TransformedPosterior(
            posterior=posterior,
            sample_transform=lambda samples: samples,
            mean_transform=lambda mean, variance: original_moments(mean, variance)[0],
            variance_transform=lambda mean, variance: original_moments(mean, variance)[1],
        )

    def __init__(
        self,
        train_X: Tensor,
        train_T: Tensor,
        train_Y: Tensor,
        likelihood: Likelihood | None = None,
        mean_module_X: Mean | None = None,
        mean_module_T: Mean | None = None,
        covar_module_X: Module | None = None,
        covar_module_T: Module | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
    ) -> None:
        """Initialize the model while retaining the raw product-space tensors."""
        raw_train_X = train_X.detach().clone()
        raw_train_T = train_T.detach().clone()
        raw_train_Y = train_Y.detach().clone()

        super().__init__(
            train_X=train_X,
            train_T=train_T,
            train_Y=train_Y,
            likelihood=likelihood,
            mean_module_X=mean_module_X,
            mean_module_T=mean_module_T,
            covar_module_X=covar_module_X,
            covar_module_T=covar_module_T,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=None,
        )
        self._store_raw_tensor("train_T", raw_train_T)

    @property
    def raw_train_T(self) -> Tensor:
        """Caller-supplied task / time coordinates before upstream broadcasting."""
        value = self._get_raw_tensor("train_T")
        if value is None:  # pragma: no cover - guarded by constructor contract
            raise RuntimeError("raw_train_T was unexpectedly stored as None.")
        return value


class MixedLatentKroneckerGP(LatentKroneckerGP):
    """Latent Kronecker GP with native mixed covariance on the X factor.

    The Kronecker structure separates design covariance ``K_X`` from the
    task/time covariance ``K_T``. Since BoTorch exposes ``covar_module_X``
    directly, categorical design dimensions can use robotorchan's native mixed
    covariance without changing the latent Kronecker or iterative-inference
    semantics. ``train_T`` remains entirely under the upstream T covariance.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_T: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        likelihood: Likelihood | None = None,
        mean_module_X: Mean | None = None,
        mean_module_T: Mean | None = None,
        cont_kernel_factory: ContinuousKernelFactory | None = None,
        covar_module_T: Module | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
    ) -> None:
        input_dim = train_X.shape[-1]
        normalized_cat_dims = normalize_feature_dims(cat_dims, input_dim=input_dim)
        covar_module_X = make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=normalized_cat_dims,
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=cont_kernel_factory,
        )
        super().__init__(
            train_X=train_X,
            train_T=train_T,
            train_Y=train_Y,
            likelihood=likelihood,
            mean_module_X=mean_module_X,
            mean_module_T=mean_module_T,
            covar_module_X=covar_module_X,
            covar_module_T=covar_module_T,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        self.cat_dims = tuple(normalized_cat_dims)
