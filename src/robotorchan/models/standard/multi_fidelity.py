"""Multi-fidelity GP wrapper with robotorchan model conventions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from botorch.models import SingleTaskMultiFidelityGP as BoTorchSingleTaskMultiFidelityGP
from botorch.models.map_saas import add_saas_prior
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.module import Module
from torch import Tensor

from robotorchan.models.base import (
    ContinuousKernelFactory,
    ExactGPModelMixin,
    make_mixed_covar_module,
    normalize_feature_dims,
)
from robotorchan.reduction.input import (
    PCAInputReducer,
    PLSInputReducer,
    RandomProjectionInputReducer,
)


class SingleTaskMultiFidelityGP(ExactGPModelMixin, BoTorchSingleTaskMultiFidelityGP):
    """BoTorch ``SingleTaskMultiFidelityGP`` with robotorchan conveniences."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        linear_truncated: bool = True,
        nu: float = 2.5,
        covar_module: Module | None = None,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize the multi-fidelity GP while retaining raw training tensors."""
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            iteration_fidelity=iteration_fidelity,
            data_fidelities=data_fidelities,
            linear_truncated=linear_truncated,
            nu=nu,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=raw_train_Yvar,
        )


class MixedSingleTaskMultiFidelityGP(SingleTaskMultiFidelityGP):
    """Multi-fidelity GP with native categorical design covariance.

    Fidelity columns remain structural and are modeled by BoTorch's native
    fidelity kernels. The mixed covariance is restricted to non-fidelity
    design columns. The linear-truncated path is intentionally unsupported
    because it owns the complete input covariance and cannot preserve a
    separate mixed design covariance.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        linear_truncated: bool = False,
        nu: float = 2.5,
        cont_kernel_factory: ContinuousKernelFactory | None = None,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize a mixed-input multi-fidelity GP."""
        if linear_truncated:
            raise ValueError(
                "MixedSingleTaskMultiFidelityGP requires linear_truncated=False "
                "because the linear-truncated fidelity kernel owns the full "
                "input covariance."
            )

        input_dim = train_X.shape[-1]
        fidelity_dims: list[int] = []
        if iteration_fidelity is not None:
            fidelity_dims.extend(
                normalize_feature_dims([iteration_fidelity], input_dim, name="iteration_fidelity")
            )
        if data_fidelities is not None:
            fidelity_dims.extend(
                normalize_feature_dims(data_fidelities, input_dim, name="data_fidelities")
            )
        if len(set(fidelity_dims)) != len(fidelity_dims):
            raise ValueError("Fidelity dimensions must not contain duplicates.")

        normalized_cat_dims = normalize_feature_dims(
            cat_dims,
            input_dim,
            name="cat_dims",
            excluded_dims=fidelity_dims,
        )
        batch_shape = train_X.shape[:-2]
        if train_Y.shape[-1] > 1:
            batch_shape = torch.Size((*batch_shape, train_Y.shape[-1]))
        data_covar_module = make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=normalized_cat_dims,
            excluded_dims=fidelity_dims,
            batch_shape=batch_shape,
            cont_kernel_factory=cont_kernel_factory,
        )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            iteration_fidelity=iteration_fidelity,
            data_fidelities=data_fidelities,
            linear_truncated=False,
            nu=nu,
            covar_module=data_covar_module,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.cat_dims = tuple(normalized_cat_dims)
        self.fidelity_dims = tuple(sorted(fidelity_dims))


class MapSaasMultiFidelityGP(SingleTaskMultiFidelityGP):
    """Exact multi-fidelity GP with SAAS shrinkage on design dimensions only."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        linear_truncated: bool = False,
        nu: float = 2.5,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize design-only MAP-SAAS with native BoTorch fidelity kernels."""
        if linear_truncated:
            raise ValueError("MapSaasMultiFidelityGP requires linear_truncated=False.")

        input_dim = train_X.shape[-1]
        fidelity_dims: list[int] = []
        normalized_iteration: int | None = None
        if iteration_fidelity is not None:
            normalized_iteration = normalize_feature_dims(
                [iteration_fidelity], input_dim, name="iteration_fidelity"
            )[0]
            fidelity_dims.append(normalized_iteration)
        normalized_data = tuple(
            normalize_feature_dims(
                () if data_fidelities is None else data_fidelities,
                input_dim,
                name="data_fidelities",
            )
        )
        fidelity_dims.extend(normalized_data)
        if len(set(fidelity_dims)) != len(fidelity_dims):
            raise ValueError("Fidelity dimensions must not contain duplicates.")

        design_dims = tuple(dim for dim in range(input_dim) if dim not in fidelity_dims)
        if not design_dims:
            raise ValueError("MapSaasMultiFidelityGP requires at least one design dimension.")

        design_kernel = MaternKernel(
            nu=nu,
            ard_num_dims=len(design_dims),
            active_dims=list(design_dims),
        )
        add_saas_prior(design_kernel)
        data_covar_module = ScaleKernel(design_kernel)

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            iteration_fidelity=normalized_iteration,
            data_fidelities=normalized_data,
            linear_truncated=False,
            nu=nu,
            covar_module=data_covar_module,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.design_dims = design_dims
        self.fidelity_dims = tuple(fidelity_dims)
        self.iteration_fidelity = normalized_iteration
        self.data_fidelities = normalized_data
        self.saas_design_kernel = design_kernel


class PCAMultiFidelityGP(SingleTaskMultiFidelityGP):
    """Multi-fidelity GP with PCA restricted to non-fidelity design features."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        n_components: int,
        train_Yvar: Tensor | None = None,
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        *,
        center: bool = True,
        linear_truncated: bool = False,
        nu: float = 2.5,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        input_dim = train_X.shape[-1]
        fidelity_dims: list[int] = []
        if iteration_fidelity is not None:
            fidelity_dims.extend(
                normalize_feature_dims([iteration_fidelity], input_dim, name="iteration_fidelity")
            )
        if data_fidelities is not None:
            fidelity_dims.extend(
                normalize_feature_dims(data_fidelities, input_dim, name="data_fidelities")
            )
        if not fidelity_dims:
            raise ValueError("At least one fidelity feature must be specified.")
        if len(set(fidelity_dims)) != len(fidelity_dims):
            raise ValueError("Fidelity dimensions must not contain duplicates.")
        if linear_truncated:
            raise ValueError(
                "PCAMultiFidelityGP requires linear_truncated=False so PLS can remain "
                "separate from fidelity covariance."
            )

        normalized_iteration = (
            None
            if iteration_fidelity is None
            else normalize_feature_dims([iteration_fidelity], input_dim, name="iteration_fidelity")[
                0
            ]
        )
        normalized_data = tuple(
            normalize_feature_dims(data_fidelities or (), input_dim, name="data_fidelities")
        )

        self.raw_input_dim = input_dim
        self.iteration_fidelity = normalized_iteration
        self.data_fidelities = normalized_data
        self.fidelity_dims = tuple(sorted(fidelity_dims))
        self.design_dims = tuple(dim for dim in range(input_dim) if dim not in self.fidelity_dims)
        if not self.design_dims:
            raise ValueError("PCA requires at least one non-fidelity design feature.")

        design_X = train_X[..., list(self.design_dims)]
        reducer = PCAInputReducer(n_components=n_components, center=center)
        reduced_design_X = reducer.fit_transform(design_X, train_Y)
        encoded_train_X = self._join_design_and_fidelity(
            reduced_design_X,
            train_X[..., list(self.fidelity_dims)],
        )
        encoded_fidelity_dims = list(range(reduced_design_X.shape[-1], encoded_train_X.shape[-1]))
        raw_to_encoded = dict(zip(self.fidelity_dims, encoded_fidelity_dims, strict=True))
        encoded_iteration = (
            None if normalized_iteration is None else raw_to_encoded[normalized_iteration]
        )
        encoded_data = [raw_to_encoded[dim] for dim in normalized_data]

        super().__init__(
            train_X=encoded_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            iteration_fidelity=encoded_iteration,
            data_fidelities=encoded_data,
            linear_truncated=False,
            nu=nu,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.input_reducer = reducer
        self.reduced_design_dim = reduced_design_X.shape[-1]
        self.encoded_fidelity_dims = tuple(encoded_fidelity_dims)
        self._store_supervised_training_data(train_X, train_Y, train_Yvar)

    @staticmethod
    def _join_design_and_fidelity(design_X: Tensor, fidelity_X: Tensor) -> Tensor:
        return torch.cat((design_X, fidelity_X), dim=-1)

    def _encode_inputs(self, X: Tensor) -> Tensor:
        if X.shape[-1] != self.raw_input_dim:
            raise ValueError(
                f"Expected final input dimension {self.raw_input_dim}, got {X.shape[-1]}."
            )
        design_X = X[..., list(self.design_dims)]
        fidelity_X = X[..., list(self.fidelity_dims)]
        return self._join_design_and_fidelity(
            self.input_reducer.transform(design_X),
            fidelity_X,
        )

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the BoTorch multi-fidelity posterior from raw-space inputs."""
        return super().posterior(self._encode_inputs(X), *args, **kwargs)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Any:
        """Condition the encoded GP using observations supplied in raw coordinates."""
        return super().condition_on_observations(X=self._encode_inputs(X), Y=Y, **kwargs)

    def fantasize(self, X: Tensor, sampler: Any, **kwargs: Any) -> Any:
        """Construct fantasy models from candidate inputs supplied in raw coordinates."""
        encoded_X = self._encode_inputs(X)
        posterior = super().posterior(encoded_X, observation_noise=True)
        fantasy_Y = sampler(posterior)
        return super().condition_on_observations(X=encoded_X, Y=fantasy_Y, **kwargs)


class RandomProjectionMultiFidelityGP(SingleTaskMultiFidelityGP):
    """Multi-fidelity GP with random projection restricted to non-fidelity design features."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        n_components: int,
        train_Yvar: Tensor | None = None,
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        *,
        random_state: int = 0,
        linear_truncated: bool = False,
        nu: float = 2.5,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        input_dim = train_X.shape[-1]
        fidelity_dims: list[int] = []
        if iteration_fidelity is not None:
            fidelity_dims.extend(
                normalize_feature_dims([iteration_fidelity], input_dim, name="iteration_fidelity")
            )
        if data_fidelities is not None:
            fidelity_dims.extend(
                normalize_feature_dims(data_fidelities, input_dim, name="data_fidelities")
            )
        if not fidelity_dims:
            raise ValueError("At least one fidelity feature must be specified.")
        if len(set(fidelity_dims)) != len(fidelity_dims):
            raise ValueError("Fidelity dimensions must not contain duplicates.")
        if linear_truncated:
            raise ValueError(
                "RandomProjectionMultiFidelityGP requires linear_truncated=False so "
                "random projection can remain separate from fidelity covariance."
            )

        normalized_iteration = (
            None
            if iteration_fidelity is None
            else normalize_feature_dims([iteration_fidelity], input_dim, name="iteration_fidelity")[
                0
            ]
        )
        normalized_data = tuple(
            normalize_feature_dims(data_fidelities or (), input_dim, name="data_fidelities")
        )

        self.raw_input_dim = input_dim
        self.iteration_fidelity = normalized_iteration
        self.data_fidelities = normalized_data
        self.fidelity_dims = tuple(sorted(fidelity_dims))
        self.design_dims = tuple(dim for dim in range(input_dim) if dim not in self.fidelity_dims)
        if not self.design_dims:
            raise ValueError("Random projection requires at least one non-fidelity design feature.")

        design_X = train_X[..., list(self.design_dims)]
        reducer = RandomProjectionInputReducer(n_components=n_components, random_state=random_state)
        reduced_design_X = reducer.fit_transform(design_X, train_Y)
        encoded_train_X = self._join_design_and_fidelity(
            reduced_design_X,
            train_X[..., list(self.fidelity_dims)],
        )
        encoded_fidelity_dims = list(range(reduced_design_X.shape[-1], encoded_train_X.shape[-1]))
        raw_to_encoded = dict(zip(self.fidelity_dims, encoded_fidelity_dims, strict=True))
        encoded_iteration = (
            None if normalized_iteration is None else raw_to_encoded[normalized_iteration]
        )
        encoded_data = [raw_to_encoded[dim] for dim in normalized_data]

        super().__init__(
            train_X=encoded_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            iteration_fidelity=encoded_iteration,
            data_fidelities=encoded_data,
            linear_truncated=False,
            nu=nu,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.input_reducer = reducer
        self.reduced_design_dim = reduced_design_X.shape[-1]
        self.encoded_fidelity_dims = tuple(encoded_fidelity_dims)
        self._store_supervised_training_data(train_X, train_Y, train_Yvar)

    @staticmethod
    def _join_design_and_fidelity(design_X: Tensor, fidelity_X: Tensor) -> Tensor:
        return torch.cat((design_X, fidelity_X), dim=-1)

    def _encode_inputs(self, X: Tensor) -> Tensor:
        if X.shape[-1] != self.raw_input_dim:
            raise ValueError(
                f"Expected final input dimension {self.raw_input_dim}, got {X.shape[-1]}."
            )
        design_X = X[..., list(self.design_dims)]
        fidelity_X = X[..., list(self.fidelity_dims)]
        return self._join_design_and_fidelity(
            self.input_reducer.transform(design_X),
            fidelity_X,
        )

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the BoTorch multi-fidelity posterior from raw-space inputs."""
        return super().posterior(self._encode_inputs(X), *args, **kwargs)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Any:
        """Condition the encoded GP using observations supplied in raw coordinates."""
        return super().condition_on_observations(X=self._encode_inputs(X), Y=Y, **kwargs)

    def fantasize(self, X: Tensor, sampler: Any, **kwargs: Any) -> Any:
        """Construct fantasy models from candidate inputs supplied in raw coordinates."""
        encoded_X = self._encode_inputs(X)
        posterior = super().posterior(encoded_X, observation_noise=True)
        fantasy_Y = sampler(posterior)
        return super().condition_on_observations(X=encoded_X, Y=fantasy_Y, **kwargs)


class PLSMultiFidelityGP(SingleTaskMultiFidelityGP):
    """Multi-fidelity GP with PLS restricted to non-fidelity design features."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        n_components: int,
        train_Yvar: Tensor | None = None,
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        *,
        center: bool = True,
        linear_truncated: bool = False,
        nu: float = 2.5,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        input_dim = train_X.shape[-1]
        fidelity_dims: list[int] = []
        if iteration_fidelity is not None:
            fidelity_dims.extend(
                normalize_feature_dims([iteration_fidelity], input_dim, name="iteration_fidelity")
            )
        if data_fidelities is not None:
            fidelity_dims.extend(
                normalize_feature_dims(data_fidelities, input_dim, name="data_fidelities")
            )
        if not fidelity_dims:
            raise ValueError("At least one fidelity feature must be specified.")
        if len(set(fidelity_dims)) != len(fidelity_dims):
            raise ValueError("Fidelity dimensions must not contain duplicates.")
        if linear_truncated:
            raise ValueError(
                "PLSMultiFidelityGP requires linear_truncated=False so PCA can remain "
                "separate from fidelity covariance."
            )

        normalized_iteration = (
            None
            if iteration_fidelity is None
            else normalize_feature_dims([iteration_fidelity], input_dim, name="iteration_fidelity")[
                0
            ]
        )
        normalized_data = tuple(
            normalize_feature_dims(data_fidelities or (), input_dim, name="data_fidelities")
        )

        self.raw_input_dim = input_dim
        self.iteration_fidelity = normalized_iteration
        self.data_fidelities = normalized_data
        self.fidelity_dims = tuple(sorted(fidelity_dims))
        self.design_dims = tuple(dim for dim in range(input_dim) if dim not in self.fidelity_dims)
        if not self.design_dims:
            raise ValueError("PLS requires at least one non-fidelity design feature.")

        design_X = train_X[..., list(self.design_dims)]
        reducer = PLSInputReducer(n_components=n_components, center=center)
        reduced_design_X = reducer.fit_transform(design_X, train_Y)
        encoded_train_X = self._join_design_and_fidelity(
            reduced_design_X,
            train_X[..., list(self.fidelity_dims)],
        )
        encoded_fidelity_dims = list(range(reduced_design_X.shape[-1], encoded_train_X.shape[-1]))
        raw_to_encoded = dict(zip(self.fidelity_dims, encoded_fidelity_dims, strict=True))
        encoded_iteration = (
            None if normalized_iteration is None else raw_to_encoded[normalized_iteration]
        )
        encoded_data = [raw_to_encoded[dim] for dim in normalized_data]

        super().__init__(
            train_X=encoded_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            iteration_fidelity=encoded_iteration,
            data_fidelities=encoded_data,
            linear_truncated=False,
            nu=nu,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.input_reducer = reducer
        self.reduced_design_dim = reduced_design_X.shape[-1]
        self.encoded_fidelity_dims = tuple(encoded_fidelity_dims)
        self._store_supervised_training_data(train_X, train_Y, train_Yvar)

    @staticmethod
    def _join_design_and_fidelity(design_X: Tensor, fidelity_X: Tensor) -> Tensor:
        return torch.cat((design_X, fidelity_X), dim=-1)

    def _encode_inputs(self, X: Tensor) -> Tensor:
        if X.shape[-1] != self.raw_input_dim:
            raise ValueError(
                f"Expected final input dimension {self.raw_input_dim}, got {X.shape[-1]}."
            )
        design_X = X[..., list(self.design_dims)]
        fidelity_X = X[..., list(self.fidelity_dims)]
        return self._join_design_and_fidelity(
            self.input_reducer.transform(design_X),
            fidelity_X,
        )

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the BoTorch multi-fidelity posterior from raw-space inputs."""
        return super().posterior(self._encode_inputs(X), *args, **kwargs)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Any:
        """Condition the encoded GP using observations supplied in raw coordinates."""
        return super().condition_on_observations(X=self._encode_inputs(X), Y=Y, **kwargs)

    def fantasize(self, X: Tensor, sampler: Any, **kwargs: Any) -> Any:
        """Construct fantasy models from candidate inputs supplied in raw coordinates."""
        encoded_X = self._encode_inputs(X)
        posterior = super().posterior(encoded_X, observation_noise=True)
        fantasy_Y = sampler(posterior)
        return super().condition_on_observations(X=encoded_X, Y=fantasy_Y, **kwargs)
