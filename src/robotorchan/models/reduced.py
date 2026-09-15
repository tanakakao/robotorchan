"""GP wrappers with optional input and output dimensionality reduction."""

from __future__ import annotations

from typing import Any

from botorch.models import SingleTaskGP as BoTorchSingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.module import Module
from torch import Tensor

from robotorchan.models.base import ExactGPModelMixin
from robotorchan.models.neural_reduction import AutoEncoderInputReducer, VAEInputReducer
from robotorchan.models.output_reduction import OutputPCAReducer, OutputPLSReducer
from robotorchan.models.reduction import (
    InputReducer,
    OutputReducer,
    PCAInputReducer,
    PLSInputReducer,
    RandomProjectionInputReducer,
    ReductionMixin,
)


class ReducedGP(ReductionMixin, ExactGPModelMixin, BoTorchSingleTaskGP):
    """Single-task GP with optional frozen input and output reduction."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        input_reducer: InputReducer | None = None,
        output_reducer: OutputReducer | None = None,
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        covar_module: Module | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        if output_reducer is not None and train_Yvar is not None:
            raise NotImplementedError("Explicit train_Yvar is not supported with output reduction.")
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()
        self._original_input_dim_value = train_X.shape[-1]
        self._original_output_dim_value = train_Y.shape[-1]
        if input_reducer is None:
            reduced_train_X = train_X
        elif input_reducer.is_fitted:
            reduced_train_X = input_reducer.transform(train_X)
        else:
            reduced_train_X = input_reducer.fit_transform(train_X, train_Y)
        if output_reducer is None:
            reduced_train_Y = train_Y
        elif output_reducer.is_fitted:
            reduced_train_Y = output_reducer.transform(train_Y)
        else:
            reduced_train_Y = output_reducer.fit_transform(train_Y, train_X)
        super().__init__(train_X=reduced_train_X, train_Y=reduced_train_Y, train_Yvar=train_Yvar, likelihood=likelihood, covar_module=covar_module, mean_module=mean_module, outcome_transform=outcome_transform, input_transform=input_transform)
        self._set_reducers(input_reducer=input_reducer, output_reducer=output_reducer)
        self._store_supervised_training_data(train_X=raw_train_X, train_Y=raw_train_Y, train_Yvar=raw_train_Yvar)

    def load_state_dict(self, state_dict: dict[str, Tensor], strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        train_X = self._prepare_inputs(self.raw_train_X)
        if hasattr(self, "input_transform"):
            train_X = self.transform_inputs(train_X)
        self.set_train_data(inputs=train_X, targets=self.train_targets, strict=False)
        return result

    @property
    def original_input_dim(self) -> int:
        return self._original_input_dim_value

    @property
    def reduced_input_dim(self) -> int:
        return self.original_input_dim if self.input_reducer is None else self.input_reducer.output_dim

    @property
    def original_output_dim(self) -> int:
        return self._original_output_dim_value

    @property
    def reduced_output_dim(self) -> int:
        return self.original_output_dim if self.output_reducer is None else self.output_reducer.output_dim

    def _prepare_inputs(self, X: Tensor) -> Tensor:
        if self.input_reducer is None:
            if X.shape[-1] != self.original_input_dim:
                raise ValueError(f"Expected final input dimension {self.original_input_dim}, got {X.shape[-1]}.")
            return X
        if X.shape[-1] == self.original_input_dim:
            return self._transform_inputs(X)
        if X.shape[-1] == self.reduced_input_dim:
            return X
        raise ValueError(f"Expected final input dimension {self.original_input_dim} (original) or {self.reduced_input_dim} (reduced), got {X.shape[-1]}.")

    def _prepare_outputs(self, Y: Tensor) -> Tensor:
        if self.output_reducer is None:
            if Y.shape[-1] != self.original_output_dim:
                raise ValueError(f"Expected final output dimension {self.original_output_dim}, got {Y.shape[-1]}.")
            return Y
        if Y.shape[-1] == self.original_output_dim:
            return self.output_reducer.transform(Y)
        if Y.shape[-1] == self.reduced_output_dim:
            return Y
        raise ValueError(f"Expected final output dimension {self.original_output_dim} (original) or {self.reduced_output_dim} (reduced), got {Y.shape[-1]}.")

    def posterior(self, X: Tensor, output_indices: list[int] | None = None, observation_noise: bool | Tensor = False, posterior_transform: Any | None = None):
        if self.output_reducer is not None:
            if output_indices is not None:
                raise NotImplementedError("output_indices is not yet supported with output reduction.")
            if posterior_transform is not None:
                raise NotImplementedError("posterior_transform is not yet supported with output reduction.")
            if isinstance(observation_noise, Tensor):
                raise NotImplementedError("Tensor-valued observation_noise is not supported with output reduction.")
        posterior = super().posterior(self._prepare_inputs(X), output_indices=output_indices, observation_noise=observation_noise, posterior_transform=posterior_transform)
        return self._restore_output_posterior(posterior)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        if self.output_reducer is not None and kwargs.get("noise") is not None:
            raise NotImplementedError("Explicit observation noise is not supported with output reduction.")
        return super().condition_on_observations(X=self._prepare_inputs(X), Y=self._prepare_outputs(Y), **kwargs)


class PCAGP(ReducedGP):
    def __init__(self, train_X: Tensor, train_Y: Tensor, n_components: int, *, center: bool = True, **kwargs: Any) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, input_reducer=PCAInputReducer(n_components=n_components, center=center), **kwargs)


class PLSGP(ReducedGP):
    def __init__(self, train_X: Tensor, train_Y: Tensor, n_components: int, *, center: bool = True, eps: float = 1e-12, **kwargs: Any) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, input_reducer=PLSInputReducer(n_components=n_components, center=center, eps=eps), **kwargs)


class RandomProjectionGP(ReducedGP):
    def __init__(self, train_X: Tensor, train_Y: Tensor, n_components: int, *, random_state: int = 0, **kwargs: Any) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, input_reducer=RandomProjectionInputReducer(n_components=n_components, random_state=random_state), **kwargs)


class AutoEncoderGP(ReducedGP):
    def __init__(self, train_X: Tensor, train_Y: Tensor, latent_dim: int, *, hidden_dims: tuple[int, ...] = (64, 32), activation: str = "gelu", epochs: int = 200, learning_rate: float = 1e-3, weight_decay: float = 0.0, batch_size: int | None = None, standardize: bool = True, eps: float = 1e-8, random_state: int = 0, **kwargs: Any) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, input_reducer=AutoEncoderInputReducer(latent_dim=latent_dim, hidden_dims=hidden_dims, activation=activation, epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size, standardize=standardize, eps=eps, random_state=random_state), **kwargs)


class VAEGP(ReducedGP):
    """Single-task GP using the frozen posterior-mean representation of a VAE."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, latent_dim: int, *, hidden_dims: tuple[int, ...] = (64, 32), activation: str = "gelu", epochs: int = 200, learning_rate: float = 1e-3, weight_decay: float = 0.0, batch_size: int | None = None, standardize: bool = True, eps: float = 1e-8, random_state: int = 0, beta: float = 1.0, **kwargs: Any) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, input_reducer=VAEInputReducer(latent_dim=latent_dim, hidden_dims=hidden_dims, activation=activation, epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size, standardize=standardize, eps=eps, random_state=random_state, beta=beta), **kwargs)


class OutputPCAGP(ReducedGP):
    def __init__(self, train_X: Tensor, train_Y: Tensor, n_components: int, *, center: bool = True, **kwargs: Any) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, output_reducer=OutputPCAReducer(n_components=n_components, center=center), **kwargs)


class OutputPLSGP(ReducedGP):
    def __init__(self, train_X: Tensor, train_Y: Tensor, n_components: int, *, center: bool = True, eps: float = 1e-12, **kwargs: Any) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, output_reducer=OutputPLSReducer(n_components=n_components, center=center, eps=eps), **kwargs)
