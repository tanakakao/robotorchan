from __future__ import annotations

import pytest
import torch
from botorch.posteriors import Posterior
from torch import Tensor

from robotorchan.models.reduction import (
    InputReducer,
    OutputReducer,
    ReducerNotFittedError,
    ReductionMixin,
)


class FirstColumnsReducer(InputReducer):
    def __init__(self, n_components: int = 2) -> None:
        super().__init__()
        self.n_components = n_components
        self.last_fit_Y: Tensor | None = None

    def _fit_2d(self, X: Tensor, Y: Tensor | None) -> int:
        self.last_fit_Y = Y
        return min(self.n_components, X.shape[-1])

    def _transform_2d(self, X: Tensor) -> Tensor:
        return X[:, : self.output_dim]


class FirstColumnsOutputReducer(OutputReducer):
    def __init__(self, n_components: int = 2) -> None:
        super().__init__()
        self.n_components = n_components

    def _fit_2d(self, X: Tensor, Y: Tensor | None) -> int:
        return min(self.n_components, X.shape[-1])

    def _transform_2d(self, X: Tensor) -> Tensor:
        return X[:, : self.output_dim]

    def _inverse_transform_2d(self, Y: Tensor) -> Tensor:
        restored = Y.new_zeros(Y.shape[0], self.input_dim)
        restored[:, : self.output_dim] = Y
        return restored

    def restore_posterior(self, posterior: Posterior) -> Posterior:
        return posterior


class DummyReducedModel(ReductionMixin):
    def __init__(
        self,
        input_reducer: InputReducer | None = None,
        output_reducer: OutputReducer | None = None,
    ) -> None:
        self._set_reducers(input_reducer, output_reducer)


def test_reducer_requires_fit_before_transform() -> None:
    reducer = FirstColumnsReducer()

    with pytest.raises(ReducerNotFittedError):
        reducer.transform(torch.randn(3, 4))


def test_input_reducer_preserves_arbitrary_leading_dimensions() -> None:
    reducer = FirstColumnsReducer(n_components=2)
    train_X = torch.randn(6, 4)
    train_Y = torch.randn(6, 1)
    reducer.fit(train_X, train_Y)

    X = torch.randn(3, 5, 7, 4)
    transformed = reducer.transform(X)

    assert transformed.shape == torch.Size([3, 5, 7, 2])
    torch.testing.assert_close(transformed, X[..., :2])


def test_supervised_targets_are_available_during_input_fit() -> None:
    reducer = FirstColumnsReducer(n_components=2)
    train_X = torch.randn(6, 4)
    train_Y = torch.randn(6, 3)

    reducer.fit(train_X, train_Y)

    assert reducer.last_fit_Y is train_Y


def test_output_reducer_inverse_preserves_leading_dimensions() -> None:
    reducer = FirstColumnsOutputReducer(n_components=2)
    train_Y = torch.randn(8, 5)
    reducer.fit(train_Y)

    latent = torch.randn(2, 4, 2)
    restored = reducer.inverse_transform(latent)

    assert restored.shape == torch.Size([2, 4, 5])
    torch.testing.assert_close(restored[..., :2], latent)
    torch.testing.assert_close(restored[..., 2:], torch.zeros_like(restored[..., 2:]))


def test_reduction_mixin_routes_training_and_candidate_tensors() -> None:
    input_reducer = FirstColumnsReducer(n_components=2)
    output_reducer = FirstColumnsOutputReducer(n_components=1)
    model = DummyReducedModel(input_reducer=input_reducer, output_reducer=output_reducer)
    train_X = torch.randn(7, 4)
    train_Y = torch.randn(7, 3)

    reduced_X = model._fit_transform_inputs(train_X, train_Y)
    reduced_Y = model._fit_transform_outputs(train_X, train_Y)
    candidate_X = torch.randn(2, 5, 4)
    reduced_candidate_X = model._transform_inputs(candidate_X)

    assert reduced_X.shape == torch.Size([7, 2])
    assert reduced_Y.shape == torch.Size([7, 1])
    assert reduced_candidate_X.shape == torch.Size([2, 5, 2])


def test_reduction_mixin_is_identity_without_reducers() -> None:
    model = DummyReducedModel()
    train_X = torch.randn(5, 3)
    train_Y = torch.randn(5, 2)

    assert model._fit_transform_inputs(train_X, train_Y) is train_X
    assert model._fit_transform_outputs(train_X, train_Y) is train_Y
    assert model._transform_inputs(train_X) is train_X
