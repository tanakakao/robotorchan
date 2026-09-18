"""Tests for mixed-input reduction foundations."""

import pytest
import torch

from robotorchan.models.reduced.mixed import MixedInputLayout, MixedInputReducer
from robotorchan.reduction.input import PCAInputReducer


def test_mixed_input_layout_maps_categorical_dims_after_latent_block() -> None:
    layout = MixedInputLayout.from_cat_dims(input_dim=5, cat_dims=[1, 4], latent_dim=2)

    assert layout.cont_dims == (0, 2, 3)
    assert layout.cat_dims == (1, 4)
    assert layout.reduced_input_dim == 4
    assert layout.reduced_cat_dims == [2, 3]


def test_mixed_input_reducer_reduces_only_continuous_columns() -> None:
    train_X = torch.tensor(
        [
            [0.0, 0.0, 1.0, 2.0],
            [1.0, 1.0, 2.0, 3.0],
            [2.0, 0.0, 4.0, 5.0],
            [3.0, 1.0, 8.0, 7.0],
        ],
        dtype=torch.double,
    )
    train_Y = train_X[:, :1]
    reducer = MixedInputReducer(
        PCAInputReducer(n_components=2),
        input_dim=4,
        cat_dims=[1],
    )

    reduced_X = reducer.fit_transform(train_X, train_Y)

    assert reduced_X.shape == (4, 3)
    assert reducer.cat_dims == [2]
    torch.testing.assert_close(reduced_X[:, -1], train_X[:, 1])
    assert reducer.reducer.input_dim == 3


def test_mixed_input_reducer_preserves_categorical_values_on_transform() -> None:
    train_X = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 0.0, 4.0]],
        dtype=torch.double,
    )
    reducer = MixedInputReducer(
        PCAInputReducer(n_components=1),
        input_dim=3,
        cat_dims=[1],
    )
    reducer.fit(train_X)

    candidate_X = torch.tensor([[3.0, 2.0, 8.0]], dtype=torch.double)
    reduced_X = reducer.transform(candidate_X)

    assert reduced_X.shape == (1, 2)
    torch.testing.assert_close(reduced_X[:, -1], candidate_X[:, 1])


@pytest.mark.parametrize(
    ("cat_dims", "message"),
    [
        ([], "at least one"),
        ([1, 1], "duplicate"),
        ([3], "raw input dimension"),
        ([0, 1, 2], "continuous"),
    ],
)
def test_mixed_input_layout_rejects_invalid_cat_dims(
    cat_dims: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MixedInputLayout.from_cat_dims(input_dim=3, cat_dims=cat_dims, latent_dim=1)


def test_mixed_input_layout_rejects_wrong_candidate_dimension() -> None:
    layout = MixedInputLayout.from_cat_dims(input_dim=3, cat_dims=[1], latent_dim=1)

    with pytest.raises(ValueError, match="Expected final input dimension 3"):
        layout.continuous(torch.zeros(2, 4))


def test_mixed_input_layout_supports_negative_cat_dims() -> None:
    layout = MixedInputLayout.from_cat_dims(input_dim=5, cat_dims=[1, -1], latent_dim=2)

    assert layout.cat_dims == (1, 4)
    assert layout.cont_dims == (0, 2, 3)
    assert layout.reduced_cat_dims == [2, 3]
