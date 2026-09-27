"""Tests for latent reconstruction capabilities."""

import pytest
import torch

from robotorchan.optim.latent import PCAReconstruction, RandomProjectionReconstruction
from robotorchan.reduction import (
    PCAInputReducer,
    PLSInputReducer,
    RandomProjectionInputReducer,
    ReducerNotFittedError,
)


def test_pca_reconstruction_round_trips_latent_coordinates() -> None:
    torch.manual_seed(11)
    train_X = torch.randn(12, 5, dtype=torch.double)
    reducer = PCAInputReducer(n_components=3).fit(train_X)
    reconstruction = PCAReconstruction(reducer)
    Z = torch.randn(2, 4, 3, dtype=torch.double)

    restored = reconstruction.reconstruct(Z)

    assert restored.shape == torch.Size([2, 4, 5])
    torch.testing.assert_close(reconstruction.transform(restored), Z)


def test_pca_reconstruction_respects_center_false() -> None:
    torch.manual_seed(12)
    train_X = torch.randn(10, 4, dtype=torch.double)
    reducer = PCAInputReducer(n_components=2, center=False).fit(train_X)
    reconstruction = PCAReconstruction(reducer)
    Z = torch.randn(5, 2, dtype=torch.double)

    restored = reconstruction.reconstruct(Z)

    torch.testing.assert_close(reconstruction.transform(restored), Z)


def test_random_projection_reconstruction_round_trips_latent_coordinates() -> None:
    train_X = torch.randn(9, 6, dtype=torch.double)
    reducer = RandomProjectionInputReducer(n_components=3, random_state=7).fit(train_X)
    reconstruction = RandomProjectionReconstruction(reducer)
    Z = torch.randn(2, 5, 3, dtype=torch.double)

    restored = reconstruction.reconstruct(Z)

    assert restored.shape == torch.Size([2, 5, 6])
    torch.testing.assert_close(reconstruction.transform(restored), Z, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("kind", ["pca", "random"])
def test_transformed_outer_bounds_contain_transformed_box_corners(kind: str) -> None:
    dtype = torch.double
    bounds = torch.tensor([[-1.0, 0.0, 2.0], [2.0, 3.0, 5.0]], dtype=dtype)
    train_X = torch.tensor(
        [
            [-1.0, 0.0, 2.0],
            [2.0, 3.0, 5.0],
            [0.0, 1.0, 3.0],
            [1.0, 2.0, 4.0],
        ],
        dtype=dtype,
    )
    if kind == "pca":
        reducer = PCAInputReducer(n_components=2).fit(train_X)
        reconstruction = PCAReconstruction(reducer)
    else:
        reducer = RandomProjectionInputReducer(n_components=2, random_state=3).fit(train_X)
        reconstruction = RandomProjectionReconstruction(reducer)

    latent_bounds = reconstruction.transform_bounds(bounds)
    corners = torch.cartesian_prod(*[bounds[:, index] for index in range(3)])
    transformed = reconstruction.transform(corners)

    assert latent_bounds.shape == torch.Size([2, 2])
    assert torch.all(transformed >= latent_bounds[0] - 1e-12)
    assert torch.all(transformed <= latent_bounds[1] + 1e-12)
    torch.testing.assert_close(transformed.min(dim=0).values, latent_bounds[0])
    torch.testing.assert_close(transformed.max(dim=0).values, latent_bounds[1])


def test_reconstruction_requires_fitted_reducer() -> None:
    with pytest.raises(ReducerNotFittedError, match="must be fitted"):
        PCAReconstruction(PCAInputReducer(n_components=2))


def test_reconstruction_rejects_wrong_reducer_type() -> None:
    train_X = torch.randn(8, 4)
    train_Y = torch.randn(8, 1)
    pls = PLSInputReducer(n_components=1).fit(train_X, train_Y)

    with pytest.raises(TypeError, match="PCAInputReducer"):
        PCAReconstruction(pls)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RandomProjectionInputReducer"):
        RandomProjectionReconstruction(pls)  # type: ignore[arg-type]


def test_reconstruction_validates_latent_shape_and_bounds() -> None:
    train_X = torch.randn(8, 4)
    reconstruction = PCAReconstruction(PCAInputReducer(n_components=2).fit(train_X))

    with pytest.raises(ValueError, match="latent final dimension"):
        reconstruction.reconstruct(torch.randn(3, 3))
    with pytest.raises(ValueError, match="bounds must have shape"):
        reconstruction.transform_bounds(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="strictly below"):
        reconstruction.transform_bounds(torch.zeros(2, 4))
