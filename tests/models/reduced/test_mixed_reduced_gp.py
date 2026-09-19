from __future__ import annotations

import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    MixedPCAGP,
    MixedPLSGP,
    MixedRandomProjectionGP,
)
from robotorchan.reduction import (
    PCAInputReducer,
    PLSInputReducer,
    RandomProjectionInputReducer,
)


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(31)
    continuous = torch.randn(24, 5, dtype=torch.double)
    category_a = torch.randint(0, 3, (24, 1)).to(dtype=torch.double)
    category_b = torch.randint(0, 2, (24, 1)).to(dtype=torch.double)
    train_X = torch.cat(
        (
            continuous[:, :2],
            category_a,
            continuous[:, 2:],
            category_b,
        ),
        dim=-1,
    )
    train_Y = continuous[:, :1] - 0.4 * continuous[:, 1:2] + 0.2 * category_a - 0.1 * category_b
    return train_X, train_Y


def _candidates(n: int = 4) -> torch.Tensor:
    continuous = torch.randn(n, 5, dtype=torch.double)
    category_a = torch.randint(0, 3, (n, 1)).to(dtype=torch.double)
    category_b = torch.randint(0, 2, (n, 1)).to(dtype=torch.double)
    return torch.cat(
        (
            continuous[:, :2],
            category_a,
            continuous[:, 2:],
            category_b,
        ),
        dim=-1,
    )


def test_mixed_pca_gp_reduces_only_continuous_inputs() -> None:
    train_X, train_Y = _training_data()
    model = MixedPCAGP(
        train_X=train_X,
        train_Y=train_Y,
        n_components=3,
        cat_dims=[2, 6],
    )

    assert isinstance(model.input_reducer, PCAInputReducer)
    assert model.original_input_dim == 7
    assert model.reduced_input_dim == 5
    assert model.original_cat_dims == [2, 6]
    assert model.reduced_cat_dims == [3, 4]
    assert model.train_inputs[0].shape == torch.Size([24, 5])
    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y)
    torch.testing.assert_close(model.train_inputs[0][..., 3], train_X[..., 2])
    torch.testing.assert_close(model.train_inputs[0][..., 4], train_X[..., 6])


def test_mixed_reduced_gp_posterior_accepts_original_q_batch_inputs() -> None:
    train_X, train_Y = _training_data()
    model = MixedPCAGP(
        train_X=train_X,
        train_Y=train_Y,
        n_components=2,
        cat_dims=[2, 6],
    )
    model.eval()
    model.likelihood.eval()

    X = _candidates(8).reshape(2, 4, 7)
    posterior = model.posterior(X)

    assert posterior.mean.shape == torch.Size([2, 4, 1])
    assert posterior.variance.shape == torch.Size([2, 4, 1])


def test_mixed_reduced_gp_accepts_already_reduced_inputs() -> None:
    train_X, train_Y = _training_data()
    model = MixedPCAGP(
        train_X=train_X,
        train_Y=train_Y,
        n_components=2,
        cat_dims=[2, 6],
    )
    reduced_X = model._transform_original_inputs(_candidates())

    posterior = model.posterior(reduced_X)

    assert posterior.mean.shape == torch.Size([4, 1])


def test_mixed_reduced_gp_condition_on_observations_uses_original_space() -> None:
    train_X, train_Y = _training_data()
    model = MixedPCAGP(
        train_X=train_X,
        train_Y=train_Y,
        n_components=2,
        cat_dims=[2, 6],
    )
    model.eval()
    model.likelihood.eval()
    model.posterior(_candidates(2))

    new_X = _candidates(2)
    new_Y = torch.randn(2, 1, dtype=torch.double)
    fantasy_model = model.condition_on_observations(new_X, new_Y)

    assert fantasy_model.train_inputs[0].shape[-1] == 4
    posterior = fantasy_model.posterior(_candidates(3))
    assert posterior.mean.shape == torch.Size([3, 1])


def test_mixed_reduced_gp_exposes_exact_mll_contract() -> None:
    train_X, train_Y = _training_data()
    model = MixedPCAGP(
        train_X=train_X,
        train_Y=train_Y,
        n_components=2,
        cat_dims=[2, 6],
    )

    assert model.supports_mll is True
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_named_mixed_reduced_wrappers_select_expected_reducers() -> None:
    train_X, train_Y = _training_data()

    pca_model = MixedPCAGP(train_X, train_Y, 2, [2, 6])
    pls_model = MixedPLSGP(train_X, train_Y, 2, [2, 6])
    rp_model = MixedRandomProjectionGP(
        train_X,
        train_Y,
        2,
        [2, 6],
        random_state=19,
    )

    assert isinstance(pca_model.input_reducer, PCAInputReducer)
    assert isinstance(pls_model.input_reducer, PLSInputReducer)
    assert isinstance(rp_model.input_reducer, RandomProjectionInputReducer)


def test_mixed_reduced_gp_state_dict_round_trip_preserves_posterior() -> None:
    train_X, train_Y = _training_data()
    source = MixedPCAGP(train_X, train_Y, 2, [2, 6])
    restored = MixedPCAGP(train_X, train_Y, 2, [2, 6])
    restored.load_state_dict(source.state_dict())

    source.eval()
    source.likelihood.eval()
    restored.eval()
    restored.likelihood.eval()

    X = _candidates(4)
    source_posterior = source.posterior(X)
    restored_posterior = restored.posterior(X)

    torch.testing.assert_close(restored_posterior.mean, source_posterior.mean)
    torch.testing.assert_close(restored_posterior.variance, source_posterior.variance)


def test_mixed_reduced_gp_normalizes_negative_cat_dims_in_raw_space() -> None:
    train_X, train_Y = _training_data()
    model = MixedPCAGP(
        train_X=train_X,
        train_Y=train_Y,
        n_components=2,
        cat_dims=[2, -1],
    )

    assert model.original_cat_dims == [2, 6]
    assert model.reduced_cat_dims == [2, 3]
    torch.testing.assert_close(model.raw_train_X, train_X)


def test_classical_mixed_wrappers_are_colocated_with_standard_families() -> None:
    from robotorchan.models.reduced import base as reduced_base

    assert MixedPCAGP.__module__ == reduced_base.__name__
    assert MixedPLSGP.__module__ == reduced_base.__name__
    assert MixedRandomProjectionGP.__module__ == reduced_base.__name__


def test_classical_mixed_reducer_never_sees_category_codes() -> None:
    train_X, train_Y = _training_data()
    model = MixedPCAGP(train_X, train_Y, 2, [2, -1])

    assert model.input_reducer.input_dim == 5
    assert model.original_cat_dims == [2, 6]
    assert model.reduced_cat_dims == [2, 3]
    torch.testing.assert_close(model.train_inputs[0][..., 2], train_X[..., 2])
    torch.testing.assert_close(model.train_inputs[0][..., 3], train_X[..., 6])
