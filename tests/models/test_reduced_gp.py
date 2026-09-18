from __future__ import annotations

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    PCAGP,
    PLSGP,
    OutputPCAGP,
    OutputPLSGP,
    RandomProjectionGP,
    ReducedGP,
)
from robotorchan.reduction import (
    OutputPCAReducer,
    OutputPLSReducer,
    PCAInputReducer,
    PLSInputReducer,
    RandomProjectionInputReducer,
)


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    train_X = torch.randn(18, 6, dtype=torch.double)
    train_Y = 1.5 * train_X[:, :1] - 0.7 * train_X[:, 1:2] + 0.2 * train_X[:, 2:3].square()
    return train_X, train_Y


def _multioutput_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    train_X = torch.randn(24, 5, dtype=torch.double)
    train_Y = torch.stack(
        [
            train_X[:, 0] + 0.2 * train_X[:, 1],
            train_X[:, 1] - train_X[:, 2],
            0.5 * train_X[:, 0] + train_X[:, 3],
            train_X[:, 2].square() + 0.1 * train_X[:, 4],
            train_X[:, 0] - train_X[:, 4],
            train_X[:, 3] + train_X[:, 4],
        ],
        dim=-1,
    )
    return train_X, train_Y


def test_reduced_gp_retains_original_raw_inputs_and_trains_in_latent_space() -> None:
    train_X, train_Y = _training_data()
    model = ReducedGP(
        train_X=train_X,
        train_Y=train_Y,
        input_reducer=PCAInputReducer(n_components=3),
    )

    assert model.original_input_dim == 6
    assert model.reduced_input_dim == 3
    assert model.train_inputs[0].shape == torch.Size([18, 3])
    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y)


def test_reduced_gp_posterior_accepts_original_q_batch_inputs() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X=train_X, train_Y=train_Y, n_components=3)
    model.eval()
    model.likelihood.eval()

    X = torch.randn(2, 4, 6, dtype=torch.double)
    posterior = model.posterior(X)

    assert posterior.mean.shape == torch.Size([2, 4, 1])
    assert posterior.variance.shape == torch.Size([2, 4, 1])


def test_reduced_gp_posterior_does_not_refit_input_reducer() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X=train_X, train_Y=train_Y, n_components=2)
    assert isinstance(model.input_reducer, PCAInputReducer)
    components_before = model.input_reducer.components.clone()
    mean_before = model.input_reducer.mean.clone()

    model.posterior(torch.randn(3, 6, dtype=torch.double))

    torch.testing.assert_close(model.input_reducer.components, components_before)
    torch.testing.assert_close(model.input_reducer.mean, mean_before)


def test_reduced_gp_accepts_already_reduced_inputs_for_botorch_internals() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X=train_X, train_Y=train_Y, n_components=2)
    latent_X = model.input_reducer.transform(torch.randn(5, 6, dtype=torch.double))

    posterior = model.posterior(latent_X)

    assert posterior.mean.shape == torch.Size([5, 1])


def test_reduced_gp_condition_on_observations_accepts_original_inputs() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X=train_X, train_Y=train_Y, n_components=2)
    model.eval()
    model.likelihood.eval()
    model.posterior(torch.randn(2, 6, dtype=torch.double))

    new_X = torch.randn(2, 6, dtype=torch.double)
    new_Y = torch.randn(2, 1, dtype=torch.double)
    fantasy_model = model.condition_on_observations(X=new_X, Y=new_Y)

    assert fantasy_model.train_inputs[0].shape[-1] == 2
    assert isinstance(fantasy_model.input_reducer, PCAInputReducer)
    posterior = fantasy_model.posterior(torch.randn(3, 6, dtype=torch.double))
    assert posterior.mean.shape == torch.Size([3, 1])


def test_reduced_gp_exposes_exact_mll_contract() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X=train_X, train_Y=train_Y, n_components=2)

    assert model.supports_mll is True
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_named_reduced_gp_wrappers_select_expected_reducers() -> None:
    train_X, train_Y = _training_data()

    pca_model = PCAGP(train_X=train_X, train_Y=train_Y, n_components=2)
    pls_model = PLSGP(train_X=train_X, train_Y=train_Y, n_components=2)
    rp_model = RandomProjectionGP(
        train_X=train_X,
        train_Y=train_Y,
        n_components=2,
        random_state=12,
    )

    assert isinstance(pca_model.input_reducer, PCAInputReducer)
    assert isinstance(pls_model.input_reducer, PLSInputReducer)
    assert isinstance(rp_model.input_reducer, RandomProjectionInputReducer)


def test_output_pca_gp_trains_latent_outputs_and_restores_public_posterior() -> None:
    train_X, train_Y = _multioutput_training_data()
    model = OutputPCAGP(train_X=train_X, train_Y=train_Y, n_components=3)
    model.eval()
    model.likelihood.eval()

    assert model.input_reducer is None
    assert isinstance(model.output_reducer, OutputPCAReducer)
    assert model.original_output_dim == 6
    assert model.reduced_output_dim == 3
    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y)

    posterior = model.posterior(torch.randn(2, 4, 5, dtype=torch.double))

    assert posterior.mean.shape == torch.Size([2, 4, 6])
    assert posterior.variance.shape == torch.Size([2, 4, 6])
    assert posterior.rsample(torch.Size([5])).shape == torch.Size([5, 2, 4, 6])


def test_output_pls_gp_uses_supervised_output_reducer() -> None:
    train_X, train_Y = _multioutput_training_data()
    model = OutputPLSGP(train_X=train_X, train_Y=train_Y, n_components=2)
    model.eval()
    model.likelihood.eval()

    assert isinstance(model.output_reducer, OutputPLSReducer)
    assert model.original_output_dim == 6
    assert model.reduced_output_dim == 2

    posterior = model.posterior(torch.randn(3, 5, dtype=torch.double))

    assert posterior.mean.shape == torch.Size([3, 6])


def test_output_reduced_gp_conditions_on_original_space_outcomes() -> None:
    train_X, train_Y = _multioutput_training_data()
    model = OutputPCAGP(train_X=train_X, train_Y=train_Y, n_components=3)
    model.eval()
    model.likelihood.eval()
    model.posterior(torch.randn(2, 5, dtype=torch.double))

    new_X = torch.randn(2, 5, dtype=torch.double)
    new_Y = torch.randn(2, 6, dtype=torch.double)
    fantasy_model = model.condition_on_observations(X=new_X, Y=new_Y)

    assert fantasy_model.train_inputs[0].shape[-1] == 5
    assert fantasy_model.train_targets.shape[-1] == train_X.shape[0] + new_X.shape[0]
    posterior = fantasy_model.posterior(torch.randn(3, 5, dtype=torch.double))
    assert posterior.mean.shape == torch.Size([3, 6])


def test_output_reduced_gp_rejects_explicit_noise_variances() -> None:
    train_X, train_Y = _multioutput_training_data()
    train_Yvar = torch.full_like(train_Y, 0.01)

    with pytest.raises(NotImplementedError, match="train_Yvar"):
        OutputPCAGP(
            train_X=train_X,
            train_Y=train_Y,
            n_components=3,
            train_Yvar=train_Yvar,
        )


def test_output_reduced_gp_guards_latent_incompatible_posterior_options() -> None:
    train_X, train_Y = _multioutput_training_data()
    model = OutputPCAGP(train_X=train_X, train_Y=train_Y, n_components=3)
    X = torch.randn(2, 5, dtype=torch.double)

    with pytest.raises(NotImplementedError, match="output_indices"):
        model.posterior(X, output_indices=[0])
    with pytest.raises(NotImplementedError, match="Tensor-valued observation_noise"):
        model.posterior(X, observation_noise=torch.ones(2, 6, dtype=torch.double))


def test_combined_reduced_gp_state_dict_round_trip_preserves_posterior() -> None:
    train_X, train_Y = _multioutput_training_data()
    input_reducer = PCAInputReducer(n_components=3)
    output_reducer = OutputPCAReducer(n_components=2)
    source = ReducedGP(
        train_X=train_X,
        train_Y=train_Y,
        input_reducer=input_reducer,
        output_reducer=output_reducer,
    )
    source.eval()
    source.likelihood.eval()

    restored = ReducedGP(
        train_X=train_X,
        train_Y=train_Y,
        input_reducer=PCAInputReducer(n_components=3),
        output_reducer=OutputPCAReducer(n_components=2),
    )
    restored.load_state_dict(source.state_dict())
    restored.eval()
    restored.likelihood.eval()

    X = torch.randn(4, 5, dtype=torch.double)
    source_posterior = source.posterior(X)
    restored_posterior = restored.posterior(X)

    assert restored.input_reducer is not None
    assert restored.output_reducer is not None
    assert restored.input_reducer.is_fitted is True
    assert restored.output_reducer.is_fitted is True
    torch.testing.assert_close(restored_posterior.mean, source_posterior.mean)
    torch.testing.assert_close(restored_posterior.variance, source_posterior.variance)
