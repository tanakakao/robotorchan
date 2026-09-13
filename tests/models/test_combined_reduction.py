from __future__ import annotations

import torch

from robotorchan.models import ReducedGP
from robotorchan.models.output_reduction import OutputPCAReducer, OutputPLSReducer
from robotorchan.models.reduction import PCAInputReducer, PLSInputReducer


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(23)
    train_X = torch.randn(28, 8, dtype=torch.double)
    train_Y = torch.stack(
        [
            train_X[:, 0] + 0.4 * train_X[:, 1],
            train_X[:, 1] - 0.3 * train_X[:, 2],
            0.5 * train_X[:, 0] + train_X[:, 3],
            train_X[:, 2].square() + 0.2 * train_X[:, 4],
            train_X[:, 0] - train_X[:, 5],
            train_X[:, 3] + 0.5 * train_X[:, 6],
        ],
        dim=-1,
    )
    return train_X, train_Y


def test_combined_pca_reduction_trains_in_both_latent_spaces() -> None:
    train_X, train_Y = _training_data()
    model = ReducedGP(
        train_X=train_X,
        train_Y=train_Y,
        input_reducer=PCAInputReducer(n_components=4),
        output_reducer=OutputPCAReducer(n_components=3),
    )

    assert model.original_input_dim == 8
    assert model.reduced_input_dim == 4
    assert model.original_output_dim == 6
    assert model.reduced_output_dim == 3
    assert model.train_inputs[0].shape == torch.Size([28, 4])
    assert model.train_targets.shape == torch.Size([3, 28])
    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y)


def test_combined_reduction_restores_original_output_for_q_batch() -> None:
    train_X, train_Y = _training_data()
    model = ReducedGP(
        train_X=train_X,
        train_Y=train_Y,
        input_reducer=PCAInputReducer(n_components=4),
        output_reducer=OutputPCAReducer(n_components=3),
    )
    model.eval()
    model.likelihood.eval()

    X = torch.randn(2, 5, 8, dtype=torch.double)
    posterior = model.posterior(X)

    assert posterior.mean.shape == torch.Size([2, 5, 6])
    assert posterior.variance.shape == torch.Size([2, 5, 6])
    assert posterior.rsample(torch.Size([7])).shape == torch.Size([7, 2, 5, 6])


def test_combined_reduction_conditions_in_original_spaces_without_refitting() -> None:
    train_X, train_Y = _training_data()
    input_reducer = PCAInputReducer(n_components=4)
    output_reducer = OutputPCAReducer(n_components=3)
    model = ReducedGP(
        train_X=train_X,
        train_Y=train_Y,
        input_reducer=input_reducer,
        output_reducer=output_reducer,
    )
    model.eval()
    model.likelihood.eval()
    model.posterior(torch.randn(2, 8, dtype=torch.double))

    input_components_before = input_reducer.components.clone()
    output_components_before = output_reducer.components.clone()
    output_mean_before = output_reducer.mean.clone()

    new_X = torch.randn(3, 8, dtype=torch.double)
    new_Y = torch.randn(3, 6, dtype=torch.double)
    fantasy_model = model.condition_on_observations(X=new_X, Y=new_Y)

    assert fantasy_model.train_inputs[0].shape == torch.Size([31, 4])
    assert fantasy_model.train_targets.shape == torch.Size([3, 31])
    assert isinstance(fantasy_model.input_reducer, PCAInputReducer)
    assert isinstance(fantasy_model.output_reducer, OutputPCAReducer)
    torch.testing.assert_close(input_reducer.components, input_components_before)
    torch.testing.assert_close(output_reducer.components, output_components_before)
    torch.testing.assert_close(output_reducer.mean, output_mean_before)

    posterior = fantasy_model.posterior(torch.randn(4, 8, dtype=torch.double))
    assert posterior.mean.shape == torch.Size([4, 6])


def test_combined_supervised_pls_reducers_are_composable() -> None:
    train_X, train_Y = _training_data()
    model = ReducedGP(
        train_X=train_X,
        train_Y=train_Y,
        input_reducer=PLSInputReducer(n_components=4),
        output_reducer=OutputPLSReducer(n_components=3),
    )
    model.eval()
    model.likelihood.eval()

    assert isinstance(model.input_reducer, PLSInputReducer)
    assert isinstance(model.output_reducer, OutputPLSReducer)
    assert model.train_inputs[0].shape == torch.Size([28, 4])
    assert model.train_targets.shape == torch.Size([3, 28])

    posterior = model.posterior(torch.randn(3, 8, dtype=torch.double))

    assert posterior.mean.shape == torch.Size([3, 6])
    assert posterior.variance.shape == torch.Size([3, 6])
