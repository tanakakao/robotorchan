import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    MixedAdditiveMapSaasSingleTaskGP,
    MixedEnsembleMapSaasSingleTaskGP,
)


def _mixed_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    continuous = torch.tensor(
        [[0.10], [0.20], [0.35], [0.50], [0.65], [0.80], [0.90], [0.95]],
        dtype=torch.double,
    )
    category = torch.tensor(
        [[10.0], [20.0], [30.0], [10.0], [20.0], [30.0], [10.0], [20.0]],
        dtype=torch.double,
    )
    train_X = torch.cat([continuous, category], dim=-1)
    train_Y = torch.sin(continuous * 3.0) + 0.01 * category
    return train_X, train_Y


@pytest.mark.parametrize(
    "model_class",
    [MixedAdditiveMapSaasSingleTaskGP, MixedEnsembleMapSaasSingleTaskGP],
)
def test_mixed_map_saas_encodes_categories_internally(model_class: type) -> None:
    train_X, train_Y = _mixed_training_data()
    model = model_class(train_X=train_X, train_Y=train_Y, cat_dims=[-1], num_taus=2)

    assert model.cat_dims == (1,)
    assert model.raw_input_dim == 2
    assert model.encoded_input_dim == 4
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.category_values[0], torch.tensor([10.0, 20.0, 30.0], dtype=torch.double))

    encoded = model.input_transform.transform(train_X[:2])
    assert encoded.shape == torch.Size([2, 4])
    assert torch.equal(encoded[0], torch.tensor([0.10, 1.0, 0.0, 0.0], dtype=torch.double))


@pytest.mark.parametrize(
    "model_class",
    [MixedAdditiveMapSaasSingleTaskGP, MixedEnsembleMapSaasSingleTaskGP],
)
def test_mixed_map_saas_posterior_accepts_raw_mixed_inputs(model_class: type) -> None:
    train_X, train_Y = _mixed_training_data()
    model = model_class(train_X=train_X, train_Y=train_Y, cat_dims=[1], num_taus=2)
    model.eval()

    test_X = torch.tensor([[0.25, 20.0], [0.75, 30.0]], dtype=torch.double)
    posterior = model.posterior(test_X)

    assert posterior.mean.shape[-2:] == torch.Size([2, 1])
    assert posterior.variance.shape[-2:] == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


@pytest.mark.parametrize(
    "model_class",
    [MixedAdditiveMapSaasSingleTaskGP, MixedEnsembleMapSaasSingleTaskGP],
)
def test_mixed_map_saas_rejects_unseen_categories(model_class: type) -> None:
    train_X, train_Y = _mixed_training_data()
    model = model_class(train_X=train_X, train_Y=train_Y, cat_dims=[1], num_taus=2)
    model.eval()

    with pytest.raises(ValueError, match="unseen category"):
        model.posterior(torch.tensor([[0.5, 40.0]], dtype=torch.double))


@pytest.mark.parametrize(
    "model_class",
    [MixedAdditiveMapSaasSingleTaskGP, MixedEnsembleMapSaasSingleTaskGP],
)
def test_mixed_map_saas_exposes_exact_mll(model_class: type) -> None:
    train_X, train_Y = _mixed_training_data()
    model = model_class(train_X=train_X, train_Y=train_Y, cat_dims=[1], num_taus=2)

    assert model.supports_mll is True
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


@pytest.mark.parametrize(
    "model_class",
    [MixedAdditiveMapSaasSingleTaskGP, MixedEnsembleMapSaasSingleTaskGP],
)
def test_mixed_map_saas_rejects_custom_input_transform(model_class: type) -> None:
    train_X, train_Y = _mixed_training_data()

    with pytest.raises(ValueError, match="manages input_transform internally"):
        model_class(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[1],
            input_transform=torch.nn.Identity(),
            num_taus=2,
        )
