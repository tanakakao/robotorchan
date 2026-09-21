import pytest
import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement

pytest.importorskip("sklearn")

from robotorchan.models import ExtraTreesSurrogate, RandomForestSurrogate
from robotorchan.optim import TreeEnsembleSearchStrategy


@pytest.mark.parametrize("model_class", [RandomForestSurrogate, ExtraTreesSurrogate])
def test_tree_surrogate_accepts_raw_categorical_dimensions(model_class) -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.2, 1.0], [0.4, 2.0], [0.6, 0.0], [0.8, 1.0], [1.0, 2.0]],
        dtype=torch.double,
    )
    train_Y = train_X[:, :1] + train_X[:, 1:] * 0.2
    model = model_class(train_X, train_Y, cat_dims=[-1], n_estimators=8, random_state=0)
    model.fit()

    assert model.cat_dims == (1,)
    posterior = model.posterior(torch.tensor([[0.3, 2.0]], dtype=torch.double))
    assert posterior.mean.shape == torch.Size([1, 1])


@pytest.mark.parametrize("model_class", [RandomForestSurrogate, ExtraTreesSurrogate])
def test_tree_surrogate_rejects_non_integer_category_labels(model_class) -> None:
    train_X = torch.tensor([[0.0, 0.0], [1.0, 1.5]], dtype=torch.double)
    train_Y = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    with pytest.raises(ValueError, match="integer-valued"):
        model_class(train_X, train_Y, cat_dims=[1])


def test_tree_search_samples_integer_and_categorical_dimensions() -> None:
    train_X = torch.tensor(
        [[0.0, 0.0, 10.0], [0.5, 1.0, 20.0], [1.0, 2.0, 30.0]],
        dtype=torch.double,
    )
    train_Y = train_X[:, :1]
    model = RandomForestSurrogate(train_X, train_Y, cat_dims=[2], n_estimators=8, random_state=1)
    model.fit()
    acqf = qExpectedImprovement(model=model, best_f=train_Y.max())
    strategy = TreeEnsembleSearchStrategy(
        torch.tensor([[0.0, 0.0, 10.0], [1.0, 4.0, 30.0]], dtype=torch.double),
        num_samples=64,
        seed=5,
        integer_dims=[1],
        categorical_values={2: [10.0, 20.0, 30.0]},
    )

    result = strategy.optimize(acqf, q=2)

    assert torch.equal(result.candidates[:, 1], result.candidates[:, 1].round())
    assert set(result.candidates[:, 2].tolist()) <= {10.0, 20.0, 30.0}
    assert result.metadata["integer_dims"] == (1,)
    assert result.metadata["categorical_dims"] == (2,)


def test_tree_search_rejects_overlapping_integer_and_categorical_dimensions() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    with pytest.raises(ValueError, match="both integer and categorical"):
        TreeEnsembleSearchStrategy(bounds, integer_dims=[1], categorical_values={1: [0.0, 1.0]})
