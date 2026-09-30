import pytest
import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement

pytest.importorskip("sklearn")

from robotorchan.models import ExtraTreesSurrogate, RandomForestSurrogate, SingleTaskGP
from robotorchan.optim import TreeEnsembleSearchStrategy


def make_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.0, 1.0, 16, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 6.0)
    return train_X, train_Y


@pytest.mark.parametrize("model_class", [RandomForestSurrogate, ExtraTreesSurrogate])
def test_tree_search_optimizes_mc_acquisition(model_class) -> None:
    train_X, train_Y = make_training_data()
    model = model_class(train_X, train_Y, n_estimators=12, random_state=0)
    model.fit()
    acqf = qExpectedImprovement(model=model, best_f=train_Y.max())
    strategy = TreeEnsembleSearchStrategy(
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        num_samples=128,
        seed=17,
    )

    result = strategy.optimize(acqf, q=2)

    assert result.candidates.shape == torch.Size([2, 1])
    assert result.acquisition_value is not None
    assert torch.isfinite(result.acquisition_value)
    assert result.metadata == {
        "num_samples": 128,
        "q": 2,
        "strategy": "tree_ensemble_random_search",
    }


def test_tree_search_is_reproducible() -> None:
    train_X, train_Y = make_training_data()
    model = RandomForestSurrogate(train_X, train_Y, n_estimators=8, random_state=1)
    model.fit()
    acqf = qExpectedImprovement(model=model, best_f=train_Y.max())
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    first = TreeEnsembleSearchStrategy(bounds, num_samples=64, seed=9).optimize(acqf)
    second = TreeEnsembleSearchStrategy(bounds, num_samples=64, seed=9).optimize(acqf)

    assert torch.equal(first.candidates, second.candidates)
    assert torch.equal(first.acquisition_value, second.acquisition_value)


def test_tree_search_rejects_analytic_acquisition() -> None:
    train_X, train_Y = make_training_data()
    model = ExtraTreesSurrogate(train_X, train_Y, n_estimators=8, random_state=2)
    model.fit()
    acqf = ExpectedImprovement(model=model, best_f=train_Y.max())
    strategy = TreeEnsembleSearchStrategy(
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        num_samples=16,
    )

    with pytest.raises(TypeError, match="Monte Carlo"):
        strategy.optimize(acqf)


def test_tree_search_rejects_gp_surrogate() -> None:
    train_X, train_Y = make_training_data()
    model = SingleTaskGP(train_X, train_Y)
    acqf = qExpectedImprovement(model=model, best_f=train_Y.max())
    strategy = TreeEnsembleSearchStrategy(
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        num_samples=16,
    )

    with pytest.raises(TypeError, match="non-GP"):
        strategy.optimize(acqf)


def test_tree_search_normalizes_negative_dims_and_respects_integer_bounds() -> None:
    bounds = torch.tensor([[0.2, 0.0], [2.8, 1.0]], dtype=torch.double)
    strategy = TreeEnsembleSearchStrategy(bounds, num_samples=128, seed=5, integer_dims=[-2])

    samples = strategy._sample_candidate_batches(q=1)

    assert strategy.integer_dims == (0,)
    assert torch.all(samples[..., 0] >= 1)
    assert torch.all(samples[..., 0] <= 2)
    assert torch.equal(samples[..., 0], samples[..., 0].round())


def test_tree_search_validates_categorical_domain() -> None:
    bounds = torch.tensor([[0.0], [2.0]], dtype=torch.double)

    with pytest.raises(ValueError, match="finite and unique"):
        TreeEnsembleSearchStrategy(bounds, categorical_values={0: [0.0, float("nan")]})
    with pytest.raises(ValueError, match="within bounds"):
        TreeEnsembleSearchStrategy(bounds, categorical_values={0: [0.0, 3.0]})
