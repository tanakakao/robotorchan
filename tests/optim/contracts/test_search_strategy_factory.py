"""Search strategy factory coverage and representative execution contracts."""

import inspect

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

import robotorchan.optim as optim
from robotorchan.optim import (
    SearchStrategy,
    SobolSearchStrategy,
    TuRBOState,
    create_search_strategy,
    list_search_strategies,
)


class _SumAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.sum(dim=(-1, -2))


def test_factory_covers_public_concrete_strategies() -> None:
    expected = set()
    for name in optim.__all__:
        value = getattr(optim, name)
        if (
            inspect.isclass(value)
            and issubclass(value, SearchStrategy)
            and not inspect.isabstract(value)
        ):
            expected.add(name)
    assert list_search_strategies() == tuple(sorted(expected))


def test_sobol_factory_matches_direct_strategy() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    strategy = create_search_strategy("SobolSearchStrategy", bounds, num_samples=32, seed=4)
    direct = SobolSearchStrategy(bounds, num_samples=32, seed=4)
    assert type(strategy) is SobolSearchStrategy
    actual = strategy.optimize(_SumAcquisition(), q=2)
    expected = direct.optimize(_SumAcquisition(), q=2)
    torch.testing.assert_close(actual.candidates, expected.candidates)
    torch.testing.assert_close(actual.acquisition_value, expected.acquisition_value)
    assert actual.candidates.shape == (2, 1)


def test_callable_and_invalid_inputs() -> None:
    bounds = torch.tensor([[0.0], [1.0]])
    assert isinstance(create_search_strategy(SobolSearchStrategy, bounds), SobolSearchStrategy)
    with pytest.raises(KeyError, match="Unknown search strategy"):
        create_search_strategy("TuRBOState", bounds)
    with pytest.raises(TypeError):
        create_search_strategy("SobolSearchStrategy", bounds, typo=True)
    with pytest.raises(TypeError, match="must return a SearchStrategy"):
        create_search_strategy(lambda: None)


def test_turbo_preserves_state_and_updates_after_proposal() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    state = TuRBOState(dim=1, batch_size=2)
    strategy = create_search_strategy(
        "TuRBOStrategy",
        bounds,
        center=torch.tensor([0.5], dtype=torch.double),
        state=state,
        optimizer="sobol",
        raw_samples=32,
        seed=4,
    )
    assert isinstance(strategy, optim.TuRBOStrategy)
    assert strategy.state is state
    result = strategy.optimize(_SumAcquisition(), q=2)
    assert result.candidates.shape == (2, 1)
    assert torch.all(result.candidates >= 0.1 - 1e-6)
    assert torch.all(result.candidates <= 0.9 + 1e-6)
    updated = strategy.update_state(result.candidates[:, 0], candidates=result.candidates)
    assert updated.best_value == float(result.candidates.max())
    assert strategy.state is updated


def test_mixed_constructor_preserves_assignments() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    assignments = [{1: 0.0}, {1: 2.0}]
    strategy = create_search_strategy("MixedSpaceStrategy", bounds, fixed_features_list=assignments)
    assert isinstance(strategy, optim.MixedSpaceStrategy)
    assert strategy.fixed_features_list == assignments
