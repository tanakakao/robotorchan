"""Tests for the stateful TuRBO acquisition search strategy."""

from types import SimpleNamespace

import pytest
import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models import SingleTaskGP
from robotorchan.optim import (
    CandidateConstraints,
    TuRBOState,
    TuRBOStrategy,
    generate_turbo_restart_center,
    generate_turbo_thompson_choices,
    restart_turbo_state,
    turbo_dimension_weights_from_model,
    turbo_multifidelity_trust_region_bounds,
    turbo_trust_region_bounds,
    update_turbo_state,
)


def _problem(input_dim: int = 4):
    torch.manual_seed(19)
    train_X = torch.rand(10, input_dim, dtype=torch.double)
    train_Y = -((train_X[:, :2] - 0.7) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )
    return train_X, train_Y, bounds


def _center(train_X: torch.Tensor, train_Y: torch.Tensor) -> torch.Tensor:
    return train_X[train_Y.squeeze(-1).argmax()]


def test_state_derives_failure_tolerance_from_dimension_and_batch_size() -> None:
    state = TuRBOState(dim=20, batch_size=4)
    assert state.failure_tolerance == 5

    state = TuRBOState(dim=5, batch_size=2)
    assert state.failure_tolerance == 3


def test_explicit_failure_tolerance_overrides_derived_value() -> None:
    state = TuRBOState(dim=20, batch_size=4, failure_tolerance=7)
    assert state.failure_tolerance == 7


def test_state_validates_dimension_and_batch_size() -> None:
    with pytest.raises(ValueError, match="dim must be at least 1"):
        TuRBOState(dim=0)
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        TuRBOState(batch_size=0)


def test_state_expands_after_success_tolerance() -> None:
    state = TuRBOState(dim=4, length=0.4, success_tolerance=2, best_value=0.0)
    state = update_turbo_state(state, torch.tensor([1.0]))
    assert state.length == pytest.approx(0.4)
    assert state.success_counter == 1

    state = update_turbo_state(state, torch.tensor([2.0]))
    assert state.length == pytest.approx(0.8)
    assert state.success_counter == 0
    assert state.failure_counter == 0
    assert state.best_value == pytest.approx(2.0)


def test_state_shrinks_and_triggers_restart() -> None:
    state = TuRBOState(dim=4, length=0.2, length_min=0.15, failure_tolerance=2, best_value=1.0)
    state = update_turbo_state(state, torch.tensor([0.0]))
    assert state.failure_counter == 1

    state = update_turbo_state(state, torch.tensor([0.0]))
    assert state.length == pytest.approx(0.1)
    assert state.failure_counter == 0
    assert state.restart_triggered


def test_small_improvement_below_tolerance_counts_as_failure() -> None:
    state = TuRBOState(dim=4, best_value=100.0, failure_tolerance=4)
    next_state = update_turbo_state(
        state,
        torch.tensor([100.05]),
        relative_improvement=1e-3,
    )

    assert next_state.success_counter == 0
    assert next_state.failure_counter == 1
    assert next_state.best_value == pytest.approx(100.05)


def test_strategy_center_uses_same_improvement_tolerance_as_state() -> None:
    _, _, bounds = _problem()
    initial = torch.full((4,), 0.25, dtype=torch.double)
    candidate = torch.full((1, 4), 0.75, dtype=torch.double)
    strategy = TuRBOStrategy(
        bounds,
        center=initial,
        state=TuRBOState(dim=4, best_value=100.0),
    )

    next_state = strategy.update_state(
        torch.tensor([100.05]),
        candidates=candidate,
        relative_improvement=1e-3,
    )

    torch.testing.assert_close(strategy.center, initial)
    assert next_state.failure_counter == 1


def test_terminal_state_below_minimum_requires_restart_flag() -> None:
    state = TuRBOState(length=0.1, length_min=0.15, restart_triggered=True)
    assert state.restart_triggered
    with pytest.raises(ValueError, match="requires restart_triggered"):
        TuRBOState(length=0.1, length_min=0.15)


def test_trust_region_is_clipped_to_public_bounds() -> None:
    _, _, bounds = _problem()
    center = torch.tensor([0.1, 0.9, 0.5, 0.5], dtype=torch.double)
    strategy = TuRBOStrategy(bounds, center=center, state=TuRBOState(dim=4, length=0.8))

    trust_bounds = strategy.trust_region_bounds()

    assert torch.all(bounds[0] <= trust_bounds[0])
    assert torch.all(bounds[1] >= trust_bounds[1])
    torch.testing.assert_close(trust_bounds[0, :2], torch.tensor([0.0, 0.5], dtype=torch.double))
    torch.testing.assert_close(trust_bounds[1, :2], torch.tensor([0.5, 1.0], dtype=torch.double))


def test_geometry_scales_ard_weights_to_geometric_mean_one() -> None:
    bounds = torch.stack(
        [
            torch.zeros(3, dtype=torch.double),
            torch.ones(3, dtype=torch.double),
        ]
    )
    center = torch.full((3,), 0.5, dtype=torch.double)
    raw_weights = torch.tensor([0.25, 1.0, 4.0], dtype=torch.double)

    trust_bounds = turbo_trust_region_bounds(
        center,
        bounds,
        length=0.2,
        dimension_weights=raw_weights,
    )

    half_widths = (trust_bounds[1] - trust_bounds[0]) / 2.0
    torch.testing.assert_close(half_widths, 0.1 * raw_weights)


def test_geometry_is_scale_invariant_for_dimension_weights() -> None:
    bounds = torch.tensor(
        [[-2.0, 10.0, 100.0], [2.0, 30.0, 200.0]],
        dtype=torch.double,
    )
    center = bounds.mean(dim=0)
    weights = torch.tensor([0.5, 1.0, 2.0], dtype=torch.double)

    first = turbo_trust_region_bounds(
        center,
        bounds,
        length=0.2,
        dimension_weights=weights,
    )
    second = turbo_trust_region_bounds(
        center,
        bounds,
        length=0.2,
        dimension_weights=10.0 * weights,
    )

    torch.testing.assert_close(first, second)


def test_geometry_uses_global_ranges_and_clips_at_boundaries() -> None:
    bounds = torch.tensor(
        [[0.0, -10.0], [2.0, 10.0]],
        dtype=torch.double,
    )
    center = torch.tensor([0.1, 9.0], dtype=torch.double)

    trust_bounds = turbo_trust_region_bounds(center, bounds, length=0.5)

    torch.testing.assert_close(
        trust_bounds,
        torch.tensor([[0.0, 4.0], [0.6, 10.0]], dtype=torch.double),
    )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (torch.tensor([1.0, 2.0]), "shape"),
        (torch.tensor([1.0, 0.0, 2.0]), "strictly positive"),
        (torch.tensor([1.0, float("inf"), 2.0]), "finite"),
    ],
)
def test_geometry_rejects_invalid_dimension_weights(
    weights: torch.Tensor,
    message: str,
) -> None:
    bounds = torch.stack(
        [
            torch.zeros(3, dtype=torch.double),
            torch.ones(3, dtype=torch.double),
        ]
    )
    center = torch.full((3,), 0.5, dtype=torch.double)

    with pytest.raises(ValueError, match=message):
        turbo_trust_region_bounds(
            center,
            bounds,
            length=0.4,
            dimension_weights=weights,
        )


def test_multifidelity_geometry_keeps_fidelity_bounds_global() -> None:
    bounds = torch.tensor(
        [[0.0, 0.0, 0.25], [1.0, 1.0, 1.0]],
        dtype=torch.double,
    )
    center = torch.tensor([0.5, 0.5, 0.6], dtype=torch.double)

    trust_bounds = turbo_multifidelity_trust_region_bounds(
        center,
        bounds,
        fidelity_dims=[-1],
        length=0.2,
    )

    torch.testing.assert_close(trust_bounds[:, 2], bounds[:, 2])
    torch.testing.assert_close(
        trust_bounds[:, :2],
        torch.tensor([[0.4, 0.4], [0.6, 0.6]], dtype=torch.double),
    )


def test_multifidelity_geometry_rejects_invalid_fidelity_dims() -> None:
    bounds = torch.stack([torch.zeros(3), torch.ones(3)])
    center = torch.full((3,), 0.5)

    with pytest.raises(ValueError, match="duplicates"):
        turbo_multifidelity_trust_region_bounds(
            center,
            bounds,
            fidelity_dims=[1, -2],
            length=0.2,
        )
    with pytest.raises(ValueError, match="non-fidelity design dimension"):
        turbo_multifidelity_trust_region_bounds(
            center,
            bounds,
            fidelity_dims=[0, 1, 2],
            length=0.2,
        )


def test_strategy_uses_dimension_weights_for_trust_region() -> None:
    _, _, bounds = _problem(input_dim=3)
    center = torch.full((3,), 0.5, dtype=torch.double)
    strategy = TuRBOStrategy(
        bounds,
        center=center,
        state=TuRBOState(dim=3, length=0.2),
        dimension_weights=torch.tensor([0.25, 1.0, 4.0]),
    )

    trust_bounds = strategy.trust_region_bounds()

    half_widths = (trust_bounds[1] - trust_bounds[0]) / 2.0
    torch.testing.assert_close(
        half_widths,
        torch.tensor([0.025, 0.1, 0.4], dtype=torch.double),
    )


def test_optimize_uses_explicit_incumbent_as_center() -> None:
    train_X, train_Y, bounds = _problem()
    model = SingleTaskGP(train_X, train_Y)
    acquisition = PosteriorMean(model)
    center = _center(train_X, train_Y)
    strategy = TuRBOStrategy(bounds, center=center, num_restarts=2, raw_samples=16)

    result = strategy.optimize(acquisition, q=1)

    torch.testing.assert_close(result.metadata["trust_region_center"], center)
    trust_bounds = result.metadata["trust_region_bounds"]
    assert torch.all(trust_bounds[0] <= result.candidates)
    assert torch.all(trust_bounds[1] >= result.candidates)
    assert result.acquisition_value is not None


def test_optimize_routes_through_named_optimizer_backend() -> None:
    train_X, train_Y, bounds = _problem()
    model = SingleTaskGP(train_X, train_Y)
    acquisition = PosteriorMean(model)
    center = _center(train_X, train_Y)
    strategy = TuRBOStrategy(
        bounds,
        center=center,
        optimizer="sobol",
        raw_samples=32,
        seed=7,
        optimizer_options={"num_samples": 32},
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == (1, bounds.shape[-1])
    assert result.metadata["optimizer"] == "sobol"
    trust_bounds = result.metadata["trust_region_bounds"]
    assert torch.all(result.candidates >= trust_bounds[0])
    assert torch.all(result.candidates <= trust_bounds[1])


def test_strategy_update_state_persists_state() -> None:
    _, _, bounds = _problem()
    strategy = TuRBOStrategy(
        bounds,
        center=bounds.mean(dim=0),
        state=TuRBOState(dim=4, success_tolerance=1, best_value=0.0),
    )

    state = strategy.update_state(torch.tensor([1.0]))

    assert strategy.state is state
    assert state.length == pytest.approx(1.6)
    assert state.best_value == pytest.approx(1.0)


def test_validates_arguments_and_restart_state() -> None:
    train_X, train_Y, bounds = _problem()
    center = _center(train_X, train_Y)
    with pytest.raises(ValueError, match="num_restarts"):
        TuRBOStrategy(bounds, center=center, num_restarts=0)
    with pytest.raises(ValueError, match="raw_samples"):
        TuRBOStrategy(bounds, center=center, raw_samples=0)
    with pytest.raises(ValueError, match="center must have shape"):
        TuRBOStrategy(bounds, center=torch.zeros(3, dtype=torch.double))
    with pytest.raises(ValueError, match="values must contain"):
        update_turbo_state(TuRBOState(dim=4), torch.tensor([]))

    state = TuRBOState(dim=4, length=0.1, length_min=0.1, restart_triggered=True)
    strategy = TuRBOStrategy(bounds, center=center, state=state)
    acquisition = PosteriorMean(SingleTaskGP(train_X, train_Y))
    with pytest.raises(RuntimeError, match="restart is required"):
        strategy.optimize(acquisition)


def test_strategy_rejects_state_dimension_mismatch() -> None:
    _, _, bounds = _problem(input_dim=4)
    with pytest.raises(ValueError, match=r"state\.dim"):
        TuRBOStrategy(
            bounds,
            center=bounds.mean(dim=0),
            state=TuRBOState(dim=3),
        )


def test_thompson_choices_are_reproducible_and_inside_trust_region() -> None:
    dim = 40
    center = torch.full((dim,), 0.5, dtype=torch.double)
    trust_bounds = torch.stack(
        [
            torch.full((dim,), 0.25, dtype=torch.double),
            torch.full((dim,), 0.75, dtype=torch.double),
        ]
    )

    choices_a = generate_turbo_thompson_choices(
        center,
        trust_bounds,
        n_candidates=64,
        seed=17,
    )
    choices_b = generate_turbo_thompson_choices(
        center,
        trust_bounds,
        n_candidates=64,
        seed=17,
    )

    torch.testing.assert_close(choices_a, choices_b)
    assert torch.all(choices_a >= trust_bounds[0])
    assert torch.all(choices_a <= trust_bounds[1])
    assert torch.all((choices_a != center).sum(dim=1) >= 1)


def test_thompson_choices_force_one_perturbation_when_mask_is_empty() -> None:
    dim = 8
    center = torch.full((dim,), 0.5)
    trust_bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

    choices = generate_turbo_thompson_choices(
        center,
        trust_bounds,
        n_candidates=32,
        seed=9,
        perturbation_probability=1e-12,
    )

    assert torch.all((choices != center).sum(dim=1) >= 1)


def test_batch_state_update_uses_best_value_once_per_completed_batch() -> None:
    state = TuRBOState(
        dim=6,
        batch_size=3,
        success_tolerance=2,
        best_value=0.0,
    )

    state = update_turbo_state(state, torch.tensor([-1.0, 0.5, 0.2]))
    assert state.success_counter == 1
    assert state.failure_counter == 0
    assert state.best_value == pytest.approx(0.5)

    state = update_turbo_state(state, torch.tensor([0.4, 0.8, 0.6]))
    assert state.success_counter == 0
    assert state.length == pytest.approx(1.6)
    assert state.best_value == pytest.approx(0.8)


def test_batch_state_update_rejects_partial_or_oversized_results() -> None:
    state = TuRBOState(dim=6, batch_size=3)

    with pytest.raises(ValueError, match=r"state\.batch_size"):
        update_turbo_state(state, torch.tensor([0.1, 0.2]))
    with pytest.raises(ValueError, match=r"state\.batch_size"):
        update_turbo_state(state, torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_strategy_requires_q_to_match_state_batch_size() -> None:
    train_X, train_Y, bounds = _problem()
    model = SingleTaskGP(train_X, train_Y)
    acquisition = PosteriorMean(model)
    strategy = TuRBOStrategy(
        bounds,
        center=_center(train_X, train_Y),
        state=TuRBOState(dim=4, batch_size=2),
    )

    with pytest.raises(ValueError, match=r"q must match state\.batch_size"):
        strategy.optimize(acquisition, q=1)
    with pytest.raises(ValueError, match=r"q must match state\.batch_size"):
        strategy.thompson_sample(model, q=1, n_candidates=16)


def test_batch_thompson_sampling_returns_distinct_local_candidates() -> None:
    train_X, train_Y, bounds = _problem()
    model = SingleTaskGP(train_X, train_Y)
    strategy = TuRBOStrategy(
        bounds,
        center=_center(train_X, train_Y),
        state=TuRBOState(dim=4, batch_size=3),
        seed=21,
    )

    result = strategy.thompson_sample(model, q=3, n_candidates=64)

    assert result.candidates.shape == (3, 4)
    assert result.metadata["batch_size"] == 3
    assert torch.unique(result.candidates, dim=0).shape[0] == 3
    trust_bounds = result.metadata["trust_region_bounds"]
    assert torch.all(result.candidates >= trust_bounds[0])
    assert torch.all(result.candidates <= trust_bounds[1])


def test_restart_state_preserves_best_and_resets_local_counters() -> None:
    state = TuRBOState(
        dim=5,
        batch_size=2,
        length=0.1,
        length_min=0.1,
        success_counter=2,
        failure_counter=3,
        best_value=4.2,
        restart_triggered=True,
        restart_count=1,
    )

    restarted = restart_turbo_state(state, length=0.8)

    assert restarted.length == pytest.approx(0.8)
    assert restarted.success_counter == 0
    assert restarted.failure_counter == 0
    assert restarted.best_value == pytest.approx(4.2)
    assert not restarted.restart_triggered
    assert restarted.restart_count == 2
    assert restarted.batch_size == 2


def test_restart_state_requires_trigger_and_valid_length() -> None:
    with pytest.raises(ValueError, match="restart_triggered"):
        restart_turbo_state(TuRBOState(dim=3))
    state = TuRBOState(dim=3, length=0.1, length_min=0.1, restart_triggered=True)
    with pytest.raises(ValueError, match="restart length"):
        restart_turbo_state(state, length=2.0)


def test_restart_center_is_reproducible_and_inside_global_bounds() -> None:
    bounds = torch.tensor(
        [[-2.0, 10.0, 100.0], [2.0, 30.0, 200.0]],
        dtype=torch.double,
    )

    first = generate_turbo_restart_center(bounds, seed=27)
    second = generate_turbo_restart_center(bounds, seed=27)

    torch.testing.assert_close(first, second)
    assert torch.all(first >= bounds[0])
    assert torch.all(first <= bounds[1])


def test_strategy_restart_reenters_search_with_preserved_best_value() -> None:
    _, _, bounds = _problem()
    state = TuRBOState(
        dim=4,
        length=0.1,
        length_min=0.1,
        best_value=3.5,
        restart_triggered=True,
    )
    strategy = TuRBOStrategy(bounds, center=bounds.mean(dim=0), state=state, seed=29)

    restarted = strategy.restart()

    assert restarted.best_value == pytest.approx(3.5)
    assert restarted.length == pytest.approx(0.8)
    assert restarted.restart_count == 1
    assert not restarted.restart_triggered
    assert torch.all(strategy.center >= bounds[0])
    assert torch.all(strategy.center <= bounds[1])


def test_model_ard_lengthscales_define_turbo_dimension_weights() -> None:
    train_X, train_Y, bounds = _problem(input_dim=4)
    model = SingleTaskGP(train_X, train_Y)
    lengthscales = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.double)
    model.covar_module.lengthscale = lengthscales

    weights = turbo_dimension_weights_from_model(
        model,
        input_dim=4,
        dtype=bounds.dtype,
        device=bounds.device,
        bounds=bounds,
    )

    expected = lengthscales / torch.exp(torch.log(lengthscales).mean())
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(weights.prod(), torch.ones((), dtype=torch.double))


def test_raw_model_lengthscales_are_scaled_by_bound_widths() -> None:
    _, _, _ = _problem(input_dim=2)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 100.0]], dtype=torch.double)
    lengthscales = torch.tensor([1.0, 100.0], dtype=torch.double)
    model = SimpleNamespace(covar_module=SimpleNamespace(lengthscale=lengthscales))

    weights = turbo_dimension_weights_from_model(
        model,
        input_dim=2,
        dtype=bounds.dtype,
        device=bounds.device,
        bounds=bounds,
    )

    torch.testing.assert_close(weights, torch.ones(2, dtype=torch.double))


def test_strategy_can_refresh_geometry_from_model_ard_lengthscales() -> None:
    train_X, train_Y, bounds = _problem(input_dim=4)
    model = SingleTaskGP(train_X, train_Y)
    model.covar_module.lengthscale = torch.tensor(
        [0.25, 0.5, 1.0, 2.0],
        dtype=torch.double,
    )
    strategy = TuRBOStrategy(
        bounds,
        center=torch.full((4,), 0.5, dtype=torch.double),
        state=TuRBOState(dim=4, length=0.2),
    )

    weights = strategy.update_dimension_weights_from_model(model)
    trust_bounds = strategy.trust_region_bounds()

    torch.testing.assert_close(weights, strategy.dimension_weights)
    half_widths = (trust_bounds[1] - trust_bounds[0]) / 2.0
    torch.testing.assert_close(half_widths, 0.1 * weights)


def test_model_geometry_aggregates_batched_ard_lengthscales_by_median() -> None:
    _, _, bounds = _problem(input_dim=4)
    lengthscales = torch.tensor(
        [
            [0.2, 0.5, 1.0, 2.0],
            [0.4, 1.0, 2.0, 4.0],
            [0.8, 2.0, 4.0, 8.0],
        ],
        dtype=torch.double,
    )
    model = SimpleNamespace(covar_module=SimpleNamespace(lengthscale=lengthscales))

    weights = turbo_dimension_weights_from_model(
        model,
        input_dim=4,
        dtype=bounds.dtype,
        device=bounds.device,
        bounds=bounds,
    )

    median = lengthscales.median(dim=0).values
    expected = median / torch.exp(torch.log(median).mean())
    torch.testing.assert_close(weights, expected)


def test_model_geometry_rejects_non_public_lengthscale_dimension() -> None:
    _, _, bounds = _problem(input_dim=4)
    model = SimpleNamespace(
        covar_module=SimpleNamespace(lengthscale=torch.ones(3, dtype=torch.double))
    )

    with pytest.raises(ValueError, match="public input dimension"):
        turbo_dimension_weights_from_model(
            model,
            input_dim=4,
            dtype=bounds.dtype,
            device=bounds.device,
        )


def test_model_geometry_rejects_reduced_space_lengthscales() -> None:
    model = SimpleNamespace(
        original_input_dim=8,
        reduced_input_dim=3,
        covar_module=SimpleNamespace(lengthscale=torch.ones(3, dtype=torch.double)),
    )
    bounds = torch.stack(
        [
            torch.zeros(8, dtype=torch.double),
            torch.ones(8, dtype=torch.double),
        ]
    )

    with pytest.raises(ValueError, match="reduced input space"):
        turbo_dimension_weights_from_model(
            model,
            input_dim=8,
            dtype=bounds.dtype,
            device=bounds.device,
            bounds=bounds,
        )


def test_optimize_intersects_trust_region_with_linear_inequality() -> None:
    train_X, train_Y, bounds = _problem()
    acquisition = PosteriorMean(SingleTaskGP(train_X, train_Y))
    strategy = TuRBOStrategy(
        bounds,
        center=torch.full((4,), 0.75, dtype=torch.double),
        num_restarts=4,
        raw_samples=64,
    )
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([0]),
                torch.tensor([1.0], dtype=torch.double),
                0.7,
            ),
        )
    )

    result = strategy.optimize(acquisition, constraints=constraints)

    assert result.candidates[0, 0] >= 0.7 - 1e-6
    trust_bounds = result.metadata["trust_region_bounds"]
    assert torch.all(result.candidates >= trust_bounds[0])
    assert torch.all(result.candidates <= trust_bounds[1])
    assert result.metadata["candidate_constraints"]


def test_optimize_supports_linear_equality_inside_trust_region() -> None:
    train_X, train_Y, bounds = _problem()
    acquisition = PosteriorMean(SingleTaskGP(train_X, train_Y))
    strategy = TuRBOStrategy(
        bounds,
        center=torch.full((4,), 0.5, dtype=torch.double),
        num_restarts=4,
        raw_samples=64,
    )
    constraints = CandidateConstraints(
        equality_constraints=(
            (
                torch.tensor([0, 1]),
                torch.tensor([1.0, 1.0], dtype=torch.double),
                1.0,
            ),
        )
    )

    result = strategy.optimize(acquisition, constraints=constraints)

    torch.testing.assert_close(
        result.candidates[0, :2].sum(),
        torch.tensor(1.0, dtype=torch.double),
        atol=1e-5,
        rtol=0.0,
    )


def test_optimize_supports_nonlinear_constraint_with_feasible_initial_conditions() -> None:
    train_X, train_Y, bounds = _problem()
    acquisition = PosteriorMean(SingleTaskGP(train_X, train_Y))
    strategy = TuRBOStrategy(
        bounds,
        center=torch.full((4,), 0.7, dtype=torch.double),
        num_restarts=2,
        raw_samples=32,
    )
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: x[0] - 0.6, True),)
    )
    initial = torch.full((2, 1, 4), 0.7, dtype=torch.double)

    result = strategy.optimize(
        acquisition,
        constraints=constraints,
        batch_initial_conditions=initial,
    )

    assert result.candidates[0, 0] >= 0.6 - 1e-6


def test_thompson_sampling_filters_candidate_constraints() -> None:
    train_X, train_Y, bounds = _problem()
    model = SingleTaskGP(train_X, train_Y)
    strategy = TuRBOStrategy(
        bounds,
        center=torch.full((4,), 0.75, dtype=torch.double),
        seed=53,
    )
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([0]),
                torch.tensor([1.0], dtype=torch.double),
                0.7,
            ),
        ),
        nonlinear_inequality_constraints=((lambda x: 0.9 - x[..., 1], True),),
    )

    result = strategy.thompson_sample(model, n_candidates=128, constraints=constraints)

    assert result.candidates[0, 0] >= 0.7
    assert result.candidates[0, 1] <= 0.9
    assert result.metadata["candidate_constraints"]


def test_thompson_sampling_rejects_infeasible_candidate_pool() -> None:
    train_X, train_Y, bounds = _problem()
    model = SingleTaskGP(train_X, train_Y)
    strategy = TuRBOStrategy(
        bounds,
        center=torch.full((4,), 0.5, dtype=torch.double),
        seed=59,
    )
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([0]),
                torch.tensor([1.0], dtype=torch.double),
                0.99,
            ),
        )
    )

    with pytest.raises(RuntimeError, match="fewer than q feasible points"):
        strategy.thompson_sample(model, n_candidates=32, constraints=constraints)


def test_thompson_sampling_rejects_interpoint_constraints() -> None:
    train_X, train_Y, bounds = _problem()
    model = SingleTaskGP(train_X, train_Y)
    strategy = TuRBOStrategy(
        bounds,
        center=_center(train_X, train_Y),
        state=TuRBOState(dim=4, batch_size=2),
    )
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([[0, 0], [1, 0]]),
                torch.tensor([1.0, -1.0], dtype=torch.double),
                0.0,
            ),
        )
    )

    with pytest.raises(NotImplementedError, match="inter-point linear"):
        strategy.thompson_sample(model, q=2, n_candidates=32, constraints=constraints)


def test_noise_aware_state_uses_state_values_for_success_decision() -> None:
    state = TuRBOState(
        dim=2,
        best_value=1.0,
        observed_best_value=1.0,
    )

    next_state = update_turbo_state(
        state,
        torch.tensor([2.5]),
        state_values=torch.tensor([0.9]),
    )

    assert next_state.best_value == pytest.approx(1.0)
    assert next_state.observed_best_value == pytest.approx(2.5)
    assert next_state.success_counter == 0
    assert next_state.failure_counter == 1


def test_noise_aware_strategy_moves_center_by_state_utility() -> None:
    bounds = torch.stack([torch.zeros(2), torch.ones(2)])
    initial = torch.tensor([0.5, 0.5])
    candidates = torch.tensor([[0.2, 0.2], [0.8, 0.8]])
    strategy = TuRBOStrategy(
        bounds,
        center=initial,
        state=TuRBOState(dim=2, batch_size=2, best_value=0.5),
    )

    next_state = strategy.update_state(
        torch.tensor([10.0, 1.0]),
        candidates=candidates,
        state_values=torch.tensor([0.4, 0.8]),
    )

    torch.testing.assert_close(strategy.center, candidates[1])
    assert next_state.best_value == pytest.approx(0.8)
    assert next_state.observed_best_value == pytest.approx(10.0)


def test_turbo_candidate_paths_share_stable_result_metadata() -> None:
    train_X, train_Y, bounds = _problem()
    model = SingleTaskGP(train_X, train_Y)
    strategy = TuRBOStrategy(
        bounds,
        center=_center(train_X, train_Y),
        seed=67,
        num_restarts=2,
        raw_samples=16,
    )

    acquisition_result = strategy.optimize(PosteriorMean(model))
    thompson_result = strategy.thompson_sample(model, n_candidates=32)
    common_keys = {
        "trust_region_center",
        "trust_region_bounds",
        "trust_region_length",
        "success_counter",
        "failure_counter",
        "restart_count",
        "batch_size",
        "candidate_generation",
        "candidate_constraints",
    }

    assert common_keys <= acquisition_result.metadata.keys()
    assert common_keys <= thompson_result.metadata.keys()
    assert acquisition_result.metadata["candidate_generation"] == "acquisition"
    assert thompson_result.metadata["candidate_generation"] == "thompson"
