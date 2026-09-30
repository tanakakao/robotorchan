"""Integration regressions for TuRBO public-space state handling."""

import pytest
import torch
from botorch.acquisition.analytic import ExpectedImprovement, PosteriorMean
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import (
    PCAGP,
    PLSGP,
    EnsembleMapSaasSingleTaskGP,
    RandomProjectionGP,
    SingleTaskGP,
)
from robotorchan.optim import (
    TuRBOState,
    TuRBOStrategy,
    generate_turbo_thompson_choices,
    update_turbo_state,
)


def test_first_observation_initializes_turbo_state_as_success() -> None:
    state = update_turbo_state(TuRBOState(dim=3), torch.tensor([0.5]))

    assert state.best_value == pytest.approx(0.5)
    assert state.success_counter == 1
    assert state.failure_counter == 0


def test_reduced_gp_uses_explicit_public_space_incumbent() -> None:
    torch.manual_seed(23)
    input_dim = 6
    train_X = torch.rand(12, input_dim, dtype=torch.double)
    train_Y = -((train_X[:, :2] - 0.65) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )
    model = PCAGP(train_X, train_Y, n_components=3)
    acquisition = PosteriorMean(model)
    incumbent = train_X[train_Y.squeeze(-1).argmax()]
    strategy = TuRBOStrategy(
        bounds,
        center=incumbent,
        num_restarts=2,
        raw_samples=16,
    )

    result = strategy.optimize(acquisition)

    torch.testing.assert_close(result.metadata["trust_region_center"], incumbent)
    assert result.candidates.shape == (1, input_dim)


@pytest.mark.parametrize(
    ("model_cls", "model_kwargs"),
    [
        (PCAGP, {"n_components": 3}),
        (PLSGP, {"n_components": 3}),
        (RandomProjectionGP, {"n_components": 3, "random_state": 17}),
    ],
)
def test_frozen_reduced_models_optimize_turbo_in_public_space(
    model_cls,
    model_kwargs,
) -> None:
    torch.manual_seed(47)
    input_dim = 8
    train_X = torch.rand(16, input_dim, dtype=torch.double)
    train_Y = -((train_X[:, :3] - 0.6) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )
    model = model_cls(train_X, train_Y, **model_kwargs)
    acquisition = PosteriorMean(model)
    incumbent = train_X[train_Y.squeeze(-1).argmax()]
    strategy = TuRBOStrategy(
        bounds,
        center=incumbent,
        num_restarts=2,
        raw_samples=16,
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == (1, input_dim)
    assert torch.all(result.candidates >= bounds[0])
    assert torch.all(result.candidates <= bounds[1])
    torch.testing.assert_close(result.metadata["trust_region_center"], incumbent)


def test_update_state_moves_incumbent_when_candidate_improves() -> None:
    bounds = torch.stack([torch.zeros(3), torch.ones(3)])
    initial = torch.tensor([0.2, 0.3, 0.4])
    improved = torch.tensor([[0.8, 0.7, 0.6]])
    strategy = TuRBOStrategy(
        bounds,
        center=initial,
        state=TuRBOState(dim=3, best_value=0.0),
    )

    strategy.update_state(torch.tensor([1.0]), candidates=improved)

    torch.testing.assert_close(strategy.center, improved[0])


def test_baseline_turbo1_runs_multiple_local_bo_iterations() -> None:
    torch.manual_seed(31)
    input_dim = 3
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )

    def objective(x: torch.Tensor) -> torch.Tensor:
        return -((x - 0.72) ** 2).sum(dim=-1, keepdim=True)

    train_X = torch.rand(8, input_dim, dtype=torch.double)
    train_Y = objective(train_X)
    best_index = train_Y.squeeze(-1).argmax()
    strategy = TuRBOStrategy(
        bounds,
        center=train_X[best_index],
        state=TuRBOState(
            dim=input_dim,
            best_value=float(train_Y[best_index].item()),
        ),
        num_restarts=2,
        raw_samples=32,
    )

    initial_best = strategy.state.best_value
    for _ in range(2):
        model = SingleTaskGP(train_X, train_Y)
        acquisition = ExpectedImprovement(
            model,
            best_f=float(train_Y.max().item()),
        )
        result = strategy.optimize(acquisition)
        new_X = result.candidates.detach()
        new_Y = objective(new_X)

        strategy.update_state(new_Y, candidates=new_X)
        train_X = torch.cat([train_X, new_X], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        trust_bounds = result.metadata["trust_region_bounds"]
        assert torch.all(new_X >= trust_bounds[0])
        assert torch.all(new_X <= trust_bounds[1])

    assert train_X.shape == (10, input_dim)
    assert strategy.state.best_value >= initial_best
    assert not strategy.state.restart_triggered


def test_turbo_thompson_sampling_selects_local_candidate() -> None:
    torch.manual_seed(37)
    input_dim = 4
    train_X = torch.rand(12, input_dim, dtype=torch.double)
    train_Y = -((train_X - 0.65) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )
    model = SingleTaskGP(train_X, train_Y)
    incumbent = train_X[train_Y.squeeze(-1).argmax()]
    strategy = TuRBOStrategy(
        bounds,
        center=incumbent,
        state=TuRBOState(
            dim=input_dim,
            best_value=float(train_Y.max().item()),
        ),
        seed=13,
    )

    result = strategy.thompson_sample(model, n_candidates=64)

    assert result.candidates.shape == (1, input_dim)
    assert result.acquisition_value is None
    assert result.metadata["candidate_generation"] == "thompson"
    trust_bounds = result.metadata["trust_region_bounds"]
    assert torch.all(result.candidates >= trust_bounds[0])
    assert torch.all(result.candidates <= trust_bounds[1])


def test_batch_turbo_runs_joint_acquisition_and_state_update() -> None:
    torch.manual_seed(41)
    input_dim = 3
    batch_size = 2
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )

    def objective(x: torch.Tensor) -> torch.Tensor:
        return -((x - 0.68) ** 2).sum(dim=-1, keepdim=True)

    train_X = torch.rand(10, input_dim, dtype=torch.double)
    train_Y = objective(train_X)
    best_index = train_Y.squeeze(-1).argmax()
    strategy = TuRBOStrategy(
        bounds,
        center=train_X[best_index],
        state=TuRBOState(
            dim=input_dim,
            batch_size=batch_size,
            best_value=float(train_Y[best_index].item()),
        ),
        num_restarts=2,
        raw_samples=32,
    )
    model = SingleTaskGP(train_X, train_Y)
    acquisition = qExpectedImprovement(
        model,
        best_f=float(train_Y.max().item()),
    )

    result = strategy.optimize(acquisition, q=batch_size)
    new_X = result.candidates.detach()
    new_Y = objective(new_X)
    next_state = strategy.update_state(new_Y, candidates=new_X)

    assert new_X.shape == (batch_size, input_dim)
    assert result.metadata["batch_size"] == batch_size
    assert next_state.best_value >= float(train_Y.max().item())
    trust_bounds = result.metadata["trust_region_bounds"]
    assert torch.all(new_X >= trust_bounds[0])
    assert torch.all(new_X <= trust_bounds[1])


def test_turbo_can_resume_candidate_generation_after_restart() -> None:
    torch.manual_seed(43)
    input_dim = 3
    train_X = torch.rand(10, input_dim, dtype=torch.double)
    train_Y = -((train_X - 0.6) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )
    model = SingleTaskGP(train_X, train_Y)
    state = TuRBOState(
        dim=input_dim,
        length=0.1,
        length_min=0.1,
        best_value=float(train_Y.max().item()),
        restart_triggered=True,
    )
    strategy = TuRBOStrategy(
        bounds,
        center=train_X[train_Y.squeeze(-1).argmax()],
        state=state,
        seed=31,
    )

    strategy.restart()
    result = strategy.thompson_sample(model, n_candidates=64)

    assert result.candidates.shape == (1, input_dim)
    assert result.metadata["restart_count"] == 1
    assert torch.all(result.candidates >= result.metadata["trust_region_bounds"][0])
    assert torch.all(result.candidates <= result.metadata["trust_region_bounds"][1])


def test_turbo_high_dimensional_map_saas_thompson_path() -> None:
    torch.manual_seed(47)
    input_dim = 20
    train_X = torch.rand(24, input_dim, dtype=torch.double)
    train_Y = -((train_X[:, :3] - 0.7) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )
    model = EnsembleMapSaasSingleTaskGP(train_X, train_Y, num_taus=2)
    incumbent = train_X[train_Y.squeeze(-1).argmax()]
    strategy = TuRBOStrategy(
        bounds,
        center=incumbent,
        state=TuRBOState(
            dim=input_dim,
            best_value=float(train_Y.max().item()),
        ),
        seed=37,
    )

    result = strategy.thompson_sample(model, n_candidates=128)

    assert result.candidates.shape == (1, input_dim)
    trust_bounds = result.metadata["trust_region_bounds"]
    assert torch.all(result.candidates >= trust_bounds[0])
    assert torch.all(result.candidates <= trust_bounds[1])

def test_turbo_async_pending_is_acquisition_context_not_state_update() -> None:
    torch.manual_seed(61)
    input_dim = 2
    train_X = torch.rand(12, input_dim, dtype=torch.double)
    train_Y = -((train_X - 0.65) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )
    best_index = train_Y.squeeze(-1).argmax()
    strategy = TuRBOStrategy(
        bounds,
        center=train_X[best_index],
        state=TuRBOState(
            dim=input_dim,
            best_value=float(train_Y.max().item()),
        ),
        num_restarts=2,
        raw_samples=32,
    )
    model = SingleTaskGP(train_X, train_Y)
    acquisition = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_X,
        sampler=SobolQMCNormalSampler(torch.Size([32]), seed=611),
    )
    pending = torch.tensor([[0.2, 0.8], [0.8, 0.2]], dtype=torch.double)
    state_before = strategy.state

    result = strategy.optimize(acquisition, X_pending=pending)

    assert strategy.state is state_before
    assert result.metadata["n_pending"] == 2
    assert acquisition.X_pending is None


def test_turbo_async_completion_updates_only_completed_evaluation() -> None:
    torch.manual_seed(67)
    input_dim = 2
    train_X = torch.rand(10, input_dim, dtype=torch.double)
    train_Y = -((train_X - 0.6) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )
    best_index = train_Y.squeeze(-1).argmax()
    strategy = TuRBOStrategy(
        bounds,
        center=train_X[best_index],
        state=TuRBOState(
            dim=input_dim,
            best_value=float(train_Y.max().item()),
        ),
        num_restarts=2,
        raw_samples=32,
    )
    completed = torch.tensor([[0.6, 0.6]], dtype=torch.double)
    completed_value = -((completed - 0.6) ** 2).sum(dim=-1)
    pending = torch.tensor([[0.25, 0.75]], dtype=torch.double)

    strategy.update_state(completed_value, candidates=completed)
    state_after_completion = strategy.state
    updated_X = torch.cat([train_X, completed], dim=0)
    updated_Y = torch.cat([train_Y, completed_value.unsqueeze(-1)], dim=0)
    acquisition = qLogNoisyExpectedImprovement(
        model=SingleTaskGP(updated_X, updated_Y),
        X_baseline=updated_X,
        sampler=SobolQMCNormalSampler(torch.Size([32]), seed=671),
    )

    result = strategy.optimize(acquisition, X_pending=pending)

    assert strategy.state is state_after_completion
    assert result.metadata["n_pending"] == 1
    assert acquisition.X_pending is None


def test_turbo_optimize_restores_preexisting_pending_points() -> None:
    torch.manual_seed(71)
    train_X = torch.rand(10, 2, dtype=torch.double)
    train_Y = -((train_X - 0.7) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [torch.zeros(2, dtype=torch.double), torch.ones(2, dtype=torch.double)]
    )
    model = SingleTaskGP(train_X, train_Y)
    acquisition = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_X,
        sampler=SobolQMCNormalSampler(torch.Size([32]), seed=711),
    )
    original_pending = torch.tensor([[0.1, 0.1]], dtype=torch.double)
    acquisition.set_X_pending(original_pending)
    strategy = TuRBOStrategy(
        bounds,
        center=train_X[train_Y.squeeze(-1).argmax()],
        num_restarts=2,
        raw_samples=32,
    )

    strategy.optimize(
        acquisition,
        X_pending=torch.tensor([[0.9, 0.9]], dtype=torch.double),
    )

    torch.testing.assert_close(acquisition.X_pending, original_pending)


def test_turbo_thompson_rejects_pending_pool_candidate() -> None:
    torch.manual_seed(73)
    input_dim = 3
    train_X = torch.rand(10, input_dim, dtype=torch.double)
    train_Y = -((train_X - 0.6) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )
    center = train_X[train_Y.squeeze(-1).argmax()]
    strategy = TuRBOStrategy(bounds, center=center, seed=733)
    trust_bounds = strategy.trust_region_bounds()
    pending = generate_turbo_thompson_choices(
        center,
        trust_bounds,
        n_candidates=1,
        seed=733,
    )

    with pytest.raises(RuntimeError, match="non-pending"):
        strategy.thompson_sample(
            SingleTaskGP(train_X, train_Y),
            n_candidates=1,
            X_pending=pending,
        )

