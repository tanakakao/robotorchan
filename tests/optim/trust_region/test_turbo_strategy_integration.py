"""Integration regressions for TuRBO public-space state handling."""

import pytest
import torch
from botorch.acquisition.analytic import ExpectedImprovement, PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.risk_measures import Expectation
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.transforms.input import InputPerturbation
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from robotorchan.models import (
    PCAGP,
    PLSGP,
    EnsembleMapSaasSingleTaskGP,
    MixedSingleTaskGP,
    ModelListGP,
    RandomProjectionGP,
    SingleTaskGP,
    SingleTaskMultiFidelityGP,
)
from robotorchan.optim import (
    MixedVariableSpace,
    TuRBOState,
    TuRBOStrategy,
    generate_turbo_thompson_choices,
    turbo_mixed_trust_region_bounds,
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
    pending = torch.tensor([[0.2, 0.8], [0.8, 0.2]], dtype=torch.double)
    acquisition = qLogNoisyExpectedImprovement(
        model=SingleTaskGP(train_X, train_Y),
        X_baseline=train_X,
        sampler=SobolQMCNormalSampler(torch.Size([32]), seed=611),
        X_pending=pending,
    )
    state_before = strategy.state
    baseline_with_pending = acquisition.X_baseline.detach().clone()

    result = strategy.optimize(acquisition)

    assert strategy.state is state_before
    torch.testing.assert_close(acquisition.X_baseline, baseline_with_pending)
    assert result.candidates.shape == torch.Size([1, input_dim])


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
        X_pending=pending,
    )

    result = strategy.optimize(acquisition)

    assert strategy.state is state_after_completion
    assert result.candidates.shape == torch.Size([1, input_dim])


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


def test_turbo_qlognei_runs_with_input_perturbation_in_nominal_space() -> None:
    torch.manual_seed(79)
    input_dim = 2
    train_X = 0.1 + 0.8 * torch.rand(14, input_dim, dtype=torch.double)
    train_Y = -((train_X - 0.65) ** 2).sum(dim=-1, keepdim=True)
    bounds = torch.tensor([[0.1, 0.1], [0.9, 0.9]], dtype=torch.double)
    perturbations = torch.tensor(
        [[0.0, 0.0], [0.01, -0.01], [-0.01, 0.01]],
        dtype=torch.double,
    )
    model = SingleTaskGP(
        train_X,
        train_Y,
        input_transform=InputPerturbation(perturbation_set=perturbations),
    )
    model.eval()
    acquisition = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_X,
        sampler=SobolQMCNormalSampler(torch.Size([32]), seed=791),
        objective=Expectation(n_w=perturbations.shape[0]),
        prune_baseline=False,
    )
    incumbent = train_X[train_Y.squeeze(-1).argmax()]
    strategy = TuRBOStrategy(
        bounds,
        center=incumbent,
        state=TuRBOState(
            dim=input_dim,
            best_value=float(train_Y.max().item()),
            observed_best_value=float(train_Y.max().item()),
        ),
        num_restarts=2,
        raw_samples=32,
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == torch.Size([1, input_dim])
    assert result.metadata["trust_region_center"].shape == torch.Size([input_dim])
    assert result.metadata["trust_region_bounds"].shape == torch.Size([2, input_dim])
    assert torch.all(result.candidates >= result.metadata["trust_region_bounds"][0])
    assert torch.all(result.candidates <= result.metadata["trust_region_bounds"][1])


def test_turbo_mixed_optimizes_continuous_region_and_exact_categories() -> None:
    torch.manual_seed(83)
    train_X = torch.tensor(
        [
            [0.15, 0.0],
            [0.35, 0.0],
            [0.75, 0.0],
            [0.20, 1.0],
            [0.55, 1.0],
            [0.85, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = -((train_X[:, :1] - 0.7) ** 2) + 0.15 * train_X[:, 1:]
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    model = MixedSingleTaskGP(train_X, train_Y, cat_dims=[1])
    acquisition = PosteriorMean(model)
    variable_space = MixedVariableSpace(bounds, categorical_values={1: [0.0, 1.0]})
    strategy = TuRBOStrategy(
        bounds,
        center=torch.tensor([0.55, 1.0], dtype=torch.double),
        state=TuRBOState(dim=2, length=0.4),
        num_restarts=2,
        raw_samples=32,
    )

    result = strategy.optimize_mixed(acquisition, variable_space)

    trust_bounds = result.metadata["trust_region_bounds"]
    assert result.candidates.shape == torch.Size([1, 2])
    assert trust_bounds[0, 0] > bounds[0, 0]
    assert trust_bounds[1, 0] < bounds[1, 0]
    torch.testing.assert_close(trust_bounds[:, 1], bounds[:, 1])
    assert result.candidates[0, 1].item() in {0.0, 1.0}
    assert result.metadata["n_discrete_assignments"] == 2


def test_turbo_mixed_enumerates_integer_values_without_numeric_shrinking() -> None:
    bounds = torch.tensor([[0.0, 1.0], [1.0, 3.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(bounds, integer_dims=(1,))
    local = turbo_mixed_trust_region_bounds(
        torch.tensor([0.5, 2.0], dtype=torch.double),
        variable_space,
        length=0.2,
    )

    assert local[0, 0] > bounds[0, 0]
    assert local[1, 0] < bounds[1, 0]
    torch.testing.assert_close(local[:, 1], bounds[:, 1])


def _multiobjective_turbo_problem() -> tuple[ModelListGP, torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.05, 0.95, 10, dtype=torch.double).unsqueeze(-1)
    first = -((train_X - 0.25) ** 2)
    second = -((train_X - 0.75) ** 2)
    train_Y = torch.cat([first, second], dim=-1)
    model = ModelListGP(
        SingleTaskGP(train_X, first),
        SingleTaskGP(train_X, second),
    )
    return model, train_X, train_Y


def test_turbo_optimizes_qlogehvi_inside_local_region() -> None:
    model, train_X, train_Y = _multiobjective_turbo_problem()
    ref_point = train_Y.min(dim=0).values - 0.1
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_Y)
    acquisition = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
    )
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    strategy = TuRBOStrategy(
        bounds,
        center=train_X[4],
        num_restarts=2,
        raw_samples=32,
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == torch.Size([1, 1])
    trust_bounds = result.metadata["trust_region_bounds"]
    assert torch.all(result.candidates >= trust_bounds[0])
    assert torch.all(result.candidates <= trust_bounds[1])


def test_turbo_optimizes_qlognehvi_inside_local_region() -> None:
    model, train_X, train_Y = _multiobjective_turbo_problem()
    ref_point = train_Y.min(dim=0).values - 0.1
    acquisition = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        X_baseline=train_X,
        prune_baseline=False,
    )
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    strategy = TuRBOStrategy(
        bounds,
        center=train_X[5],
        num_restarts=2,
        raw_samples=32,
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == torch.Size([1, 1])
    trust_bounds = result.metadata["trust_region_bounds"]
    assert torch.all(result.candidates >= trust_bounds[0])
    assert torch.all(result.candidates <= trust_bounds[1])


def test_turbo_state_accepts_singleton_output_dimension() -> None:
    strategy = TuRBOStrategy(
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        center=torch.tensor([0.5], dtype=torch.double),
    )

    state = strategy.update_state(
        torch.tensor([[0.6]], dtype=torch.double),
        candidates=torch.tensor([[0.55]], dtype=torch.double),
    )

    assert state.best_value == pytest.approx(0.6)


def test_turbo_state_rejects_multiobjective_vectors_without_scalar_utility() -> None:
    strategy = TuRBOStrategy(
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        center=torch.tensor([0.5], dtype=torch.double),
    )

    with pytest.raises(ValueError, match="scalar utility"):
        strategy.update_state(
            torch.tensor([[0.4, 0.6]], dtype=torch.double),
            candidates=torch.tensor([[0.55]], dtype=torch.double),
        )


def test_turbo_multifidelity_kg_keeps_fidelity_globally_selectable() -> None:
    design = torch.linspace(0.0, 1.0, 6, dtype=torch.double)
    low = torch.stack((design, torch.full_like(design, 0.5)), dim=-1)
    high = torch.stack((design, torch.ones_like(design)), dim=-1)
    train_X = torch.cat((low, high), dim=0)
    train_Y = -((train_X[:, :1] - 0.7) ** 2) + 0.1 * train_X[:, 1:]
    model = SingleTaskMultiFidelityGP(train_X, train_Y, data_fidelities=[1])
    bounds = torch.tensor([[0.0, 0.5], [1.0, 1.0]], dtype=torch.double)
    center = torch.tensor([0.6, 1.0], dtype=torch.double)

    target = PosteriorMean(model)
    target_strategy = TuRBOStrategy(bounds, center=center, num_restarts=2, raw_samples=16)
    target_result = target_strategy.optimize_multifidelity(
        target,
        fidelity_dims=[1],
    )
    current_value = target_result.acquisition_value
    assert current_value is not None

    cost_model = AffineFidelityCostModel(fidelity_weights={1: 1.0}, fixed_cost=0.1)
    cost_utility = InverseCostWeightedUtility(cost_model=cost_model)

    def project(X: torch.Tensor) -> torch.Tensor:
        projected = X.clone()
        projected[..., 1] = 1.0
        return projected

    acquisition = qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=4,
        current_value=current_value,
        cost_aware_utility=cost_utility,
        project=project,
    )
    strategy = TuRBOStrategy(bounds, center=center, num_restarts=2, raw_samples=16)
    result = strategy.optimize_multifidelity(acquisition, fidelity_dims=[1])

    trust_bounds = result.metadata["trust_region_bounds"]
    torch.testing.assert_close(trust_bounds[:, 1], bounds[:, 1])
    torch.testing.assert_close(
        trust_bounds[:, 0],
        torch.tensor([0.2, 1.0], dtype=torch.double),
    )
    assert result.candidates.shape == (1, 2)
    assert torch.all(result.candidates >= trust_bounds[0])
    assert torch.all(result.candidates <= trust_bounds[1])


def test_multifidelity_state_utility_can_decouple_low_fidelity_observation() -> None:
    bounds = torch.tensor([[0.0, 0.5], [1.0, 1.0]], dtype=torch.double)
    center = torch.tensor([0.4, 1.0], dtype=torch.double)
    strategy = TuRBOStrategy(
        bounds,
        center=center,
        state=TuRBOState(dim=2, best_value=0.8),
    )
    low_fidelity_candidate = torch.tensor([[0.7, 0.5]], dtype=torch.double)

    state = strategy.update_state(
        torch.tensor([[1.2]], dtype=torch.double),
        state_values=torch.tensor([[0.7]], dtype=torch.double),
        candidates=low_fidelity_candidate,
    )

    torch.testing.assert_close(strategy.center, center)
    assert state.observed_best_value == pytest.approx(1.2)
    assert state.best_value == pytest.approx(0.8)
