"""Runtime smoke tests for capability-advertised BoTorch workflows."""

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.cost import AffineFidelityCostModel
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from robotorchan.acquisition.mixed_one_shot import optimize_mixed_one_shot_acqf
from robotorchan.models import (
    PCAGP,
    PLSGP,
    KroneckerMultiTaskGP,
    MixedKroneckerMultiTaskGP,
    MixedPCAGP,
    MixedPLSGP,
    MixedRandomProjectionGP,
    MixedReducedGP,
    MixedSingleTaskGP,
    MixedSingleTaskMultiFidelityGP,
    RandomForestSurrogate,
    RandomProjectionGP,
    ReducedGP,
    SingleTaskGP,
    SingleTaskMultiFidelityGP,
)
from robotorchan.reduction.input import PCAInputReducer


def test_single_task_gp_runtime_supports_mc_acquisition() -> None:
    train_x = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_y = torch.sin(train_x * 3.0)

    model = SingleTaskGP(train_x, train_y)
    model.eval()

    posterior = model.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))
    samples = posterior.rsample(torch.Size([4]))

    assert samples.shape == torch.Size([4, 2, 1])
    assert torch.isfinite(samples).all()

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
    )
    value = acquisition(torch.tensor([[[0.5]]], dtype=torch.double))
    candidate, optimized_value = optimize_acqf(
        acq_function=acquisition,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=1,
        num_restarts=2,
        raw_samples=8,
    )

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
    assert candidate.shape == torch.Size([1, 1])
    assert torch.isfinite(optimized_value).all()


def test_mixed_single_task_gp_runtime_supports_mc_acquisition() -> None:
    train_x = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 1.0],
            [0.4, 0.0],
            [0.6, 1.0],
            [0.8, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_y = torch.sin(train_x[:, :1] * 3.0) + 0.2 * train_x[:, 1:].eq(1.0).to(dtype=torch.double)

    model = MixedSingleTaskGP(train_x, train_y, cat_dims=[1])
    model.eval()

    test_x = torch.tensor([[0.25, 0.0], [0.75, 1.0]], dtype=torch.double)
    samples = model.posterior(test_x).rsample(torch.Size([4]))

    assert samples.shape == torch.Size([4, 2, 1])
    assert torch.isfinite(samples).all()

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
    )
    value = acquisition(torch.tensor([[[0.5, 1.0]]], dtype=torch.double))
    candidate, optimized_value = optimize_acqf_mixed(
        acq_function=acquisition,
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        q=1,
        num_restarts=2,
        raw_samples=8,
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
    )

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
    assert candidate.shape == torch.Size([1, 2])
    assert candidate[0, 1].item() in {0.0, 1.0}
    assert torch.isfinite(optimized_value).all()


def test_kronecker_multitask_gp_runtime_supports_multi_output_sampling() -> None:
    train_x = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_y = torch.cat(
        [
            torch.sin(train_x * 3.0),
            torch.cos(train_x * 3.0),
        ],
        dim=-1,
    )

    model = KroneckerMultiTaskGP(train_x, train_y)
    model.eval()

    test_x = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    posterior = model.posterior(test_x)
    samples = posterior.rsample(torch.Size([4]))

    assert posterior.mean.shape == torch.Size([2, 2])
    assert samples.shape == torch.Size([4, 2, 2])
    assert torch.isfinite(samples).all()


def test_kronecker_multitask_gp_runtime_supports_scalarized_mc_acquisition() -> None:
    train_x = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_y = torch.cat(
        [
            torch.sin(train_x * 3.0),
            torch.cos(train_x * 3.0),
        ],
        dim=-1,
    )

    model = KroneckerMultiTaskGP(train_x, train_y)
    model.eval()

    weights = torch.tensor([0.7, 0.3], dtype=torch.double)
    objective = GenericMCObjective(lambda samples, X=None: samples @ weights)
    best_f = (train_y @ weights).max()
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
        objective=objective,
    )
    value = acquisition(torch.tensor([[[0.5]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_kronecker_multitask_gp_runtime_supports_scalarized_qucb() -> None:
    train_x = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_y = torch.cat(
        [
            torch.sin(train_x * 3.0),
            torch.cos(train_x * 3.0),
        ],
        dim=-1,
    )
    model = KroneckerMultiTaskGP(train_x, train_y)
    model.eval()

    weights = torch.tensor([0.7, 0.3], dtype=torch.double)
    objective = GenericMCObjective(lambda samples, X=None: samples @ weights)
    acquisition = qUpperConfidenceBound(
        model=model,
        beta=0.2,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
        objective=objective,
    )
    value = acquisition(torch.tensor([[[0.5]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_kronecker_multitask_gp_runtime_optimizes_scalarized_qlogei() -> None:
    train_x = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_y = torch.cat(
        [
            torch.sin(train_x * 3.0),
            torch.cos(train_x * 3.0),
        ],
        dim=-1,
    )
    model = KroneckerMultiTaskGP(train_x, train_y)
    model.eval()

    weights = torch.tensor([0.7, 0.3], dtype=torch.double)
    objective = GenericMCObjective(lambda samples, X=None: samples @ weights)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=(train_y @ weights).max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
        objective=objective,
    )
    candidate, value = optimize_acqf(
        acq_function=acquisition,
        bounds=torch.tensor([[0.1], [0.9]], dtype=torch.double),
        q=1,
        num_restarts=2,
        raw_samples=8,
    )

    assert candidate.shape == torch.Size([1, 1])
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()


def test_kronecker_multitask_gp_runtime_supports_multi_objective_acquisition() -> None:
    train_x = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_y = torch.cat(
        [
            torch.sin(train_x * 3.0),
            torch.cos(train_x * 3.0),
        ],
        dim=-1,
    )

    model = KroneckerMultiTaskGP(train_x, train_y)
    model.eval()

    ref_point = train_y.min(dim=0).values - 0.1
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_y)
    acquisition = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
    )
    value = acquisition(torch.tensor([[[0.5]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_kronecker_multitask_gp_runtime_supports_noisy_multi_objective_acquisition() -> None:
    train_x = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_y = torch.cat(
        [
            torch.sin(train_x * 3.0),
            torch.cos(train_x * 3.0),
        ],
        dim=-1,
    )

    model = KroneckerMultiTaskGP(train_x, train_y)
    model.eval()

    ref_point = train_y.min(dim=0).values - 0.1
    acquisition = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        X_baseline=train_x,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
        prune_baseline=False,
    )
    value = acquisition(torch.tensor([[[0.5]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_mixed_kronecker_multitask_gp_runtime_optimizes_scalarized_qlogei() -> None:
    train_x = torch.tensor(
        [
            [0.15, 0.0],
            [0.27, 1.0],
            [0.41, 0.0],
            [0.59, 1.0],
            [0.73, 0.0],
            [0.85, 1.0],
        ],
        dtype=torch.double,
    )
    continuous = train_x[:, :1]
    category = train_x[:, 1:2]
    train_y = torch.cat(
        [
            torch.sin(continuous * 3.0) + 0.1 * category,
            torch.cos(continuous * 3.0) - 0.1 * category,
        ],
        dim=-1,
    )
    model = MixedKroneckerMultiTaskGP(train_x, train_y, cat_dims=[1])
    model.eval()

    weights = torch.tensor([0.7, 0.3], dtype=torch.double)
    objective = GenericMCObjective(lambda samples, X=None: samples @ weights)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=(train_y @ weights).max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
        objective=objective,
    )
    candidate, value = optimize_acqf_mixed(
        acq_function=acquisition,
        bounds=torch.tensor([[0.1, 0.0], [0.9, 1.0]], dtype=torch.double),
        q=1,
        num_restarts=2,
        raw_samples=16,
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        options={"maxiter": 12},
    )

    assert candidate.shape == torch.Size([1, 2])
    assert 0.1 <= candidate[0, 0].item() <= 0.9
    assert candidate[0, 1].item() in {0.0, 1.0}
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()


def test_random_forest_runtime_supports_mc_acquisition() -> None:
    train_x = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    train_y = torch.sin(train_x * 3.0)

    model = RandomForestSurrogate(
        train_x,
        train_y,
        n_estimators=16,
        random_state=0,
    )
    model.fit()

    posterior = model.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))
    samples = posterior.rsample(torch.Size([4]))

    assert samples.shape == torch.Size([4, 2, 1])
    assert torch.isfinite(samples).all()

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=IndexSampler(sample_shape=torch.Size([8])),
    )
    value = acquisition(torch.tensor([[[0.5]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_single_task_gp_runtime_supports_knowledge_gradient_fantasies() -> None:
    train_x = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_y = torch.sin(train_x * 3.0)

    model = SingleTaskGP(train_x, train_y)
    model.eval()

    acquisition = qKnowledgeGradient(
        model=model,
        num_fantasies=4,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([4])),
    )
    candidate = torch.tensor([[[0.5]]], dtype=torch.double)
    fantasy_points = acquisition.get_augmented_q_batch_size(q=1)
    augmented_x = candidate.expand(1, fantasy_points, 1).clone()
    value = acquisition(augmented_x)

    assert fantasy_points == 5
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_single_task_multifidelity_gp_runtime_supports_mc_acquisition() -> None:
    design = torch.linspace(0.0, 1.0, 8, dtype=torch.double)
    fidelity = torch.tensor([0.25, 0.5, 0.75, 1.0] * 2, dtype=torch.double)
    train_x = torch.stack([design, fidelity], dim=-1)
    train_y = (torch.sin(design * 3.0) + 0.2 * fidelity).unsqueeze(-1)

    model = SingleTaskMultiFidelityGP(
        train_x,
        train_y,
        data_fidelities=[1],
        linear_truncated=False,
    )
    model.eval()

    test_x = torch.tensor([[0.25, 1.0], [0.75, 1.0]], dtype=torch.double)
    posterior = model.posterior(test_x)
    samples = posterior.rsample(torch.Size([4]))

    assert samples.shape == torch.Size([4, 2, 1])
    assert torch.isfinite(samples).all()

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
    )
    value = acquisition(torch.tensor([[[0.5, 1.0]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_mixed_single_task_multifidelity_gp_runtime_supports_mc_acquisition() -> None:
    design = torch.linspace(0.0, 1.0, 8, dtype=torch.double)
    category = torch.tensor([0.0, 1.0] * 4, dtype=torch.double)
    fidelity = torch.tensor([0.25, 0.5, 0.75, 1.0] * 2, dtype=torch.double)
    train_x = torch.stack([design, category, fidelity], dim=-1)
    train_y = (torch.sin(design * 3.0) + 0.15 * category + 0.2 * fidelity).unsqueeze(-1)

    model = MixedSingleTaskMultiFidelityGP(
        train_x,
        train_y,
        cat_dims=[1],
        data_fidelities=[2],
    )
    model.eval()

    test_x = torch.tensor([[0.25, 0.0, 1.0], [0.75, 1.0, 1.0]], dtype=torch.double)
    samples = model.posterior(test_x).rsample(torch.Size([4]))

    assert samples.shape == torch.Size([4, 2, 1])
    assert torch.isfinite(samples).all()

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
    )
    value = acquisition(torch.tensor([[[0.5, 1.0, 1.0]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def _assert_multifidelity_model_supports_knowledge_gradient(
    model,
    candidate: torch.Tensor,
) -> None:
    model.eval()
    acquisition = qKnowledgeGradient(
        model=model,
        num_fantasies=4,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([4])),
    )
    fantasy_points = acquisition.get_augmented_q_batch_size(q=1)
    value = acquisition(candidate.expand(1, fantasy_points, candidate.shape[-1]).clone())

    assert fantasy_points == 5
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_multifidelity_models_runtime_support_knowledge_gradient_fantasies() -> None:
    design = torch.linspace(0.0, 1.0, 8, dtype=torch.double)
    fidelity = torch.tensor([0.25, 0.5, 0.75, 1.0] * 2, dtype=torch.double)

    continuous_x = torch.stack([design, fidelity], dim=-1)
    continuous_y = (torch.sin(design * 3.0) + 0.2 * fidelity).unsqueeze(-1)
    continuous_model = SingleTaskMultiFidelityGP(
        continuous_x,
        continuous_y,
        data_fidelities=[1],
        linear_truncated=False,
    )
    _assert_multifidelity_model_supports_knowledge_gradient(
        continuous_model,
        torch.tensor([[0.5, 1.0]], dtype=torch.double),
    )

    category = torch.tensor([0.0, 1.0] * 4, dtype=torch.double)
    mixed_x = torch.stack([design, category, fidelity], dim=-1)
    mixed_y = (torch.sin(design * 3.0) + 0.15 * category + 0.2 * fidelity).unsqueeze(-1)
    mixed_model = MixedSingleTaskMultiFidelityGP(
        mixed_x,
        mixed_y,
        cat_dims=[1],
        data_fidelities=[2],
    )
    _assert_multifidelity_model_supports_knowledge_gradient(
        mixed_model,
        torch.tensor([[0.5, 1.0, 1.0]], dtype=torch.double),
    )


def test_multifidelity_projection_and_cost_utility_runtime() -> None:
    design = torch.linspace(0.0, 1.0, 8, dtype=torch.double)
    fidelity = torch.tensor([0.25, 0.5, 0.75, 1.0] * 2, dtype=torch.double)
    train_x = torch.stack([design, fidelity], dim=-1)
    train_y = (torch.sin(design * 3.0) + 0.2 * fidelity).unsqueeze(-1)

    model = SingleTaskMultiFidelityGP(
        train_x,
        train_y,
        data_fidelities=[1],
        linear_truncated=False,
    )
    model.eval()

    candidate = torch.tensor([[[0.5, 0.5]]], dtype=torch.double)
    projected = project_to_target_fidelity(
        candidate,
        d=candidate.shape[-1],
        target_fidelities={1: 1.0},
    )
    cost_model = AffineFidelityCostModel(fidelity_weights={1: 1.0}, fixed_cost=0.1)
    cost_utility = InverseCostWeightedUtility(cost_model=cost_model)

    assert projected.shape == candidate.shape
    assert torch.equal(projected[..., 0], candidate[..., 0])
    assert torch.all(projected[..., 1] == 1.0)

    deltas = torch.ones(torch.Size([4, 1, 1]), dtype=torch.double)
    weighted = cost_utility(candidate, deltas=deltas)

    assert weighted.shape == deltas.shape
    assert torch.isfinite(weighted).all()
    assert torch.all(weighted > 0)


def test_multifidelity_knowledge_gradient_runtime_with_cost_and_projection() -> None:
    design = torch.linspace(0.0, 1.0, 8, dtype=torch.double)
    fidelity = torch.tensor([0.25, 0.5, 0.75, 1.0] * 2, dtype=torch.double)
    train_x = torch.stack([design, fidelity], dim=-1)
    train_y = (torch.sin(design * 3.0) + 0.2 * fidelity).unsqueeze(-1)

    model = SingleTaskMultiFidelityGP(
        train_x,
        train_y,
        data_fidelities=[1],
        linear_truncated=False,
    )
    model.eval()

    target_fidelities = {1: 1.0}

    def project(x: torch.Tensor) -> torch.Tensor:
        return project_to_target_fidelity(
            x,
            d=x.shape[-1],
            target_fidelities=target_fidelities,
        )

    cost_model = AffineFidelityCostModel(fidelity_weights={1: 1.0}, fixed_cost=0.1)
    target_mean = PosteriorMean(model)
    _, current_value = optimize_acqf(
        acq_function=target_mean,
        bounds=torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.double),
        q=1,
        num_restarts=3,
        raw_samples=16,
        fixed_features={1: 1.0},
    )
    acquisition = qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=4,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([4])),
        cost_aware_utility=InverseCostWeightedUtility(cost_model=cost_model),
        current_value=current_value,
        project=project,
    )
    fantasy_points = acquisition.get_augmented_q_batch_size(q=1)
    candidate = torch.tensor([[0.5, 0.5]], dtype=torch.double)
    value = acquisition(candidate.expand(1, fantasy_points, 2).clone())

    assert fantasy_points == 5
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_mixed_multifidelity_knowledge_gradient_runtime() -> None:
    design = torch.linspace(0.0, 1.0, 8, dtype=torch.double)
    category = torch.tensor([0.0, 1.0] * 4, dtype=torch.double)
    fidelity = torch.tensor([0.25, 0.5, 0.75, 1.0] * 2, dtype=torch.double)
    train_x = torch.stack([design, category, fidelity], dim=-1)
    train_y = (torch.sin(design * 3.0) + 0.15 * category + 0.2 * fidelity).unsqueeze(-1)

    model = MixedSingleTaskMultiFidelityGP(
        train_x,
        train_y,
        cat_dims=[1],
        data_fidelities=[2],
    )
    model.eval()

    def project(x: torch.Tensor) -> torch.Tensor:
        return project_to_target_fidelity(
            x,
            d=x.shape[-1],
            target_fidelities={2: 1.0},
        )

    target_mean = PosteriorMean(model)
    current_values = []
    for category_value in (0.0, 1.0):
        _, value = optimize_acqf(
            acq_function=target_mean,
            bounds=torch.tensor(
                [[0.0, category_value, 1.0], [1.0, category_value, 1.0]],
                dtype=torch.double,
            ),
            q=1,
            num_restarts=3,
            raw_samples=16,
            fixed_features={1: category_value, 2: 1.0},
        )
        current_values.append(value)
    current_value = torch.stack(current_values).max()

    cost_model = AffineFidelityCostModel(fidelity_weights={2: 1.0}, fixed_cost=0.1)
    acquisition = qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=4,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([4])),
        cost_aware_utility=InverseCostWeightedUtility(cost_model=cost_model),
        current_value=current_value,
        project=project,
    )
    fantasy_points = acquisition.get_augmented_q_batch_size(q=1)
    augmented_batch = torch.tensor(
        [
            [0.5, 1.0, 0.5],
            [0.2, 0.0, 1.0],
            [0.4, 1.0, 1.0],
            [0.6, 0.0, 1.0],
            [0.8, 1.0, 1.0],
        ],
        dtype=torch.double,
    ).unsqueeze(0)
    value = acquisition(augmented_batch)

    assert fantasy_points == augmented_batch.shape[-2]
    assert augmented_batch[0, 0, 1].item() == 1.0
    assert augmented_batch[0, 1:, 1].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_pca_gp_runtime_supports_mc_acquisition() -> None:
    train_x = torch.stack(
        [
            torch.linspace(0.0, 1.0, 8, dtype=torch.double),
            torch.linspace(1.0, 0.0, 8, dtype=torch.double),
            torch.linspace(0.2, 0.9, 8, dtype=torch.double),
        ],
        dim=-1,
    )
    train_y = torch.sin(train_x[:, :1] * 3.0)

    model = PCAGP(train_x, train_y, n_components=2)
    model.eval()

    test_x = torch.tensor(
        [[0.25, 0.75, 0.4], [0.75, 0.25, 0.7]],
        dtype=torch.double,
    )
    posterior = model.posterior(test_x)
    samples = posterior.rsample(torch.Size([4]))

    assert samples.shape == torch.Size([4, 2, 1])
    assert torch.isfinite(samples).all()

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
    )
    value = acquisition(torch.tensor([[[0.5, 0.5, 0.55]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_mixed_pca_gp_runtime_supports_mc_acquisition() -> None:
    continuous_a = torch.linspace(0.0, 1.0, 8, dtype=torch.double)
    categorical = torch.tensor([0.0, 1.0] * 4, dtype=torch.double)
    continuous_b = torch.linspace(1.0, 0.0, 8, dtype=torch.double)
    train_x = torch.stack([continuous_a, categorical, continuous_b], dim=-1)
    train_y = (torch.sin(continuous_a * 3.0) + 0.1 * categorical).unsqueeze(-1)

    model = MixedPCAGP(train_x, train_y, n_components=1, cat_dims=[1])
    model.eval()

    test_x = torch.tensor(
        [[0.25, 0.0, 0.75], [0.75, 1.0, 0.25]],
        dtype=torch.double,
    )
    posterior = model.posterior(test_x)
    samples = posterior.rsample(torch.Size([4]))

    assert samples.shape == torch.Size([4, 2, 1])
    assert torch.isfinite(samples).all()

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
    )
    value = acquisition(torch.tensor([[[0.5, 1.0, 0.5]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_pca_gp_runtime_supports_knowledge_gradient() -> None:
    train_x = torch.stack(
        [
            torch.linspace(0.0, 1.0, 8, dtype=torch.double),
            torch.linspace(1.0, 0.0, 8, dtype=torch.double),
            torch.linspace(0.2, 0.9, 8, dtype=torch.double),
        ],
        dim=-1,
    )
    train_y = torch.sin(train_x[:, :1] * 3.0)
    model = PCAGP(train_x, train_y, n_components=2)
    model.eval()

    acquisition = qKnowledgeGradient(
        model=model,
        num_fantasies=4,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([4])),
    )
    candidate = torch.tensor([[[0.5, 0.5, 0.55]]], dtype=torch.double)
    fantasy_points = acquisition.get_augmented_q_batch_size(q=1)
    value = acquisition(candidate.expand(1, fantasy_points, 3).clone())

    assert fantasy_points == 5
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_mixed_pca_gp_runtime_supports_knowledge_gradient() -> None:
    continuous_a = torch.linspace(0.0, 1.0, 8, dtype=torch.double)
    categorical = torch.tensor([0.0, 1.0] * 4, dtype=torch.double)
    continuous_b = torch.linspace(1.0, 0.0, 8, dtype=torch.double)
    train_x = torch.stack([continuous_a, categorical, continuous_b], dim=-1)
    train_y = (torch.sin(continuous_a * 3.0) + 0.1 * categorical).unsqueeze(-1)
    model = MixedPCAGP(train_x, train_y, n_components=1, cat_dims=[1])
    model.eval()

    acquisition = qKnowledgeGradient(
        model=model,
        num_fantasies=4,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([4])),
    )
    candidate = torch.tensor([[[0.5, 1.0, 0.5]]], dtype=torch.double)
    fantasy_points = acquisition.get_augmented_q_batch_size(q=1)
    value = acquisition(candidate.expand(1, fantasy_points, 3).clone())

    assert fantasy_points == 5
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def _assert_reduced_model_supports_knowledge_gradient(model, candidate: torch.Tensor) -> None:
    model.eval()
    acquisition = qKnowledgeGradient(
        model=model,
        num_fantasies=4,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([4])),
    )
    fantasy_points = acquisition.get_augmented_q_batch_size(q=1)
    value = acquisition(candidate.expand(1, fantasy_points, candidate.shape[-1]).clone())

    assert fantasy_points == 5
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_remaining_continuous_reduced_models_support_knowledge_gradient() -> None:
    train_x = torch.stack(
        [
            torch.linspace(0.0, 1.0, 8, dtype=torch.double),
            torch.linspace(1.0, 0.0, 8, dtype=torch.double),
            torch.linspace(0.2, 0.9, 8, dtype=torch.double),
        ],
        dim=-1,
    )
    train_y = torch.sin(train_x[:, :1] * 3.0)
    candidate = torch.tensor([[[0.5, 0.5, 0.55]]], dtype=torch.double)

    models = (
        ReducedGP(train_x, train_y),
        PLSGP(train_x, train_y, n_components=1),
        RandomProjectionGP(train_x, train_y, n_components=2),
    )
    for model in models:
        _assert_reduced_model_supports_knowledge_gradient(model, candidate)


def test_remaining_mixed_reduced_models_support_knowledge_gradient() -> None:
    continuous_a = torch.linspace(0.0, 1.0, 8, dtype=torch.double)
    categorical = torch.tensor([0.0, 1.0] * 4, dtype=torch.double)
    continuous_b = torch.linspace(1.0, 0.0, 8, dtype=torch.double)
    train_x = torch.stack([continuous_a, categorical, continuous_b], dim=-1)
    train_y = (torch.sin(continuous_a * 3.0) + 0.1 * categorical).unsqueeze(-1)
    candidate = torch.tensor([[[0.5, 1.0, 0.5]]], dtype=torch.double)

    models = (
        MixedReducedGP(
            train_x,
            train_y,
            input_reducer=PCAInputReducer(n_components=1),
            cat_dims=[1],
        ),
        MixedPLSGP(train_x, train_y, n_components=1, cat_dims=[1]),
        MixedRandomProjectionGP(train_x, train_y, n_components=1, cat_dims=[1]),
    )
    for model in models:
        _assert_reduced_model_supports_knowledge_gradient(model, candidate)


def test_mixed_qkg_candidate_generation_runtime() -> None:
    train_x = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 1.0],
            [0.4, 0.0],
            [0.6, 1.0],
            [0.8, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_y = (torch.sin(train_x[:, :1] * 3.0) + 0.2 * train_x[:, 1:]).contiguous()
    model = MixedSingleTaskGP(train_x, train_y, cat_dims=[1])
    model.eval()
    acquisition = qKnowledgeGradient(
        model=model,
        num_fantasies=1,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1])),
    )

    candidate, value = optimize_mixed_one_shot_acqf(
        acquisition,
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        categorical_features={1: [0.0, 1.0]},
        num_restarts=2,
        raw_samples=8,
        max_assignments=4,
    )

    assert candidate.shape == torch.Size([1, 2])
    assert candidate[0, 1].item() in {0.0, 1.0}
    assert torch.isfinite(value).all()


def test_mixed_qmfkg_candidate_generation_runtime() -> None:
    design = torch.linspace(0.0, 1.0, 8, dtype=torch.double)
    category = torch.tensor([0.0, 1.0] * 4, dtype=torch.double)
    fidelity = torch.tensor([0.25, 0.5, 0.75, 1.0] * 2, dtype=torch.double)
    train_x = torch.stack([design, category, fidelity], dim=-1)
    train_y = (torch.sin(design * 3.0) + 0.15 * category + 0.2 * fidelity).unsqueeze(-1)
    model = MixedSingleTaskMultiFidelityGP(
        train_x,
        train_y,
        cat_dims=[1],
        data_fidelities=[2],
    )
    model.eval()

    def project(x: torch.Tensor) -> torch.Tensor:
        return project_to_target_fidelity(
            x,
            d=x.shape[-1],
            target_fidelities={2: 1.0},
        )

    target_mean = PosteriorMean(model)
    current_values = []
    for category_value in (0.0, 1.0):
        _, current = optimize_acqf(
            acq_function=target_mean,
            bounds=torch.tensor(
                [[0.0, category_value, 1.0], [1.0, category_value, 1.0]],
                dtype=torch.double,
            ),
            q=1,
            num_restarts=2,
            raw_samples=8,
            fixed_features={1: category_value, 2: 1.0},
        )
        current_values.append(current)
    current_value = torch.stack(current_values).max()
    cost_model = AffineFidelityCostModel(fidelity_weights={2: 1.0}, fixed_cost=0.1)
    acquisition = qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=1,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1])),
        cost_aware_utility=InverseCostWeightedUtility(cost_model=cost_model),
        current_value=current_value,
        project=project,
    )

    candidate, value = optimize_mixed_one_shot_acqf(
        acquisition,
        torch.tensor([[0.0, 0.0, 0.25], [1.0, 1.0, 1.0]], dtype=torch.double),
        categorical_features={1: [0.0, 1.0]},
        num_restarts=2,
        raw_samples=8,
        max_assignments=4,
    )

    assert candidate.shape == torch.Size([1, 3])
    assert candidate[0, 1].item() in {0.0, 1.0}
    assert 0.25 <= candidate[0, 2].item() <= 1.0
    assert torch.isfinite(value).all()
