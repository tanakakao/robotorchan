"""End-to-end candidate proposals using masked Gaussian-process acquisitions."""

import pytest
import torch
from botorch.acquisition.acquisition import OneShotAcquisitionFunction
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, ModelListGP, SingleTaskGP, SingleTaskMultiFidelityGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.optim import optimize_acqf
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.acquisition import create_acquisition
from robotorchan.optim import optimize_acqf_decoupled
from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.domains.variable_space import MixedVariableSpace


@pytest.mark.parametrize("kind", ["kg", "constrained", "epig", "hvkg", "mfhvkg"])
@pytest.mark.parametrize("q", [1, 2])
def test_decoupled_gp_candidate_proposal(kind, q):
    x = torch.linspace(0, 1, 6, dtype=torch.double)
    X = x.unsqueeze(-1)
    if kind == "mfhvkg":
        fidelity = torch.arange(6, dtype=x.dtype) % 2
        X = torch.stack((x, fidelity), dim=-1)
        model = ModelListGP(
            SingleTaskMultiFidelityGP(X, (x.sin() + fidelity).unsqueeze(-1), data_fidelities=[1]),
            SingleTaskMultiFidelityGP(X, (x.cos() + fidelity).unsqueeze(-1), data_fidelities=[1]),
        )
    else:
        model = ModelListGP(SingleTaskGP(X, 1 + X.sin()), SingleTaskGP(X, X - 0.5))
    model.eval()
    with torch.no_grad():
        model.posterior(X[:1])
    original = torch.tensor([[True, False]])
    common = dict(model=model, X_evaluation_mask=original)
    if kind in ("kg", "constrained"):
        extra = {}
        if kind == "constrained":
            extra = dict(
                constraint_indices=(1,),
                constraint_thresholds=X.new_tensor([0.0]),
                infeasible_utility=0.0,
            )
        acq = create_acquisition(
            "qDecoupledKnowledgeGradient",
            **common,
            decision_X=X,
            weights=X.new_tensor([1.0, 0.0]),
            sampler=ListSampler(
                SobolQMCNormalSampler(torch.Size([4]), seed=42),
                SobolQMCNormalSampler(torch.Size([4]), seed=43),
            ),
            **extra,
        )
    elif kind == "epig":
        acq = create_acquisition(
            "qDecoupledExpectedPredictiveInformationGain",
            **common,
            target_X=X,
            observation_noise=X.new_tensor([0.01, 0.02]),
        )
    else:
        extra = {}
        name = "qHypervolumeKnowledgeGradient"
        if kind == "mfhvkg":
            name = "qMultiFidelityHypervolumeKnowledgeGradient"

            def project(points):
                return project_to_target_fidelity(points, d=2, target_fidelities={1: 1.0})

            extra = dict(target_fidelities={1: 1.0}, project=project)
        acq = create_acquisition(
            name,
            **common,
            ref_point=X.new_tensor([-1.0, -1.0]),
            num_fantasies=2,
            num_pareto=2,
            **extra,
        )
    bounds = torch.stack((X.new_zeros(X.shape[-1]), X.new_ones(X.shape[-1])))
    q_aug = acq.get_augmented_q_batch_size(q) if isinstance(acq, OneShotAcquisitionFunction) else q
    generator = torch.Generator().manual_seed(42)
    initial = torch.rand(2, q_aug, X.shape[-1], generator=generator, dtype=X.dtype)
    masks = torch.tensor([[True, False], [False, True], [True, True]])
    options = {"maxiter": 5}
    candidate, mask, value = optimize_acqf_decoupled(
        acq,
        bounds,
        masks,
        q=q,
        num_restarts=2,
        raw_samples=8,
        options=options,
        batch_initial_conditions=initial,
    )
    assert candidate.shape == (q, X.shape[-1])
    assert mask.shape == (q, 2)
    assert (candidate >= bounds[0]).all() and (candidate <= bounds[1]).all()
    assert (masks.unsqueeze(0) == mask.unsqueeze(1)).all(dim=-1).any(dim=-1).all()
    assert torch.isfinite(value).all()
    assert torch.equal(acq.X_evaluation_mask, original)

    # A single allowed all-output mask must agree with direct BoTorch optimization.
    full_candidate, full_mask, full_value = optimize_acqf_decoupled(
        acq,
        bounds,
        masks[-1:],
        q=q,
        num_restarts=2,
        raw_samples=8,
        options=options,
        batch_initial_conditions=initial,
    )
    acq.X_evaluation_mask = masks[-1:].expand(q, -1).clone()
    direct_candidate, direct_value = optimize_acqf(
        acq,
        bounds,
        q=q,
        num_restarts=2,
        raw_samples=8,
        options=options,
        batch_initial_conditions=initial,
    )
    assert torch.equal(full_mask, masks[-1:].expand(q, -1))
    torch.testing.assert_close(full_candidate, direct_candidate)
    torch.testing.assert_close(full_value, direct_value)


@pytest.mark.parametrize("multifidelity", [False, True])
def test_native_cost_aware_decoupled_one_shot(multifidelity):
    x = torch.linspace(0, 1, 6, dtype=torch.double)
    X = x.unsqueeze(-1)
    if multifidelity:
        X = torch.stack((x, torch.arange(6, dtype=x.dtype) % 2), dim=-1)
        model = ModelListGP(
            SingleTaskMultiFidelityGP(X, x.sin().unsqueeze(-1), data_fidelities=[1]),
            SingleTaskMultiFidelityGP(X, x.cos().unsqueeze(-1), data_fidelities=[1]),
        )
    else:
        model = ModelListGP(SingleTaskGP(X, X.sin()), SingleTaskGP(X, X.cos()))
    model.eval()
    ref = X.new_tensor([-1.0, -1.0])
    extra = {}
    name = "qHypervolumeKnowledgeGradient"
    target = X
    if multifidelity:
        name = "qMultiFidelityHypervolumeKnowledgeGradient"

        def project(points):
            return project_to_target_fidelity(points, d=2, target_fidelities={1: 1.0})

        extra = dict(target_fidelities={1: 1.0}, project=project)
        target = project(X)
    with torch.no_grad():
        current_value = X.new_tensor(Hypervolume(ref).compute(model.posterior(target).mean))

    def costs(points):
        # Native utility must receive actual candidates, not augmented one-shot rows.
        assert points.shape[-2] == 1
        coordinate = points[..., -1:]
        return torch.cat((1 + coordinate, 3 + 2 * coordinate), dim=-1)

    cost_model = GenericDeterministicModel(costs, num_outputs=2)
    utility = InverseCostWeightedUtility(cost_model=cost_model, use_mean=True)
    masks = torch.tensor([[True, False], [False, True], [True, True]])
    acq = create_acquisition(
        name,
        model=model,
        ref_point=ref,
        num_fantasies=2,
        num_pareto=2,
        X_evaluation_mask=masks[:1],
        current_value=current_value,
        cost_aware_utility=utility,
        **extra,
    )
    bounds = torch.stack((X.new_zeros(X.shape[-1]), X.new_ones(X.shape[-1])))
    initial = torch.rand(
        2,
        acq.get_augmented_q_batch_size(1),
        X.shape[-1],
        generator=torch.Generator().manual_seed(42),
        dtype=X.dtype,
    )
    candidate, mask, score = optimize_acqf_decoupled(
        acq,
        bounds,
        masks,
        num_restarts=2,
        raw_samples=8,
        options={"maxiter": 5},
        batch_initial_conditions=initial,
    )
    assert candidate.shape == (1, X.shape[-1])
    assert (masks == mask).all(dim=-1).any()
    assert torch.isfinite(score).all()
    assert torch.equal(acq.X_evaluation_mask, masks[:1])
    with pytest.raises(ValueError, match="already has a cost-aware utility"):
        optimize_acqf_decoupled(acq, bounds, masks, output_costs=X.new_tensor([1.0, 2.0]))


def test_learned_log_cost_decoupled_epig():
    X = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
    model = ModelListGP(SingleTaskGP(X, X.sin()), SingleTaskGP(X, X.cos()))
    cost_model = SingleTaskGP(X, torch.cat((1 + X, 2 + 3 * X), dim=-1).log())
    model.eval()
    fit_gpytorch_mll(ExactMarginalLogLikelihood(cost_model.likelihood, cost_model))
    cost_model.eval()
    masks = torch.tensor([[True, False], [False, True], [True, True]])
    acq = create_acquisition(
        "qDecoupledExpectedPredictiveInformationGain",
        model=model,
        target_X=X,
        observation_noise=X.new_tensor([0.01, 0.02]),
        X_evaluation_mask=masks[:1],
    )

    def cost(points, mask):
        posterior = cost_model.posterior(points)
        expected = (posterior.mean + 0.5 * posterior.variance).exp()
        return (expected * mask.to(points)).sum(dim=(-2, -1))

    candidate, mask, score = optimize_acqf_decoupled(
        acq,
        X.new_tensor([[0.0], [1.0]]),
        masks,
        cost_function=cost,
        num_restarts=2,
        raw_samples=8,
        options={"maxiter": 5, "seed": 42},
    )
    assert candidate.shape == (1, 1)
    assert (masks == mask).all(dim=-1).any()
    assert torch.isfinite(score)
    assert cost(candidate.unsqueeze(0), mask).item() > 0


@pytest.mark.parametrize("kind", ["kg", "epig", "hvkg"])
@pytest.mark.parametrize("q, strategy", [(1, "joint"), (2, "joint"), (2, "sequential")])
def test_mixed_gp_decoupled_candidate(kind, q, strategy):
    if kind == "hvkg" and strategy == "sequential":
        pytest.skip("one-shot sequential is not supported")
    torch.manual_seed(42)
    x = torch.linspace(0, 1, 6, dtype=torch.double)
    X = torch.stack((x, torch.arange(6, dtype=x.dtype) % 2), dim=-1)
    model = ModelListGP(
        MixedSingleTaskGP(X, (x.sin() + X[:, 1]).unsqueeze(-1), cat_dims=[1]),
        MixedSingleTaskGP(X, (x.cos() - X[:, 1]).unsqueeze(-1), cat_dims=[1]),
    )
    model.eval()
    with torch.no_grad():
        model.posterior(X[:1])
    masks = torch.tensor([[True, False], [False, True]])
    original = masks[:1]
    common = dict(model=model, X_evaluation_mask=original)
    if kind == "kg":
        acq = create_acquisition(
            "qDecoupledKnowledgeGradient",
            **common,
            decision_X=X,
            weights=X.new_tensor([1.0, 1.0]),
            sampler=ListSampler(
                SobolQMCNormalSampler(torch.Size([2]), seed=42),
                SobolQMCNormalSampler(torch.Size([2]), seed=43),
            ),
        )
    elif kind == "epig":
        acq = create_acquisition(
            "qDecoupledExpectedPredictiveInformationGain",
            **common,
            target_X=X,
            observation_noise=X.new_tensor([0.01, 0.02]),
        )
    else:
        acq = create_acquisition(
            "qHypervolumeKnowledgeGradient",
            **common,
            ref_point=X.new_tensor([-2.0, -2.0]),
            num_fantasies=1,
            num_pareto=1,
        )
    bounds = X.new_tensor([[0.0, 0.0], [1.0, 1.0]])
    space = MixedVariableSpace(bounds, categorical_values={1: (0.0, 1.0)})
    candidate, mask, value = optimize_acqf_decoupled(
        acq,
        bounds,
        masks,
        variable_space=space,
        q=q,
        strategy=strategy,
        num_restarts=2,
        raw_samples=8,
    )
    assert candidate.shape == (q, 2)
    assert ((candidate[:, 1] == 0) | (candidate[:, 1] == 1)).all()
    assert (candidate >= bounds[0]).all() and (candidate <= bounds[1]).all()
    assert (masks.unsqueeze(0) == mask.unsqueeze(1)).all(dim=-1).any(dim=-1).all()
    assert torch.isfinite(value).all()
    assert acq.X_evaluation_mask is original


@pytest.mark.parametrize("kind", ["kg", "epig", "hvkg"])
@pytest.mark.parametrize("discrete", [False, True])
def test_target_fidelity_decoupled_proposal(kind, discrete):
    torch.manual_seed(42)
    x = torch.linspace(0, 1, 6, dtype=torch.double)
    fidelity = torch.arange(6, dtype=x.dtype) % 2
    X = torch.stack((x, fidelity), dim=-1)
    model = ModelListGP(
        SingleTaskMultiFidelityGP(X, (x.sin() + fidelity).unsqueeze(-1), data_fidelities=[1]),
        SingleTaskMultiFidelityGP(X, (x.cos() + fidelity).unsqueeze(-1), data_fidelities=[1]),
    )
    model.eval()
    with torch.no_grad():
        model.posterior(X[:1])
    masks = torch.tensor([[True, False], [False, True]])
    target = {1: 1.0}
    common = dict(model=model, X_evaluation_mask=masks[:1], target_fidelities=target)

    def project(points):
        return project_to_target_fidelity(points, d=2, target_fidelities=target)

    def costs(points):
        fidelity = points[..., 1:2]
        return torch.cat((1 + fidelity, 2 + 3 * fidelity), dim=-1)

    def cost(points, mask):
        return (costs(points) * mask.to(points)).sum(dim=(-2, -1))

    extra = {}
    if kind == "kg":
        acq = create_acquisition(
            "qDecoupledKnowledgeGradient",
            **common,
            decision_X=X,
            weights=X.new_tensor([1.0, 1.0]),
            sampler=ListSampler(
                SobolQMCNormalSampler(torch.Size([2]), seed=42),
                SobolQMCNormalSampler(torch.Size([2]), seed=43),
            ),
        )
        torch.testing.assert_close(acq.decision_X, project(X))
        extra["cost_function"] = cost
    elif kind == "epig":
        acq = create_acquisition(
            "qDecoupledExpectedPredictiveInformationGain",
            **common,
            target_X=X,
            observation_noise=X.new_tensor([0.01, 0.02]),
        )
        extra["cost_function"] = cost
    else:
        ref = X.new_tensor([-2.0, -2.0])
        with torch.no_grad():
            current = X.new_tensor(Hypervolume(ref).compute(model.posterior(project(X)).mean))
        acq = create_acquisition(
            "qMultiFidelityHypervolumeKnowledgeGradient",
            **common,
            project=project,
            ref_point=ref,
            num_fantasies=1,
            num_pareto=1,
            current_value=current,
            cost_aware_utility=InverseCostWeightedUtility(
                cost_model=GenericDeterministicModel(costs, num_outputs=2), use_mean=True
            ),
        )
    bounds = X.new_tensor([[0.0, 0.0], [1.0, 1.0]])
    space = MixedVariableSpace(bounds, categorical_values={1: (0.0, 1.0)}) if discrete else None
    candidate, mask, score = optimize_acqf_decoupled(
        acq,
        bounds,
        masks,
        variable_space=space,
        num_restarts=2,
        raw_samples=8,
        **extra,
    )
    assert candidate.shape == (1, 2)
    assert (candidate >= bounds[0]).all() and (candidate <= bounds[1]).all()
    if discrete:
        assert candidate[0, 1].item() in (0.0, 1.0)
    assert (masks == mask).all(dim=-1).any()
    assert torch.isfinite(score)
    assert cost(candidate, mask) > 0


@pytest.mark.parametrize("kind", ["kg", "epig"])
def test_fidelity_projection_equivalence_and_output_specific_batch(kind):
    x = torch.linspace(0, 1, 6, dtype=torch.double)
    X = torch.stack((x, torch.arange(6, dtype=x.dtype) % 2), dim=-1)
    model = ModelListGP(
        SingleTaskMultiFidelityGP(X, x.sin().unsqueeze(-1), data_fidelities=[1]),
        SingleTaskMultiFidelityGP(X, x.cos().unsqueeze(-1), data_fidelities=[1]),
    )
    model.eval()
    with torch.no_grad():
        model.posterior(X[:1])
    masks = torch.tensor([[True, False], [False, True]])
    target = project_to_target_fidelity(X, d=2, target_fidelities={1: 1.0})

    def make(points, projection):
        common = dict(model=model, X_evaluation_mask=masks, target_fidelities=projection)
        if kind == "kg":
            return create_acquisition(
                "qDecoupledKnowledgeGradient",
                **common,
                decision_X=points,
                weights=X.new_tensor([1.0, 1.0]),
                sampler=ListSampler(
                    SobolQMCNormalSampler(torch.Size([4]), seed=42),
                    SobolQMCNormalSampler(torch.Size([4]), seed=43),
                ),
            )
        return create_acquisition(
            "qDecoupledExpectedPredictiveInformationGain",
            **common,
            target_X=points,
            observation_noise=X.new_tensor([0.01, 0.02]),
        )

    projected = make(X, {1: 1.0})
    explicit = make(target, None)
    # Same design, different output and fidelity on each actual candidate row.
    batch = X.new_tensor([[[0.4, 0.0], [0.4, 1.0]]]).requires_grad_()
    value = projected(batch)
    torch.testing.assert_close(value, explicit(batch))
    assert torch.isfinite(torch.autograd.grad(value.sum(), batch)[0]).all()

    bounds = X.new_tensor([[0.0, 0.0], [1.0, 1.0]])
    full = torch.ones(1, 2, dtype=torch.bool)
    initial = X.new_tensor([[[0.2, 1.0], [0.8, 1.0]], [[0.4, 1.0], [0.6, 1.0]]])
    candidate, mask, score = optimize_acqf_decoupled(
        projected,
        bounds,
        full,
        q=2,
        fixed_features={1: 1.0},
        batch_initial_conditions=initial,
        num_restarts=2,
        raw_samples=8,
        options={"maxiter": 5},
    )
    explicit.X_evaluation_mask = full.expand(2, -1)
    direct, direct_score = optimize_acqf(
        explicit,
        bounds,
        q=2,
        fixed_features={1: 1.0},
        batch_initial_conditions=initial,
        num_restarts=2,
        raw_samples=8,
        options={"maxiter": 5},
    )
    assert mask.all()
    assert (candidate[:, 1] == 1).all()
    torch.testing.assert_close(candidate, direct)
    torch.testing.assert_close(score, direct_score)


def test_fidelity_dependent_noise_and_shared_design():
    x = torch.linspace(0, 1, 6, dtype=torch.double)
    X = torch.stack((x, torch.arange(6, dtype=x.dtype) % 2), dim=-1)
    model = ModelListGP(
        SingleTaskMultiFidelityGP(X, x.sin().unsqueeze(-1), data_fidelities=[1]),
        SingleTaskMultiFidelityGP(X, x.cos().unsqueeze(-1), data_fidelities=[1]),
    )
    model.eval()
    masks = torch.tensor([[True, False], [False, True]])

    def noise(points):
        return 0.01 + 0.1 * (1 - points[..., 1])

    def constant(points):
        return points.new_full(points.shape[:-1], 0.01)

    common = dict(
        model=model,
        target_X=X,
        target_fidelities={1: 1.0},
        X_evaluation_mask=masks,
        observation_noise=X.new_tensor([0.01, 0.01]),
    )
    baseline = create_acquisition("qDecoupledExpectedPredictiveInformationGain", **common)
    acq = create_acquisition(
        "qDecoupledExpectedPredictiveInformationGain",
        **common,
        candidate_noise_functions=[noise, noise],
        target_noise_functions=[constant, constant],
    )
    high = X.new_tensor([[[0.4, 1.0], [0.4, 1.0]]])
    low = X.new_tensor([[[0.4, 0.0], [0.4, 1.0]]]).requires_grad_()
    torch.testing.assert_close(acq(high), baseline(high))
    assert (acq(low) <= baseline(low) + 1e-10).all()
    assert torch.isfinite(torch.autograd.grad(acq(low).sum(), low)[0]).all()

    constraints = CandidateConstraints(
        equality_constraints=((torch.tensor([[0, 0], [1, 0]]), X.new_tensor([1.0, -1.0]), 0.0),),
    )
    candidate, selected, score = optimize_acqf_decoupled(
        acq,
        X.new_tensor([[0.0, 0.0], [1.0, 1.0]]),
        masks,
        q=2,
        constraints=constraints,
        mask_constraint=lambda mask: bool(torch.equal(mask, masks)),
        batch_initial_conditions=X.new_tensor([[[0.2, 0.0], [0.2, 1.0]], [[0.8, 0.0], [0.8, 1.0]]]),
        num_restarts=2,
        raw_samples=8,
        options={"maxiter": 10},
    )
    torch.testing.assert_close(candidate[0, 0], candidate[1, 0], atol=1e-6, rtol=0)
    assert torch.equal(selected, masks)
    assert torch.isfinite(score)


@pytest.mark.parametrize("kind", ["epig", "hvkg"])
@pytest.mark.parametrize("discrete", [False, True])
def test_shared_design_fidelity_linear_constraints(kind, discrete):
    torch.manual_seed(42)
    x = torch.linspace(0, 1, 6, dtype=torch.double)
    X = torch.stack((x, torch.arange(6, dtype=x.dtype) % 2), dim=-1)
    model = ModelListGP(
        SingleTaskMultiFidelityGP(X, x.sin().unsqueeze(-1), data_fidelities=[1]),
        SingleTaskMultiFidelityGP(X, x.cos().unsqueeze(-1), data_fidelities=[1]),
    )
    model.eval()
    masks = torch.tensor([[True, False], [False, True]])
    common = dict(model=model, X_evaluation_mask=masks, target_fidelities={1: 1.0})
    if kind == "epig":
        acq = create_acquisition(
            "qDecoupledExpectedPredictiveInformationGain",
            **common,
            target_X=X,
            observation_noise=X.new_tensor([0.01, 0.02]),
        )
    else:
        acq = create_acquisition(
            "qMultiFidelityHypervolumeKnowledgeGradient",
            **common,
            ref_point=X.new_tensor([-2.0, -2.0]),
            num_fantasies=1,
            num_pareto=1,
        )
    bounds = X.new_tensor([[0.0, 0.0], [1.0, 1.0]])
    space = MixedVariableSpace(bounds, categorical_values={1: (0.0, 1.0)}) if discrete else None
    constraints = CandidateConstraints(
        equality_constraints=((torch.tensor([[0, 0], [1, 0]]), X.new_tensor([1.0, -1.0]), 0.0),),
    )
    candidate, mask, score = optimize_acqf_decoupled(
        acq,
        bounds,
        masks,
        q=2,
        variable_space=space,
        constraints=constraints,
        mask_constraint=lambda mask: bool(torch.equal(mask, masks)),
        num_restarts=2,
        raw_samples=16,
    )
    torch.testing.assert_close(candidate[0, 0], candidate[1, 0], atol=1e-5, rtol=0)
    assert torch.equal(mask, masks)
    assert torch.isfinite(score)
    if discrete:
        assert ((candidate[:, 1] == 0) | (candidate[:, 1] == 1)).all()
