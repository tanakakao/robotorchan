"""End-to-end candidate proposals using masked Gaussian-process acquisitions."""

import pytest
import torch
from botorch.acquisition.acquisition import OneShotAcquisitionFunction
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models import ModelListGP, SingleTaskGP, SingleTaskMultiFidelityGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.optim import optimize_acqf
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume

from robotorchan.acquisition import create_acquisition
from robotorchan.optim import optimize_acqf_decoupled


@pytest.mark.parametrize("kind", ["kg", "constrained", "epig", "hvkg", "mfhvkg"])
def test_decoupled_gp_candidate_proposal(kind):
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
    q_aug = acq.get_augmented_q_batch_size(1) if isinstance(acq, OneShotAcquisitionFunction) else 1
    generator = torch.Generator().manual_seed(42)
    initial = torch.rand(2, q_aug, X.shape[-1], generator=generator, dtype=X.dtype)
    masks = torch.tensor([[True, False], [False, True], [True, True]])
    options = {"maxiter": 5}
    candidate, mask, value = optimize_acqf_decoupled(
        acq,
        bounds,
        masks,
        num_restarts=2,
        raw_samples=8,
        options=options,
        batch_initial_conditions=initial,
    )
    assert candidate.shape == (1, X.shape[-1])
    assert mask.shape == (1, 2)
    assert (candidate >= bounds[0]).all() and (candidate <= bounds[1]).all()
    assert (masks == mask).all(dim=-1).any()
    assert torch.isfinite(value).all()
    assert torch.equal(acq.X_evaluation_mask, original)

    # A single allowed all-output mask must agree with direct BoTorch optimization.
    full_candidate, full_mask, full_value = optimize_acqf_decoupled(
        acq,
        bounds,
        masks[-1:],
        num_restarts=2,
        raw_samples=8,
        options=options,
        batch_initial_conditions=initial,
    )
    acq.X_evaluation_mask = masks[-1:]
    direct_candidate, direct_value = optimize_acqf(
        acq,
        bounds,
        q=1,
        num_restarts=2,
        raw_samples=8,
        options=options,
        batch_initial_conditions=initial,
    )
    assert torch.equal(full_mask, masks[-1:])
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
