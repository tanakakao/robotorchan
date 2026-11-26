"""Candidate and mask enumeration contracts."""

from types import SimpleNamespace

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import create_optimizer, optimize_acqf_decoupled
from robotorchan.optim.domains.variable_space import MixedVariableSpace
from robotorchan.workflow.cost_budget import CostBudget


class MaskedQuadratic(AcquisitionFunction):
    def __init__(self):
        super().__init__(model=SimpleNamespace(num_outputs=2))
        self.X_evaluation_mask = torch.tensor([[True, False]])

    def forward(self, X):
        second = self.X_evaluation_mask[0, 1]
        center = 0.75 if second else 0.25
        return -(X[..., 0, 0] - center).square() + second.to(X)


def test_decoupled_optimizer_selects_mask_and_restores_state():
    acq = MaskedQuadratic()
    original = acq.X_evaluation_mask
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    masks = torch.tensor([[True, False], [False, True]])
    optimize = create_optimizer("optimize_acqf_decoupled", num_restarts=2, raw_samples=16)
    candidate, selected, value = optimize(acq, bounds, masks)
    torch.testing.assert_close(candidate, bounds.new_tensor([[0.75]]), atol=1e-4, rtol=0)
    assert torch.equal(selected, masks[1:2])
    torch.testing.assert_close(value, bounds.new_tensor(1.0))
    assert acq.X_evaluation_mask is original


def test_decoupled_optimizer_restores_state_on_failure(monkeypatch):
    acq = MaskedQuadratic()
    original = acq.X_evaluation_mask

    def fail(**kwargs):
        raise RuntimeError("optimization failed")

    monkeypatch.setattr("robotorchan.optim.decoupled.optimize_acqf", fail)
    with pytest.raises(RuntimeError, match="optimization failed"):
        optimize_acqf_decoupled(acq, torch.tensor([[0.0], [1.0]]), torch.tensor([[False, True]]))
    assert acq.X_evaluation_mask is original


def test_decoupled_optimizer_rejects_empty_mask():
    with pytest.raises(ValueError, match="nonempty"):
        optimize_acqf_decoupled(
            MaskedQuadratic(), torch.tensor([[0.0], [1.0]]), torch.tensor([[False, False]])
        )


def test_fixed_cost_and_budget_mask_selection(monkeypatch):
    acq = MaskedQuadratic()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    masks = torch.tensor([[True, False], [False, True], [True, True]])

    def optimize(**kwargs):
        mask = kwargs["acq_function"].X_evaluation_mask
        gain = 6.0 if mask[0, 1] else 2.0
        return bounds.new_tensor([[0.5]]), bounds.new_tensor(gain)

    monkeypatch.setattr("robotorchan.optim.decoupled.optimize_acqf", optimize)
    budget = CostBudget(total=5.0)
    budget.reserve("pending", 2.0)
    _, mask, score = optimize_acqf_decoupled(
        acq, bounds, masks, output_costs=bounds.new_tensor([1.0, 4.0]), budget=budget
    )
    assert torch.equal(mask, masks[:1])
    torch.testing.assert_close(score, bounds.new_tensor(2.0))
    assert budget.remaining == 3.0 and budget.spent == 0.0
    _, mask, score = optimize_acqf_decoupled(
        acq, bounds, masks, mask_costs=bounds.new_tensor([2.0, 4.0, 1.0])
    )
    assert torch.equal(mask, masks[-1:])
    torch.testing.assert_close(score, bounds.new_tensor(6.0))


def test_no_affordable_mask():
    bounds = torch.tensor([[0.0], [1.0]])
    with pytest.raises(ValueError, match="no affordable"):
        optimize_acqf_decoupled(
            MaskedQuadratic(),
            bounds,
            torch.tensor([[True, False]]),
            output_costs=bounds.new_tensor([2.0, 3.0]),
            budget=CostBudget(total=1.0),
        )


def test_negative_gain_cost_ranking(monkeypatch):
    bounds = torch.tensor([[0.0], [1.0]])
    masks = torch.tensor([[True, False], [False, True]])

    def optimize(**kwargs):
        return bounds.new_tensor([[0.5]]), bounds.new_tensor(-1.0)

    monkeypatch.setattr("robotorchan.optim.decoupled.optimize_acqf", optimize)
    _, mask, score = optimize_acqf_decoupled(
        MaskedQuadratic(), bounds, masks, output_costs=bounds.new_tensor([1.0, 4.0])
    )
    assert torch.equal(mask, masks[:1])
    torch.testing.assert_close(score, bounds.new_tensor(-1.0))


def test_input_cost_changes_continuous_candidate():
    acq = MaskedQuadratic()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    masks = torch.tensor([[False, True]])

    def cost(X, mask):
        return 1 + 4 * X[..., 0, 0]

    candidate, mask, score = optimize_acqf_decoupled(
        acq, bounds, masks, cost_function=cost, num_restarts=3, raw_samples=32
    )
    assert candidate.item() < 0.5
    assert torch.equal(mask, masks)
    acq.X_evaluation_mask = mask
    expected = acq(candidate) / cost(candidate, mask)
    torch.testing.assert_close(score, expected)


def test_input_cost_budget_requires_initial_conditions():
    bounds = torch.tensor([[0.0], [1.0]])
    with pytest.raises(ValueError, match="batch_initial_conditions"):
        optimize_acqf_decoupled(
            MaskedQuadratic(),
            bounds,
            torch.tensor([[True, False]]),
            cost_function=lambda X, mask: X[..., 0, 0] + 1,
            budget=CostBudget(total=1.0),
        )


def test_input_cost_budget_constrains_candidate():
    acq = MaskedQuadratic()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    masks = torch.tensor([[False, True]])
    budget = CostBudget(total=1.1)

    def cost(X, mask):
        return 1 + X[..., 0, 0]

    candidate, mask, score = optimize_acqf_decoupled(
        acq,
        bounds,
        masks,
        cost_function=cost,
        budget=budget,
        batch_initial_conditions=bounds.new_tensor([[[0.0]], [[0.05]], [[0.8]]]),
        num_restarts=3,
    )
    assert budget.can_afford(cost(candidate.unsqueeze(0), mask).item())
    assert torch.equal(mask, masks)
    assert torch.isfinite(score)
    assert budget.spent == 0


def test_budget_boundary_falls_back_to_feasible_restart(monkeypatch):
    acq = MaskedQuadratic()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    masks = torch.tensor([[False, True]])
    budget = CostBudget(total=1.1)

    def cost(X, mask):
        return 1 + X[..., 0, 0]

    def optimize(**kwargs):
        return bounds.new_tensor([[0.10000001]]), bounds.new_tensor(1.0)

    monkeypatch.setattr("robotorchan.optim.decoupled.optimize_acqf", optimize)
    candidate, mask, score = optimize_acqf_decoupled(
        acq,
        bounds,
        masks,
        cost_function=cost,
        budget=budget,
        batch_initial_conditions=bounds.new_tensor([[[0.0]], [[0.05]]]),
        num_restarts=2,
    )
    assert budget.can_afford(cost(candidate.unsqueeze(0), mask).item())
    torch.testing.assert_close(candidate, bounds.new_tensor([[0.05]]))
    acq.X_evaluation_mask = mask
    torch.testing.assert_close(score, acq(candidate) / cost(candidate, mask))


def test_input_cost_budget_skips_unaffordable_masks():
    acq = MaskedQuadratic()
    original = acq.X_evaluation_mask
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    masks = torch.tensor([[True, False], [False, True]])
    budget = CostBudget(total=2.0)
    budget.reserve("pending", 0.5)

    def cost(X, mask):
        return 1 + X[..., 0, 0] + 2 * mask[0, 1].to(X)

    candidate, mask, score = optimize_acqf_decoupled(
        acq,
        bounds,
        masks,
        cost_function=cost,
        budget=budget,
        batch_initial_conditions=bounds.new_tensor([[[0.0]], [[0.2]]]),
        num_restarts=2,
    )
    assert torch.equal(mask, masks[:1])
    assert budget.can_afford(cost(candidate.unsqueeze(0), mask).item())
    assert torch.isfinite(score)
    assert budget.remaining == 1.5
    assert acq.X_evaluation_mask is original

    with pytest.raises(ValueError, match="no affordable"):
        optimize_acqf_decoupled(
            acq,
            bounds,
            masks[1:],
            cost_function=cost,
            budget=budget,
            batch_initial_conditions=bounds.new_tensor([[[0.0]], [[0.2]]]),
            num_restarts=2,
        )
    assert acq.X_evaluation_mask is original


def test_joint_mask_assignments_and_batch_cost(monkeypatch):
    acq = MaskedQuadratic()
    original = acq.X_evaluation_mask
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    masks = torch.tensor([[True, False], [False, True]])
    seen = []

    def optimize(**kwargs):
        mask = kwargs["acq_function"].X_evaluation_mask
        seen.append(mask.clone())
        gain = 9.0 if torch.equal(mask, masks) else 1.0
        return bounds.new_tensor([[0.2], [0.8]]), bounds.new_tensor(gain)

    monkeypatch.setattr("robotorchan.optim.decoupled.optimize_acqf", optimize)
    candidate, mask, score = optimize_acqf_decoupled(
        acq, bounds, masks, q=2, output_costs=bounds.new_tensor([1.0, 2.0])
    )
    assert len(seen) == 4
    assert candidate.shape == (2, 1)
    assert torch.equal(mask, masks)
    torch.testing.assert_close(score, bounds.new_tensor(3.0))
    assert acq.X_evaluation_mask is original
    with pytest.raises(ValueError, match="max_mask_assignments"):
        optimize_acqf_decoupled(acq, bounds, masks, q=2, max_mask_assignments=3)


def test_mixed_decoupled_candidate():
    acq = MaskedQuadratic()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    masks = torch.tensor([[True, False], [False, True]])
    space = MixedVariableSpace(bounds, categorical_values={0: (0.0, 1.0)})
    candidate, mask, score = optimize_acqf_decoupled(
        acq, bounds, masks, variable_space=space, num_restarts=2, raw_samples=8
    )
    torch.testing.assert_close(candidate, bounds.new_tensor([[1.0]]))
    assert torch.equal(mask, masks[1:])
    torch.testing.assert_close(score, bounds.new_tensor(0.9375))


class BatchMaskedQuadratic(MaskedQuadratic):
    def forward(self, X):
        second = self.X_evaluation_mask[:, 1].to(X)
        center = 0.25 + 0.5 * second
        return (2 + second - (X[..., :, 0] - center).square()).sum(dim=-1)


@pytest.mark.parametrize("strategy", ["joint", "sequential", "heuristic"])
@pytest.mark.parametrize("optimizer", ["random", "sobol", "botorch"])
def test_batch_strategies_with_backend_and_budget(strategy, optimizer):
    acq = BatchMaskedQuadratic()
    original = acq.X_evaluation_mask
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    masks = torch.tensor([[True, False], [False, True]])
    budget = CostBudget(total=3.0)
    candidate, mask, score = optimize_acqf_decoupled(
        acq,
        bounds,
        masks,
        q=3,
        strategy=strategy,
        optimizer=optimizer,
        output_costs=bounds.new_tensor([1.0, 2.0]),
        budget=budget,
        max_mask_assignments=8 if strategy == "joint" else 2,
        num_restarts=2,
        raw_samples=16,
        seed=42,
    )
    assert candidate.shape == (3, 1)
    assert torch.equal(mask, masks[:1].expand(3, -1))
    assert torch.isfinite(score)
    assert budget.remaining == 3.0
    assert acq.X_evaluation_mask is original


def test_heuristic_assignment_limit_and_seed():
    from robotorchan.optim.decoupled import _heuristic_assignments

    first = list(_heuristic_assignments(3, 5, 7, 42, torch.tensor([2.0, 1.0, 3.0])))
    second = list(_heuristic_assignments(3, 5, 7, 42, torch.tensor([2.0, 1.0, 3.0])))
    assert first == second
    assert len(set(first)) == 7
    assert first[0] == (1,) * 5


@pytest.mark.parametrize("strategy", ["joint", "heuristic"])
def test_joint_mask_constraint(strategy):
    acq = BatchMaskedQuadratic()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    masks = torch.tensor([[True, False], [False, True]])
    _, selected, _ = optimize_acqf_decoupled(
        acq,
        bounds,
        masks,
        q=2,
        strategy=strategy,
        optimizer="sobol",
        mask_constraint=lambda mask: bool(mask.any(dim=0).all()),
        raw_samples=16,
    )
    assert selected.any(dim=0).all()


def test_joint_input_cost_budget():
    acq = BatchMaskedQuadratic()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    masks = torch.tensor([[False, True]])
    budget = CostBudget(total=2.5)

    def cost(X, mask):
        return 2 + X[..., :, 0].sum(dim=-1)

    initial = bounds.new_tensor([[[0.1], [0.1]], [[0.2], [0.2]]])
    candidate, selected, score = optimize_acqf_decoupled(
        acq,
        bounds,
        masks,
        q=2,
        cost_function=cost,
        budget=budget,
        batch_initial_conditions=initial,
        num_restarts=2,
    )
    assert candidate.shape == (2, 1)
    assert budget.can_afford(cost(candidate, selected).item())
    assert torch.isfinite(score)
    assert budget.remaining == 2.5


@pytest.mark.parametrize("optimizer", ["torch_adam", "torch_adamw", "torch_sgd"])
def test_named_gradient_backend(optimizer):
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidate, mask, score = optimize_acqf_decoupled(
        BatchMaskedQuadratic(),
        bounds,
        torch.tensor([[True, False], [False, True]]),
        q=2,
        optimizer=optimizer,
        num_restarts=2,
        raw_samples=8,
    )
    assert candidate.shape == (2, 1)
    assert mask.shape == (2, 2)
    assert torch.isfinite(score)
