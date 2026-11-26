"""Candidate and mask enumeration contracts."""

from types import SimpleNamespace

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import create_optimizer, optimize_acqf_decoupled
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
    assert candidate.item() <= 0.1
    assert torch.equal(mask, masks)
    assert torch.isfinite(score)
    assert budget.spent == 0
