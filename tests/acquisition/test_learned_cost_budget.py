"""Learned positive-cost surrogate and BoTorch utility integration."""

import torch
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.objective import GenericMCObjective

from robotorchan.models import SingleTaskGP
from robotorchan.workflow import CostBudget


def test_log_cost_gp_utility_and_budget_settlement() -> None:
    train_X = torch.tensor(
        [[0.0, 0.5], [0.5, 0.5], [1.0, 0.5], [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    observed_costs = 0.2 + 0.5 * train_X[:, :1] + train_X[:, 1:]
    cost_model = SingleTaskGP(train_X, observed_costs.log())
    cost_model.fit()
    cost_model.eval()
    objective = GenericMCObjective(lambda samples, X=None: samples.squeeze(-1).exp())
    utility = InverseCostWeightedUtility(
        cost_model=cost_model, cost_objective=objective, use_mean=True
    )
    candidate = torch.tensor([[[0.4, 0.5]]], dtype=torch.double)
    gain = torch.ones(1, 1, dtype=torch.double)
    weighted_gain = utility(X=candidate, deltas=gain)
    estimate = cost_model.posterior(candidate).mean.exp().sum().item()

    assert torch.isfinite(weighted_gain).all()
    assert (weighted_gain > 0).all()
    assert estimate > 0

    budget = CostBudget(total=3.0)
    budget.reserve("first", estimate)
    budget.settle("first", actual_cost=1.2)
    assert budget.spent == 1.2
    assert budget.reserved == 0
    assert budget.remaining == 1.8
