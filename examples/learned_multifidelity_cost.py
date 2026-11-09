"""Learn positive evaluation costs and reuse BoTorch cost-aware utility.

Run with: python examples/learned_multifidelity_cost.py
"""

import torch
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.objective import GenericMCObjective

from robotorchan.models import SingleTaskGP
from robotorchan.workflow import CostBudget


def main() -> None:
    dtype = torch.double
    design = torch.linspace(0.0, 1.0, 6, dtype=dtype)
    low = torch.stack((design, torch.full_like(design, 0.5)), dim=-1)
    high = torch.stack((design, torch.ones_like(design)), dim=-1)
    train_X = torch.cat((low, high))
    actual_costs = 0.2 + 0.3 * train_X[:, :1] + 1.5 * train_X[:, 1:]
    # Fit a GP on log(cost) so exponentiating predictions ensures positivity.
    cost_model = SingleTaskGP(train_X, actual_costs.log())
    cost_model.fit()
    cost_model.eval()

    # This uses exp(E[log(cost)]), a positive plug-in estimate, not E[cost].
    cost_objective = GenericMCObjective(lambda samples, X=None: samples.squeeze(-1).exp())
    cost_utility = InverseCostWeightedUtility(
        cost_model=cost_model,
        cost_objective=cost_objective,
        use_mean=True,
    )
    candidates = torch.tensor([[[0.4, 0.5]]], dtype=dtype)
    value_gain = torch.ones(1, 1, dtype=dtype)
    weighted_gain = cost_utility(X=candidates, deltas=value_gain)

    with torch.no_grad():
        predicted_log_cost = cost_model.posterior(candidates).mean.squeeze(-1)
        predicted_cost = predicted_log_cost.exp().sum().item()

    budget = CostBudget(total=10.0)
    if budget.can_afford(predicted_cost):
        budget.reserve("evaluation-1", predicted_cost)
        # Replace with the measured cost returned by the real evaluator.
        measured_cost = 0.2 + 0.3 * 0.4 + 1.5 * 0.5
        budget.settle("evaluation-1", measured_cost)
    print(f"Cost-weighted gain: {weighted_gain.item():.4f}")
    print(f"Predicted cost: {predicted_cost:.4f}")
    print(f"Actual spent: {budget.spent:.4f}")
    print(f"Remaining budget: {budget.remaining:.4f}")


if __name__ == "__main__":
    main()
