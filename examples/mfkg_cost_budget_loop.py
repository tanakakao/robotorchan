"""Sequential cost-aware multi-fidelity BO with native BoTorch MF-KG.

Run: python examples/mfkg_cost_budget_loop.py
The evaluator is synthetic; replace it with a measured objective and cost.
"""

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.models.cost import AffineFidelityCostModel
from botorch.optim import optimize_acqf

from robotorchan.models import SingleTaskMultiFidelityGP
from robotorchan.optim.backends import optimize_acqf_botorch
from robotorchan.workflow import CostBudget


def evaluate(X: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Return a measured response and cost at the requested fidelity."""
    response = torch.sin(5 * X[:, :1]) + 0.2 * (1 - X[:, 1:])
    cost = (0.1 + X[:, 1]).sum().item()
    return response, cost


def project_to_target(X: torch.Tensor) -> torch.Tensor:
    projected = X.clone()
    projected[..., 1] = 1.0
    return projected


def run(*, iterations: int = 2, budget_total: float = 10.0) -> CostBudget:
    torch.manual_seed(7)
    dtype = torch.double
    x = torch.linspace(0.0, 1.0, 5, dtype=dtype)
    low = torch.stack((x, torch.full_like(x, 0.5)), dim=-1)
    high = torch.stack((x, torch.ones_like(x)), dim=-1)
    train_X = torch.cat((low, high))
    train_Y, _ = evaluate(train_X)
    # Initial observations are supplied before the tracked optimization budget.
    budget = CostBudget(total=budget_total)
    bounds = torch.tensor([[0.0, 0.5], [1.0, 1.0]], dtype=dtype)
    target_bounds = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=dtype)
    cost_model = AffineFidelityCostModel(fidelity_weights={1: 1.0}, fixed_cost=0.1)
    utility = InverseCostWeightedUtility(cost_model=cost_model)

    for step in range(iterations):
        model = SingleTaskMultiFidelityGP(train_X, train_Y, data_fidelities=[1])
        model.fit()
        model.eval()
        _, current_value = optimize_acqf(
            acq_function=PosteriorMean(model),
            bounds=target_bounds,
            q=1,
            num_restarts=2,
            raw_samples=8,
            fixed_features={1: 1.0},
        )
        acquisition = qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=4,
            current_value=current_value,
            cost_aware_utility=utility,
            project=project_to_target,
        )
        candidate, _ = optimize_acqf_botorch(
            acquisition,
            bounds,
            q=1,
            num_restarts=2,
            raw_samples=16,
        )
        predicted_cost = cost_model(candidate).sum().item()
        if not budget.can_afford(predicted_cost):
            break
        evaluation_id = f"evaluation-{step}"
        budget.reserve(evaluation_id, predicted_cost)
        try:
            response, actual_cost = evaluate(candidate)
        except Exception:
            budget.cancel(evaluation_id)
            raise
        budget.settle(evaluation_id, actual_cost)
        train_X = torch.cat((train_X, candidate), dim=0)
        train_Y = torch.cat((train_Y, response), dim=0)
        print(
            f"step={step + 1} fidelity={candidate[0, 1].item():.3f} "
            f"predicted_cost={predicted_cost:.3f} actual_cost={actual_cost:.3f} "
            f"remaining={budget.remaining:.3f}"
        )
    return budget


if __name__ == "__main__":
    run()
