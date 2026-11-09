"""Cost-aware multi-fidelity TuRBO using existing public APIs.

Run: python examples/turbo_multifidelity_cost_budget.py
"""

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.models.cost import AffineFidelityCostModel
from botorch.optim import optimize_acqf

from robotorchan.models import SingleTaskMultiFidelityGP
from robotorchan.optim import TuRBOState, TuRBOStrategy
from robotorchan.workflow import CostBudget


def evaluate(X: torch.Tensor) -> tuple[torch.Tensor, float]:
    values = torch.sin(5.0 * X[:, :1]) + 0.2 * (1.0 - X[:, 1:])
    return values, (0.1 + X[:, 1]).sum().item()


def project(X: torch.Tensor) -> torch.Tensor:
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
    bounds = torch.tensor([[0.0, 0.5], [1.0, 1.0]], dtype=dtype)
    target_bounds = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=dtype)
    cost_model = AffineFidelityCostModel(fidelity_weights={1: 1.0}, fixed_cost=0.1)
    budget = CostBudget(total=budget_total)
    initial_best = int(train_Y[5:].argmax().item()) + 5
    strategy = TuRBOStrategy(
        bounds,
        center=train_X[initial_best],
        state=TuRBOState(dim=2, batch_size=1),
        num_restarts=2,
        raw_samples=16,
        seed=7,
    )

    for step in range(iterations):
        model = SingleTaskMultiFidelityGP(train_X, train_Y, data_fidelities=[1])
        model.fit()
        model.eval()
        _, current_value = optimize_acqf(
            PosteriorMean(model),
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
            cost_aware_utility=InverseCostWeightedUtility(cost_model=cost_model),
            project=project,
        )
        result = strategy.optimize_multifidelity(acquisition, fidelity_dims=[1])
        candidate = result.candidates.detach()
        estimated_cost = cost_model(candidate).sum().item()
        if not budget.can_afford(estimated_cost):
            break
        evaluation_id = f"turbo-evaluation-{step}"
        budget.reserve(evaluation_id, estimated_cost)
        try:
            new_Y, actual_cost = evaluate(candidate)
        except Exception:
            budget.cancel(evaluation_id)
            raise
        budget.settle(evaluation_id, actual_cost)
        train_X = torch.cat((train_X, candidate))
        train_Y = torch.cat((train_Y, new_Y))
        # Compare candidates at target fidelity, not by their raw low-fidelity response.
        with torch.no_grad():
            state_values = model.posterior(project(candidate)).mean.detach()
        strategy.update_state(new_Y, candidates=candidate, state_values=state_values)
        if strategy.state.restart_triggered:
            strategy.restart()
        print(
            f"step={step + 1} fidelity={candidate[0, 1].item():.3f} "
            f"spent={budget.spent:.3f} remaining={budget.remaining:.3f}"
        )
    return budget


if __name__ == "__main__":
    run()
