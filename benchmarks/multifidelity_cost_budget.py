"""Reproducible equal-cost-budget comparison of MF-KG and random search.

Run: python benchmarks/multifidelity_cost_budget.py
Results are illustrative, not a claim of algorithmic superiority.
"""

import argparse
import csv
from pathlib import Path

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.models.cost import AffineFidelityCostModel
from botorch.optim import optimize_acqf

from robotorchan.models import SingleTaskMultiFidelityGP
from robotorchan.workflow import CostBudget


def objective(X: torch.Tensor) -> torch.Tensor:
    return torch.sin(5.0 * X[..., :1]) + 0.2 * (1.0 - X[..., 1:])


def project(X: torch.Tensor) -> torch.Tensor:
    result = X.clone()
    result[..., 1] = 1.0
    return result


def experiment(policy: str, seed: int, total_budget: float) -> list[dict[str, float | int | str]]:
    torch.manual_seed(seed)
    dtype = torch.double
    design = torch.linspace(0.0, 1.0, 5, dtype=dtype)
    train_X = torch.cat(
        (
            torch.stack((design, torch.full_like(design, 0.5)), dim=-1),
            torch.stack((design, torch.ones_like(design)), dim=-1),
        )
    )
    train_Y = objective(train_X)
    budget = CostBudget(total=total_budget)
    cost_model = AffineFidelityCostModel(fidelity_weights={1: 1.0}, fixed_cost=0.1)
    bounds = torch.tensor([[0.0, 0.5], [1.0, 1.0]], dtype=dtype)
    target_bounds = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=dtype)
    rows: list[dict[str, float | int | str]] = []
    best_target = float(train_Y[5:].max().item())

    for step in range(10):
        if policy == "random":
            candidate = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(1, 2, dtype=dtype)
        else:
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
            candidate, _ = optimize_acqf(
                acquisition,
                bounds=bounds,
                q=1,
                num_restarts=2,
                raw_samples=16,
            )
        predicted_cost = float(cost_model(candidate).sum().item())
        if not budget.can_afford(predicted_cost):
            break
        budget.reserve(f"evaluation-{step}", predicted_cost)
        measured_cost = float((0.1 + candidate[:, 1]).sum().item())
        new_Y = objective(candidate)
        budget.settle(f"evaluation-{step}", measured_cost)
        train_X = torch.cat((train_X, candidate))
        train_Y = torch.cat((train_Y, new_Y))
        # Only target-fidelity observations are directly comparable as observed incumbents.
        if bool(torch.isclose(candidate[0, 1], torch.tensor(1.0, dtype=dtype))):
            best_target = max(best_target, float(new_Y.max().item()))
        rows.append(
            {
                "policy": policy,
                "seed": seed,
                "step": step + 1,
                "spent": budget.spent,
                "remaining": budget.remaining,
                "fidelity": float(candidate[0, 1].item()),
                "best_observed_target": best_target,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=Path("multifidelity_cost_results.csv"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    rows = [
        row
        for seed in range(args.seeds)
        for policy in ("random", "mfkg")
        for row in experiment(policy, seed, args.budget)
    ]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
