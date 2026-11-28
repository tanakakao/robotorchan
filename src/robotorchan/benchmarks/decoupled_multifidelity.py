"""Equal-budget decoupled multi-fidelity optimization on a finite fidelity grid."""

from time import perf_counter

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskMultiFidelityGP
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import SumMarginalLogLikelihood

from robotorchan.acquisition.factory import create_acquisition
from robotorchan.benchmarks.multifidelity_problems import (
    multifidelity_oscillatory,
    multifidelity_quadratic,
)
from robotorchan.optim.decoupled import optimize_acqf_decoupled
from robotorchan.workflow.cost_budget import CostBudget


def run_multifidelity_decoupled_benchmark(
    *,
    seed: int = 0,
    budget: int = 60,
    n_initial: int = 4,
    num_samples: int = 16,
    decoupled: bool = True,
    multifidelity: bool = True,
    fit_model: bool = True,
) -> list[dict]:
    """Compare two scalarized responses with independent masked fidelity observations.

    Fidelity levels 0.25 and 1 have integer cost multipliers 1 and 4.
    Initial observations are shared across all policies at target fidelity.
    """
    if n_initial < 2 or num_samples < 2 or budget < 12 * n_initial:
        raise ValueError("Invalid multi-fidelity benchmark configuration.")
    first = multifidelity_quadratic()
    second = multifidelity_oscillatory()
    bounds = first.bounds
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    X = engine.draw(n_initial).to(bounds)
    X[:, 1] = 1
    target = engine.draw(64).to(bounds)
    target[:, 1] = 1

    def evaluate(x):
        return -torch.cat((first.evaluate_truth(x), second.evaluate_truth(x)), dim=-1)

    Y = evaluate(X)
    train_X = [X.clone(), X.clone()]
    train_Y = [Y[:, :1].clone(), Y[:, 1:].clone()]
    costs = bounds.new_tensor([1, 2])
    full = torch.ones((1, 2), dtype=torch.bool, device=bounds.device)
    masks = torch.cat((torch.eye(2, dtype=torch.bool, device=bounds.device), full))
    if not decoupled:
        masks = full
    levels = ((0.25, 1), (1.0, 4)) if multifidelity else ((1.0, 4),)
    ledger = CostBudget(budget, spent=12 * n_initial)
    weights = bounds.new_tensor([0.5, 0.5])
    oracle_grid = torch.linspace(0, 1, 4097, dtype=bounds.dtype, device=bounds.device)
    oracle_X = torch.stack((oracle_grid, torch.ones_like(oracle_grid)), dim=-1)
    optimum = (evaluate(oracle_X) @ weights).max()
    records = []
    elapsed = 0.0
    fidelity_counts = [0, 2 * n_initial]
    while True:
        start = perf_counter()
        model = ModelListGP(
            *(
                SingleTaskMultiFidelityGP(x, y, data_fidelities=[1])
                for x, y in zip(train_X, train_Y, strict=True)
            )
        )
        if fit_model:
            fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
        model.eval()
        with torch.no_grad():
            index = (model.posterior(target).mean @ weights).argmax()
            recommendation = target[index].unsqueeze(0)
            regret = float((optimum - evaluate(recommendation) @ weights).clamp_min(0).item())
        elapsed += perf_counter() - start
        records.append(
            {
                "seed": seed,
                "method": (
                    f"{'decoupled' if decoupled else 'coupled'}_"
                    f"{'mf' if multifidelity else 'hf'}_kg"
                ),
                "cost": ledger.spent,
                "simple_regret": regret,
                "observations": [len(x) for x in train_X],
                "fidelity_counts": fidelity_counts.copy(),
                "runtime_seconds": elapsed,
            }
        )
        start = perf_counter()
        best = None
        for fidelity, multiplier in levels:
            level_costs = costs * multiplier
            if not any(ledger.can_afford(float(c)) for c in masks.to(bounds) @ level_costs):
                continue
            sampler = ListSampler(
                *(
                    SobolQMCNormalSampler(torch.Size([4]), seed=seed + len(records) * 2 + j)
                    for j in range(2)
                )
            )
            acq = create_acquisition(
                "qDecoupledKnowledgeGradient",
                model=model,
                decision_X=target,
                weights=weights,
                sampler=sampler,
                X_evaluation_mask=masks[:1],
                target_fidelities={1: 1.0},
            )
            candidate, mask, value = optimize_acqf_decoupled(
                acq,
                bounds,
                masks,
                output_costs=level_costs,
                budget=ledger,
                fixed_features={1: fidelity},
                optimizer="sobol",
                optimizer_options={"num_samples": num_samples},
                seed=seed + len(records),
            )
            if best is None or float(value) > best[0]:
                best = (float(value), candidate, mask, level_costs, fidelity)
        if best is None:
            break
        _, candidate, mask, level_costs, fidelity = best
        ledger.record(float((mask.to(costs) @ level_costs).sum()))
        fidelity_counts[int(fidelity == 1.0)] += int(mask.sum())
        values = evaluate(candidate)
        for i in range(2):
            if mask[0, i]:
                train_X[i] = torch.cat((train_X[i], candidate))
                train_Y[i] = torch.cat((train_Y[i], values[:, i : i + 1]))
        elapsed += perf_counter() - start
    return records
