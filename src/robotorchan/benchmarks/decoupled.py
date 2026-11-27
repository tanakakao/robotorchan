"""Equal-budget coupled and decoupled information acquisition benchmarks."""

from time import perf_counter

import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from robotorchan.acquisition.factory import create_acquisition
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.optim.decoupled import optimize_acqf_decoupled
from robotorchan.workflow.cost_budget import CostBudget


def run_decoupled_benchmark(
    problem: BenchmarkProblem,
    *,
    seed: int = 0,
    budget: int = 30,
    output_costs: tuple[int, ...] = (1, 3),
    n_initial: int = 4,
    num_samples: int = 16,
    decoupled: bool = True,
) -> list[dict]:
    """Compare masked EPIG policies using a common finite recommendation set.

    Oracle hypervolume evaluates posterior-mean Pareto recommendations, not
    partially observed rows. Oracle values never enter training or selection.
    Initial design costs count against the same total budget.
    """
    m = problem.n_objectives
    if (
        m < 2
        or problem.reference_point is None
        or problem.n_constraints
        or len(output_costs) != m
        or any(c <= 0 for c in output_costs)
        or n_initial < 2
        or num_samples < 2
        or budget < n_initial * sum(output_costs)
    ):
        raise ValueError("Invalid unconstrained multiobjective benchmark configuration.")
    bounds = problem.bounds
    d = bounds.shape[-1]
    engine = torch.quasirandom.SobolEngine(d, scramble=True, seed=seed)
    X = bounds[0] + engine.draw(n_initial).to(bounds) * (bounds[1] - bounds[0])
    target = bounds[0] + engine.draw(64).to(bounds) * (bounds[1] - bounds[0])
    Y = problem.to_maximization(problem.objective(X))
    train_X = [X.clone() for _ in range(m)]
    train_Y = [Y[:, i : i + 1].clone() for i in range(m)]
    costs = bounds.new_tensor(output_costs)
    full = torch.ones((1, m), dtype=torch.bool, device=bounds.device)
    masks = torch.cat((torch.eye(m, dtype=torch.bool, device=bounds.device), full))
    if not decoupled:
        masks = full
    ledger = CostBudget(budget)
    ledger.record(n_initial * sum(output_costs))
    spent = n_initial * sum(output_costs)
    reference = problem.to_maximization(problem.reference_point.to(bounds))
    hv = Hypervolume(reference)
    records = []
    elapsed = 0.0
    while True:
        start = perf_counter()
        model = ModelListGP(*(SingleTaskGP(x, y) for x, y in zip(train_X, train_Y, strict=True)))
        model.eval()
        with torch.no_grad():
            mean = model.posterior(target).mean
            recommended = target[is_non_dominated(mean)]
            truth = problem.to_maximization(problem.objective(recommended))
            score = hv.compute(truth)
        elapsed += perf_counter() - start
        records.append(
            {
                "seed": seed,
                "method": "decoupled_epig" if decoupled else "coupled_epig",
                "cost": spent,
                "hypervolume": score,
                "runtime_seconds": elapsed,
                "observations": [len(x) for x in train_X],
            }
        )
        if not any(ledger.can_afford(float(c)) for c in masks.to(bounds) @ costs):
            break
        start = perf_counter()
        acq = create_acquisition(
            "qDecoupledExpectedPredictiveInformationGain",
            model=model,
            target_X=target,
            X_evaluation_mask=masks[:1],
            observation_noise=costs.new_full((m,), 0.1),
        )
        candidate, mask, _ = optimize_acqf_decoupled(
            acq,
            bounds,
            masks,
            output_costs=costs,
            budget=ledger,
            optimizer="sobol",
            optimizer_options={"num_samples": num_samples},
            seed=seed + len(records),
        )
        charge = int((mask.to(costs) @ costs).sum().item())
        ledger.record(charge)
        spent += charge
        values = problem.to_maximization(problem.objective(candidate))
        for i in range(m):
            if mask[0, i]:
                train_X[i] = torch.cat((train_X[i], candidate))
                train_Y[i] = torch.cat((train_Y[i], values[:, i : i + 1]))
        elapsed += perf_counter() - start
    return records
