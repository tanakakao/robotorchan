"""Equal-budget coupled and decoupled information acquisition benchmarks."""

from time import perf_counter
from typing import Literal

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import SumMarginalLogLikelihood

from robotorchan.acquisition.factory import create_acquisition
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.optim.decoupled import optimize_acqf_decoupled
from robotorchan.workflow.cost_budget import CostBudget


def _finite_pair_hypervolume(mean: torch.Tensor, hv: Hypervolume) -> float:
    """Best hypervolume of at most two points on a finite decision grid."""
    front = mean[is_non_dominated(mean)]
    best = max(hv.compute(point.unsqueeze(0)) for point in front)
    for i, point in enumerate(front):
        for other in front[i + 1 :]:
            best = max(best, hv.compute(torch.stack((point, other))))
    return best


def run_decoupled_benchmark(
    problem: BenchmarkProblem,
    *,
    seed: int = 0,
    budget: int = 30,
    output_costs: tuple[int, ...] = (1, 3),
    n_initial: int = 4,
    num_samples: int = 16,
    decoupled: bool = True,
    policy: Literal["epig", "kg", "hvkg", "random", "sobol"] = "epig",
    fit_model: bool = True,
) -> list[dict]:
    """Compare masked acquisition and baseline policies using a common finite recommendation set.

    Oracle hypervolume evaluates posterior-mean Pareto recommendations, not
    partially observed rows. Oracle values never enter training or selection.
    Initial design costs count against the same total budget.
    """
    if policy not in ("epig", "kg", "hvkg", "random", "sobol"):
        raise ValueError("Unknown benchmark policy.")
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
    generator = torch.Generator(device=bounds.device).manual_seed(seed)
    baseline_engine = torch.quasirandom.SobolEngine(d, scramble=True, seed=seed + 1)
    records = []
    elapsed = 0.0
    while True:
        start = perf_counter()
        model = ModelListGP(*(SingleTaskGP(x, y) for x, y in zip(train_X, train_Y, strict=True)))
        if fit_model:
            fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
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
                "method": f"{'decoupled' if decoupled else 'coupled'}_{policy}",
                "cost": spent,
                "hypervolume": score,
                "runtime_seconds": elapsed,
                "observations": [len(x) for x in train_X],
            }
        )
        if not any(ledger.can_afford(float(c)) for c in masks.to(bounds) @ costs):
            break
        start = perf_counter()
        if policy == "epig":
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
        elif policy == "kg":
            weights = costs.new_full((m,), 1.0 / m)
            sampler = ListSampler(
                *(
                    SobolQMCNormalSampler(torch.Size([4]), seed=seed + len(records) * m + j)
                    for j in range(m)
                )
            )
            acq = create_acquisition(
                "qDecoupledKnowledgeGradient",
                model=model,
                decision_X=target,
                weights=weights,
                sampler=sampler,
                X_evaluation_mask=masks[:1],
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
        elif policy == "hvkg":
            acq = create_acquisition(
                "qHypervolumeKnowledgeGradient",
                model=model,
                ref_point=reference,
                num_fantasies=2,
                num_pareto=2,
                current_value=mean.new_tensor(_finite_pair_hypervolume(mean, hv)),
                X_evaluation_mask=masks[:1],
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + len(records))
                candidate, mask, _ = optimize_acqf_decoupled(
                    acq,
                    bounds,
                    masks,
                    output_costs=costs,
                    budget=ledger,
                    num_restarts=2,
                    raw_samples=max(8, num_samples),
                    options={"maxiter": 20},
                )
        else:
            affordable = masks[(masks.to(bounds) @ costs) <= ledger.remaining]
            index = torch.randint(len(affordable), (1,), generator=generator, device=bounds.device)
            mask = affordable[index]
            if policy == "sobol":
                unit = baseline_engine.draw(1).to(bounds)
            else:
                unit = torch.rand(
                    (1, d), generator=generator, dtype=bounds.dtype, device=bounds.device
                )
            candidate = bounds[0] + unit * (bounds[1] - bounds[0])
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
