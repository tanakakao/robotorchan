"""Constrained decoupled BO with independently observed outcome constraints."""

from time import perf_counter

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, ModelListGP, SingleTaskGP
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import SumMarginalLogLikelihood

from robotorchan.acquisition.factory import create_acquisition
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.optim.decoupled import optimize_acqf_decoupled
from robotorchan.optim.domains.variable_space import MixedVariableSpace
from robotorchan.workflow.cost_budget import CostBudget


def run_constrained_decoupled_benchmark(
    problem: BenchmarkProblem,
    *,
    output_costs: tuple[int, ...],
    infeasible_utility: float,
    budget: int = 30,
    seed: int = 0,
    n_initial: int = 4,
    num_samples: int = 16,
    decoupled: bool = True,
    fit_model: bool = True,
) -> list[dict]:
    """Compare constrained KG under identical budgets and initial observations.

    Constraints are converted from g >= 0 to the KG convention g <= 0.
    Truth at the recommended point is used only for scoring, never training.
    """
    m = 1 + problem.n_constraints
    if (
        problem.n_objectives != 1
        or problem.n_constraints < 1
        or problem.optimal_value is None
        or len(output_costs) != m
        or any(c <= 0 for c in output_costs)
        or n_initial < 2
        or num_samples < 2
        or budget < n_initial * sum(output_costs)
    ):
        raise ValueError("Invalid constrained benchmark configuration.")
    bounds = problem.bounds
    engine = torch.quasirandom.SobolEngine(problem.dimension, scramble=True, seed=seed)
    integer_dims = tuple(i for i, kind in enumerate(problem.variable_types) if kind == "integer")
    cat_dims = [i for i, kind in enumerate(problem.variable_types) if kind == "categorical"]
    categories = {
        i: tuple(float(v) for v in range(int(bounds[0, i]), int(bounds[1, i]) + 1))
        for i in cat_dims
    }
    space = (
        MixedVariableSpace(bounds, integer_dims=integer_dims, categorical_values=categories)
        if integer_dims or cat_dims
        else None
    )

    def design(n):
        x = bounds[0] + engine.draw(n).to(bounds) * (bounds[1] - bounds[0])
        for i in (*integer_dims, *cat_dims):
            x[:, i] = x[:, i].round()
        return x

    X = design(n_initial)
    target = design(64)

    def evaluate(x):
        return torch.cat(
            (
                problem.to_maximization(problem.evaluate_observation(x)),
                -problem.evaluate_constraints(x),
            ),
            dim=-1,
        )

    Y = evaluate(X)
    train_X = [X.clone() for _ in range(m)]
    train_Y = [Y[:, i : i + 1].clone() for i in range(m)]
    costs = bounds.new_tensor(output_costs)
    weights = bounds.new_zeros(m)
    weights[0] = 1
    full = torch.ones((1, m), dtype=torch.bool, device=bounds.device)
    masks = torch.cat((torch.eye(m, dtype=torch.bool, device=bounds.device), full))
    if not decoupled:
        masks = full
    ledger = CostBudget(budget, spent=n_initial * sum(output_costs))
    records = []
    elapsed = 0.0
    while True:
        start = perf_counter()
        model = ModelListGP(
            *(
                MixedSingleTaskGP(x, y, cat_dims=cat_dims) if cat_dims else SingleTaskGP(x, y)
                for x, y in zip(train_X, train_Y, strict=True)
            )
        )
        if fit_model:
            fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(target)
            mean = posterior.mean
            std = posterior.variance[:, 1:].clamp_min(1e-12).sqrt()
            probability = torch.special.ndtr(-mean[:, 1:] / std).prod(dim=-1)
            utility = infeasible_utility + probability * (mean[:, 0] - infeasible_utility)
            index = int(utility.argmax())
            recommendation = target[index : index + 1]
            abstained = bool(utility[index] <= infeasible_utility)
            feasible = not abstained and bool(
                (problem.evaluate_constraints(recommendation) >= 0).all()
            )
            regret = float(problem.simple_regret(recommendation)) if feasible else None
        elapsed += perf_counter() - start
        records.append(
            {
                "seed": seed,
                "method": "decoupled_constrained_kg" if decoupled else "coupled_constrained_kg",
                "cost": ledger.spent,
                "feasible": feasible,
                "abstained": abstained,
                "simple_regret": regret,
                "runtime_seconds": elapsed,
                "observations": [len(x) for x in train_X],
                "recommendation": recommendation.squeeze(0).tolist(),
            }
        )
        if not any(ledger.can_afford(float(c)) for c in masks.to(bounds) @ costs):
            break
        start = perf_counter()
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
            constraint_indices=tuple(range(1, m)),
            constraint_thresholds=bounds.new_zeros(m - 1),
            infeasible_utility=infeasible_utility,
        )
        candidate, mask, _ = optimize_acqf_decoupled(
            acq,
            bounds,
            masks,
            output_costs=costs,
            budget=ledger,
            variable_space=space,
            optimizer="sobol",
            optimizer_options={"num_samples": num_samples},
            seed=seed + len(records),
        )
        ledger.record(float((mask.to(costs) @ costs).sum()))
        values = evaluate(candidate)
        for i in range(m):
            if mask[0, i]:
                train_X[i] = torch.cat((train_X[i], candidate))
                train_Y[i] = torch.cat((train_Y[i], values[:, i : i + 1]))
        elapsed += perf_counter() - start
    return records
