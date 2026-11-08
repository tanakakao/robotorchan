"""NSGA-II backend for genuinely vector-valued optimization targets."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from robotorchan.optim.backend_support.runtime import make_generator, validate_bounds
from robotorchan.optim.constraints.contracts import (
    CandidateConstraints,
    LinearConstraint,
    resolve_candidate_constraints,
)
from robotorchan.optim.constraints.evaluation import candidate_constraint_violation
from robotorchan.optim.constraints.validation import validate_candidate_constraints
from robotorchan.optim.domains.variable_space import MixedVariableSpace

VectorObjective = Callable[[Tensor], Tensor]


def optimize_vector_nsga2(
    objective: VectorObjective,
    bounds: Tensor,
    *,
    population_size: int = 128,
    generations: int = 100,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    seed: int | None = None,
    constraints: CandidateConstraints | None = None,
    inequality_constraints: tuple[LinearConstraint, ...] | None = None,
    equality_constraints: tuple[LinearConstraint, ...] | None = None,
) -> tuple[Tensor, Tensor]:
    """Approximate a Pareto set with a native, batched NSGA-II implementation.

    The objective receives [population, d] and must return [population, m].
    All objectives are maximized. This API is intentionally separate from
    scalar BoTorch acquisition optimization: qEHVI/qNEHVI already produce
    scalar acquisition values and do not require NSGA-II.
    """
    validate_bounds(bounds)
    _validate(
        bounds,
        population_size,
        generations,
        crossover_rate,
        mutation_rate,
        mutation_scale,
    )
    candidate_constraints = _resolve_nsga2_constraints(
        constraints, inequality_constraints, equality_constraints, bounds=bounds
    )
    generator = make_generator(bounds, seed)
    lower, upper = bounds
    population = lower + (upper - lower) * torch.rand(
        population_size,
        bounds.shape[-1],
        dtype=bounds.dtype,
        device=bounds.device,
        generator=generator,
    )
    population = _project_nsga2_equalities(population, candidate_constraints, bounds)
    for generation in range(generations):
        values = _evaluate(objective, population)
        if generation == generations - 1:
            break
        violations = _nsga2_violations(population, candidate_constraints)
        ranks, crowding = _rank_and_crowding(values, violations)
        parents_a = _binary_tournament(population, ranks, crowding, generator)
        parents_b = _binary_tournament(population, ranks, crowding, generator)
        offspring = _crossover(parents_a, parents_b, crossover_rate, generator)
        offspring = _mutate(offspring, lower, upper, mutation_rate, mutation_scale, generator)
        offspring = _project_nsga2_equalities(offspring, candidate_constraints, bounds)
        combined = torch.cat([population, offspring], dim=0)
        combined_values = _evaluate(objective, combined)
        combined_violations = _nsga2_violations(combined, candidate_constraints)
        population = _environmental_select(
            combined, combined_values, population_size, combined_violations
        )
    values = _evaluate(objective, population)
    violations = _nsga2_violations(population, candidate_constraints)
    if not bool((violations <= 0).any()):
        raise ValueError("NSGA-II found no feasible candidate.")
    ranks, _ = _rank_and_crowding(values, violations)
    mask = (ranks == 0) & (violations <= 0)
    return population[mask], values[mask]


def _evaluate(objective: VectorObjective, population: Tensor) -> Tensor:
    with torch.no_grad():
        values = objective(population)
    if values.ndim != 2 or values.shape[0] != population.shape[0]:
        raise ValueError("NSGA-II objective must return shape [population, objectives].")
    if values.shape[1] < 2:
        raise ValueError("NSGA-II requires at least two objective values.")
    return values


def _rank_and_crowding(values: Tensor, violations: Tensor | None = None) -> tuple[Tensor, Tensor]:
    n = values.shape[0]
    dominates = torch.zeros((n, n), dtype=torch.bool, device=values.device)
    for i in range(n):
        no_worse = (values[i] >= values).all(dim=1)
        strictly_better = (values[i] > values).any(dim=1)
        dominates[i] = no_worse & strictly_better
    if violations is not None:
        feasible = violations <= 0
        dominates = torch.where(
            feasible[:, None] & ~feasible[None, :],
            True,
            dominates,
        )
        dominates = torch.where(
            ~feasible[:, None] & feasible[None, :],
            False,
            dominates,
        )
        both_infeasible = ~feasible[:, None] & ~feasible[None, :]
        dominates = torch.where(
            both_infeasible,
            violations[:, None] < violations[None, :],
            dominates,
        )
    domination_count = dominates.sum(dim=0)
    ranks = torch.full((n,), -1, dtype=torch.long, device=values.device)
    remaining = domination_count.clone()
    rank = 0
    front = torch.where(remaining == 0)[0]
    while front.numel():
        ranks[front] = rank
        remaining = remaining - dominates[front].sum(dim=0)
        front = torch.where((remaining == 0) & (ranks < 0))[0]
        rank += 1
    crowding = torch.zeros(n, dtype=values.dtype, device=values.device)
    for current_rank in range(rank):
        indices = torch.where(ranks == current_rank)[0]
        if indices.numel() <= 2:
            crowding[indices] = torch.inf
            continue
        front_values = values[indices]
        for objective_idx in range(values.shape[1]):
            order = torch.argsort(front_values[:, objective_idx])
            sorted_indices = indices[order]
            crowding[sorted_indices[[0, -1]]] = torch.inf
            span = front_values[order[-1], objective_idx] - front_values[order[0], objective_idx]
            if span > 0:
                crowding[sorted_indices[1:-1]] += (
                    front_values[order[2:], objective_idx] - front_values[order[:-2], objective_idx]
                ) / span
    return ranks, crowding


def _binary_tournament(
    population: Tensor,
    ranks: Tensor,
    crowding: Tensor,
    generator: torch.Generator,
) -> Tensor:
    pairs = torch.randint(
        population.shape[0],
        (population.shape[0], 2),
        device=population.device,
        generator=generator,
    )
    first, second = pairs[:, 0], pairs[:, 1]
    choose_first = (ranks[first] < ranks[second]) | (
        (ranks[first] == ranks[second]) & (crowding[first] >= crowding[second])
    )
    return population[torch.where(choose_first, first, second)]


def _environmental_select(
    population: Tensor, values: Tensor, size: int, violations: Tensor | None = None
) -> Tensor:
    ranks, crowding = _rank_and_crowding(values, violations)
    selected: list[Tensor] = []
    count = 0
    for rank in range(int(ranks.max().item()) + 1):
        front = torch.where(ranks == rank)[0]
        remaining = size - count
        if front.numel() <= remaining:
            selected.append(front)
            count += front.numel()
        else:
            order = torch.argsort(crowding[front], descending=True)
            selected.append(front[order[:remaining]])
            break
    return population[torch.cat(selected)]


def _crossover(
    parents_a: Tensor,
    parents_b: Tensor,
    rate: float,
    generator: torch.Generator,
) -> Tensor:
    mask = (
        torch.rand(
            parents_a.shape,
            dtype=parents_a.dtype,
            device=parents_a.device,
            generator=generator,
        )
        < rate
    )
    alpha = torch.rand(
        parents_a.shape,
        dtype=parents_a.dtype,
        device=parents_a.device,
        generator=generator,
    )
    return torch.where(mask, alpha * parents_a + (1 - alpha) * parents_b, parents_a)


def _mutate(
    offspring: Tensor,
    lower: Tensor,
    upper: Tensor,
    rate: float,
    scale: float,
    generator: torch.Generator,
) -> Tensor:
    mask = (
        torch.rand(
            offspring.shape,
            dtype=offspring.dtype,
            device=offspring.device,
            generator=generator,
        )
        < rate
    )
    noise = torch.randn(
        offspring.shape,
        dtype=offspring.dtype,
        device=offspring.device,
        generator=generator,
    )
    mutated = offspring + mask * noise * scale * (upper - lower)
    return torch.maximum(torch.minimum(mutated, upper), lower)


def _validate(
    bounds: Tensor,
    population_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    mutation_scale: float,
) -> None:
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError("bounds must have shape [2, d].")
    if torch.any(bounds[0] >= bounds[1]):
        raise ValueError("Every lower bound must be strictly smaller than its upper bound.")
    if population_size < 4:
        raise ValueError("population_size must be at least 4.")
    if generations < 1:
        raise ValueError("generations must be at least 1.")
    if not 0 <= crossover_rate <= 1:
        raise ValueError("crossover_rate must lie between 0 and 1.")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must lie between 0 and 1.")
    if mutation_scale < 0:
        raise ValueError("mutation_scale must be non-negative.")


def optimize_vector_nsga2_mixed(
    objective: VectorObjective,
    bounds: Tensor,
    *,
    variable_space: MixedVariableSpace,
    population_size: int = 128,
    generations: int = 100,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    seed: int | None = None,
    constraints: CandidateConstraints | None = None,
    inequality_constraints: tuple[LinearConstraint, ...] | None = None,
    equality_constraints: tuple[LinearConstraint, ...] | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize vector objectives with discrete-safe NSGA-II genetic operators.

    Nominal categories are sampled or inherited, never interpolated or ordered.
    The returned population contains only non-dominated legal candidates.
    """
    validate_bounds(bounds)
    if not torch.equal(variable_space.bounds, bounds):
        raise ValueError("variable_space bounds must match bounds.")
    _validate(bounds, population_size, generations, crossover_rate, mutation_rate, mutation_scale)
    candidate_constraints = _resolve_nsga2_constraints(
        constraints,
        inequality_constraints,
        equality_constraints,
        bounds=bounds,
        variable_space=variable_space,
    )
    generator = make_generator(bounds, seed)
    lower, upper = bounds
    population = lower + (upper - lower) * torch.rand(
        population_size,
        bounds.shape[-1],
        dtype=bounds.dtype,
        device=bounds.device,
        generator=generator,
    )
    population = _repair_mixed_nsga2(population, variable_space, generator, resample=True)
    population = _project_nsga2_equalities(population, candidate_constraints, bounds)
    for generation in range(generations):
        values = _evaluate(objective, population)
        if generation == generations - 1:
            break
        violations = _nsga2_violations(population, candidate_constraints)
        ranks, crowding = _rank_and_crowding(values, violations)
        parents_a = _binary_tournament(population, ranks, crowding, generator)
        parents_b = _binary_tournament(population, ranks, crowding, generator)
        offspring = _crossover(parents_a, parents_b, crossover_rate, generator)
        for dim in (*variable_space.integer_dims, *variable_space.categorical_dims):
            choose_b = (
                torch.rand(population_size, device=bounds.device, generator=generator)
                < crossover_rate * 0.5
            )
            offspring[:, dim] = torch.where(choose_b, parents_b[:, dim], parents_a[:, dim])
        discrete_dims = (*variable_space.integer_dims, *variable_space.categorical_dims)
        inherited_discrete = offspring[:, discrete_dims].clone() if discrete_dims else None
        offspring = _mutate(offspring, lower, upper, mutation_rate, mutation_scale, generator)
        if inherited_discrete is not None:
            offspring[:, discrete_dims] = inherited_discrete
        offspring = _repair_mixed_nsga2(
            offspring, variable_space, generator, resample=False, mutation_rate=mutation_rate
        )
        offspring = _project_nsga2_equalities(offspring, candidate_constraints, bounds)
        combined = torch.cat([population, offspring], dim=0)
        combined_values = _evaluate(objective, combined)
        combined_violations = _nsga2_violations(combined, candidate_constraints)
        population = _environmental_select(
            combined, combined_values, population_size, combined_violations
        )
    values = _evaluate(objective, population)
    violations = _nsga2_violations(population, candidate_constraints)
    if not bool((violations <= 0).any()):
        raise ValueError("Mixed NSGA-II found no feasible candidate.")
    ranks, _ = _rank_and_crowding(values, violations)
    mask = (ranks == 0) & (violations <= 0)
    return population[mask], values[mask]


def _repair_mixed_nsga2(
    population: Tensor,
    space: MixedVariableSpace,
    generator: torch.Generator,
    *,
    resample: bool,
    mutation_rate: float = 0.0,
) -> Tensor:
    """Map discrete coordinates to legal values without categorical distances."""
    population = population.clone()
    lower, upper = space.bounds
    for dim in space.integer_dims:
        if resample:
            population[:, dim] = torch.randint(
                int(torch.ceil(lower[dim]).item()),
                int(torch.floor(upper[dim]).item()) + 1,
                (population.shape[0],),
                device=population.device,
                generator=generator,
            ).to(population.dtype)
        else:
            mutate = (
                torch.rand(population.shape[0], device=population.device, generator=generator)
                < mutation_rate
            )
            sampled = torch.randint(
                int(torch.ceil(lower[dim]).item()),
                int(torch.floor(upper[dim]).item()) + 1,
                (population.shape[0],),
                device=population.device,
                generator=generator,
            ).to(population.dtype)
            inherited = (
                population[:, dim].round().clamp(torch.ceil(lower[dim]), torch.floor(upper[dim]))
            )
            population[:, dim] = torch.where(mutate, sampled, inherited)
    for dim, categories in (space.categorical_values or {}).items():
        allowed = torch.as_tensor(categories, dtype=population.dtype, device=population.device)
        if resample:
            indices = torch.randint(
                allowed.numel(),
                (population.shape[0],),
                device=population.device,
                generator=generator,
            )
            population[:, dim] = allowed[indices]
        else:
            mutate = (
                torch.rand(population.shape[0], device=population.device, generator=generator)
                < mutation_rate
            )
            indices = torch.randint(
                allowed.numel(),
                (population.shape[0],),
                device=population.device,
                generator=generator,
            )
            population[:, dim] = torch.where(mutate, allowed[indices], population[:, dim])
    return population


def _resolve_nsga2_constraints(
    constraints: CandidateConstraints | None,
    inequality_constraints: tuple[LinearConstraint, ...] | None,
    equality_constraints: tuple[LinearConstraint, ...] | None,
    *,
    bounds: Tensor,
    variable_space: MixedVariableSpace | None = None,
) -> CandidateConstraints:
    resolved = resolve_candidate_constraints(
        constraints,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
    )
    validate_candidate_constraints(resolved, bounds=bounds, q=1)
    if resolved.nonlinear_inequality_constraints:
        raise NotImplementedError("NSGA-II does not support nonlinear constraints yet.")
    if variable_space is not None:
        discrete = set(variable_space.integer_dims) | set(variable_space.categorical_dims)
        for indices, _, _ in (
            *resolved.inequality_constraints,
            *resolved.equality_constraints,
        ):
            dimensions = indices if indices.ndim == 1 else indices[:, 1]
            if any(int(dim) in discrete for dim in dimensions):
                raise ValueError(
                    "Mixed NSGA-II constraints may reference continuous dimensions only."
                )
    return resolved


def _project_nsga2_equalities(
    population: Tensor, constraints: CandidateConstraints, bounds: Tensor
) -> Tensor:
    """Project continuous candidates onto linear equalities when possible.

    Bounds are enforced after projection; infeasible intersections remain subject
    to the existing final feasibility check.
    """
    if not constraints.equality_constraints:
        return population
    rows = []
    targets = []
    for indices, coefficients, rhs in constraints.equality_constraints:
        if indices.ndim != 1:
            raise NotImplementedError("NSGA-II does not support inter-point equalities.")
        row = torch.zeros(bounds.shape[-1], dtype=bounds.dtype, device=bounds.device)
        row.scatter_add_(0, indices.to(bounds.device), coefficients.to(bounds))
        rows.append(row)
        targets.append(rhs)
    matrix = torch.stack(rows)
    rhs_vector = torch.as_tensor(targets, dtype=bounds.dtype, device=bounds.device)
    residual = rhs_vector - population @ matrix.T
    correction = residual @ torch.linalg.pinv(matrix).T
    projected = population + correction
    return projected.clamp(min=bounds[0], max=bounds[1])


def _nsga2_violations(population: Tensor, constraints: CandidateConstraints) -> Tensor:
    return candidate_constraint_violation(population.unsqueeze(-2), constraints)
