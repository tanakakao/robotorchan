"""Mixed-variable genetic-algorithm acquisition optimizer backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.constraints.evaluation import (
    candidate_constraint_violation,
    feasibility_first_ranks,
)
from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.runtime import make_generator, validate_bounds
from robotorchan.optim.variable_space import MixedVariableSpace


def optimize_acqf_mixed_ga(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    *,
    integer_dims: Sequence[int] = (),
    categorical_values: Mapping[int, Sequence[float]] | None = None,
    population_size: int = 128,
    generations: int = 100,
    elite_fraction: float = 0.1,
    tournament_size: int = 3,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    seed: int | None = None,
    constraints: CandidateConstraints | None = None,
    equality_tolerance: float = 1e-6,
    variable_space: MixedVariableSpace | None = None,
    fixed_features: Mapping[int, float | Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize an acquisition function over continuous, integer, and categorical inputs."""
    if variable_space is not None:
        if integer_dims or categorical_values:
            raise ValueError("Use variable_space or integer_dims/categorical_values, not both.")
        if not torch.equal(variable_space.bounds, bounds):
            raise ValueError("variable_space bounds must match bounds.")
        integer_dims = variable_space.integer_dims
        categorical_values = dict(variable_space.categorical_values)
        variable_space.validate_fixed_features(fixed_features)
    else:
        categorical_values = dict(categorical_values or {})
        integer_dims = tuple(integer_dims)
    validate_bounds(bounds)
    _validate_configuration(
        q=q,
        bounds=bounds,
        integer_dims=integer_dims,
        categorical_values=categorical_values,
        population_size=population_size,
        generations=generations,
        elite_fraction=elite_fraction,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_scale=mutation_scale,
    )
    candidate_constraints = constraints or CandidateConstraints()
    generator = make_generator(bounds, seed)
    d = bounds.shape[-1]
    lower = bounds[0].repeat(q)
    upper = bounds[1].repeat(q)
    expanded_integer_dims = _expand_dims(integer_dims, q, d)
    expanded_categories = _expand_categories(categorical_values, q, d, bounds)
    expanded_fixed = _expand_fixed_features(fixed_features, q, d, bounds)

    population = lower + (upper - lower) * torch.rand(
        population_size,
        q * d,
        dtype=bounds.dtype,
        device=bounds.device,
        generator=generator,
    )
    population = _repair_structured(
        population,
        lower,
        upper,
        expanded_integer_dims,
        expanded_categories,
        generator,
        resample_categorical=True,
    )
    population = _apply_fixed_features(population, expanded_fixed)
    elite_count = max(1, int(population_size * elite_fraction))

    best_candidate: Tensor | None = None
    best_value: Tensor | None = None
    best_violation: Tensor | None = None
    for generation in range(generations):
        scores, values, violation = _evaluate_population(
            acq_function,
            population,
            q,
            d,
            candidate_constraints,
            equality_tolerance,
        )
        generation_best = scores.argmax()
        candidate_value = values[generation_best]
        candidate_violation = violation[generation_best]
        if best_candidate is None or _is_better_candidate(
            candidate_value,
            candidate_violation,
            best_value,
            best_violation,
        ):
            best_value = candidate_value.detach().clone()
            best_violation = candidate_violation.detach().clone()
            best_candidate = population[generation_best].detach().clone()
        if generation == generations - 1:
            break

        elites = population[torch.topk(scores, k=elite_count).indices]
        offspring_count = population_size - elite_count
        parents_a = _tournament_select(
            population, scores, offspring_count, tournament_size, generator
        )
        parents_b = _tournament_select(
            population, scores, offspring_count, tournament_size, generator
        )
        offspring = _crossover(
            parents_a,
            parents_b,
            crossover_rate,
            generator,
            expanded_integer_dims,
            tuple(expanded_categories),
        )
        offspring = _mutate(
            offspring,
            lower,
            upper,
            mutation_rate,
            mutation_scale,
            generator,
            expanded_integer_dims,
            expanded_categories,
        )
        offspring = _repair_structured(
            offspring,
            lower,
            upper,
            expanded_integer_dims,
            expanded_categories,
            generator,
            resample_categorical=True,
        )
        offspring = _apply_fixed_features(offspring, expanded_fixed)
        population = torch.cat([elites, offspring], dim=0)

    if best_candidate is None or best_value is None:
        raise RuntimeError("Mixed Genetic Algorithm did not generate any candidate.")
    candidate = best_candidate.reshape(q, d)
    with torch.no_grad():
        value = acq_function(candidate.unsqueeze(0)).reshape(())
    return candidate, value


def _evaluate_population(
    acq_function: AcquisitionFunction,
    population: Tensor,
    q: int,
    d: int,
    constraints: CandidateConstraints,
    equality_tolerance: float,
) -> tuple[Tensor, Tensor, Tensor]:
    candidates = population.reshape(population.shape[0], q, d)
    with torch.no_grad():
        values = acq_function(candidates)
    if values.numel() != population.shape[0]:
        raise ValueError(
            "Mixed Genetic Algorithm requires one scalar acquisition value per population member."
        )
    scores = values.reshape(population.shape[0])
    violation = candidate_constraint_violation(
        candidates, constraints, equality_tolerance=equality_tolerance
    )
    return feasibility_first_ranks(scores, violation), scores, violation


def _is_better_candidate(
    value: Tensor,
    violation: Tensor,
    best_value: Tensor | None,
    best_violation: Tensor | None,
) -> bool:
    if best_value is None or best_violation is None:
        return True
    feasible = bool(violation <= 0)
    best_feasible = bool(best_violation <= 0)
    if feasible != best_feasible:
        return feasible
    if feasible:
        return bool(value > best_value)
    if violation != best_violation:
        return bool(violation < best_violation)
    return bool(value > best_value)


def _tournament_select(
    population: Tensor,
    scores: Tensor,
    count: int,
    tournament_size: int,
    generator: torch.Generator,
) -> Tensor:
    competitors = torch.randint(
        population.shape[0],
        (count, tournament_size),
        device=population.device,
        generator=generator,
    )
    winner_offsets = scores[competitors].argmax(dim=1)
    rows = torch.arange(count, device=population.device)
    return population[competitors[rows, winner_offsets]]


def _crossover(
    parents_a: Tensor,
    parents_b: Tensor,
    crossover_rate: float,
    generator: torch.Generator,
    integer_dims: tuple[int, ...],
    categorical_dims: tuple[int, ...],
) -> Tensor:
    """Use arithmetic crossover only for continuous coordinates."""
    mask = (
        torch.rand(
            parents_a.shape,
            dtype=parents_a.dtype,
            device=parents_a.device,
            generator=generator,
        )
        < crossover_rate
    )
    alpha = torch.rand(
        parents_a.shape,
        dtype=parents_a.dtype,
        device=parents_a.device,
        generator=generator,
    )
    offspring = torch.where(mask, alpha * parents_a + (1.0 - alpha) * parents_b, parents_a)
    discrete_dims = integer_dims + categorical_dims
    if discrete_dims:
        index = torch.tensor(discrete_dims, device=parents_a.device)
        inherit_b = (
            torch.rand(
                parents_a.shape[0],
                len(discrete_dims),
                device=parents_a.device,
                generator=generator,
            )
            < crossover_rate
        )
        offspring[:, index] = torch.where(
            inherit_b,
            parents_b[:, index],
            parents_a[:, index],
        )
    return offspring


def _mutate(
    offspring: Tensor,
    lower: Tensor,
    upper: Tensor,
    mutation_rate: float,
    mutation_scale: float,
    generator: torch.Generator,
    integer_dims: tuple[int, ...],
    categorical_values: dict[int, Tensor],
) -> Tensor:
    """Mutate each variable kind without imposing a false categorical metric."""
    result = offspring.clone()
    structured = set(integer_dims) | set(categorical_values)
    continuous_dims = tuple(dim for dim in range(offspring.shape[1]) if dim not in structured)
    if continuous_dims:
        index = torch.tensor(continuous_dims, device=offspring.device)
        mask = (
            torch.rand(
                offspring.shape[0],
                len(continuous_dims),
                dtype=offspring.dtype,
                device=offspring.device,
                generator=generator,
            )
            < mutation_rate
        )
        noise = torch.randn(
            offspring.shape[0],
            len(continuous_dims),
            dtype=offspring.dtype,
            device=offspring.device,
            generator=generator,
        )
        mutated = result[:, index] + mask * noise * mutation_scale * (upper[index] - lower[index])
        result[:, index] = torch.maximum(torch.minimum(mutated, upper[index]), lower[index])
    for dim in integer_dims:
        mutate = (
            torch.rand(offspring.shape[0], device=offspring.device, generator=generator)
            < mutation_rate
        )
        step = torch.where(
            torch.rand(offspring.shape[0], device=offspring.device, generator=generator) < 0.5,
            -torch.ones(offspring.shape[0], device=offspring.device, dtype=offspring.dtype),
            torch.ones(offspring.shape[0], device=offspring.device, dtype=offspring.dtype),
        )
        result[:, dim] = torch.where(mutate, result[:, dim] + step, result[:, dim])
        result[:, dim] = result[:, dim].clamp(lower[dim], upper[dim]).round()
    for dim, values in categorical_values.items():
        mutate = (
            torch.rand(offspring.shape[0], device=offspring.device, generator=generator)
            < mutation_rate
        )
        choices = torch.randint(
            len(values),
            (offspring.shape[0],),
            device=offspring.device,
            generator=generator,
        )
        result[:, dim] = torch.where(mutate, values[choices], result[:, dim])
    return result


def _repair_structured(
    population: Tensor,
    lower: Tensor,
    upper: Tensor,
    integer_dims: tuple[int, ...],
    categorical_values: dict[int, Tensor],
    generator: torch.Generator,
    *,
    resample_categorical: bool,
) -> Tensor:
    repaired = population.clone()
    if integer_dims:
        index = torch.tensor(integer_dims, device=population.device)
        repaired[:, index] = repaired[:, index].round()
        repaired[:, index] = torch.maximum(
            torch.minimum(repaired[:, index], upper[index]), lower[index]
        )
    if resample_categorical:
        for dim, values in categorical_values.items():
            legal = (repaired[:, dim, None] == values[None, :]).any(dim=1)
            if not legal.all():
                choices = torch.randint(
                    len(values),
                    (int((~legal).sum()),),
                    device=population.device,
                    generator=generator,
                )
                repaired[~legal, dim] = values[choices]
    return repaired


def _expand_dims(dims: tuple[int, ...], q: int, d: int) -> tuple[int, ...]:
    return tuple(batch * d + dim for batch in range(q) for dim in dims)


def _expand_categories(
    categories: dict[int, Sequence[float]], q: int, d: int, bounds: Tensor
) -> dict[int, Tensor]:
    expanded: dict[int, Tensor] = {}
    for batch in range(q):
        for dim, values in categories.items():
            expanded[batch * d + dim] = torch.as_tensor(
                values, device=bounds.device, dtype=bounds.dtype
            )
    return expanded


def _validate_configuration(
    *,
    q: int,
    bounds: Tensor,
    integer_dims: tuple[int, ...],
    categorical_values: dict[int, Sequence[float]],
    population_size: int,
    generations: int,
    elite_fraction: float,
    tournament_size: int,
    crossover_rate: float,
    mutation_rate: float,
    mutation_scale: float,
) -> None:
    if q < 1:
        raise ValueError("q must be at least 1.")
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError("bounds must have shape [2, d].")
    d = bounds.shape[-1]
    structured_dims = set(integer_dims) | set(categorical_values)
    if any(dim < 0 or dim >= d for dim in structured_dims):
        raise ValueError("Structured dimensions must lie within the input dimension.")
    if set(integer_dims) & set(categorical_values):
        raise ValueError("A dimension cannot be both integer and categorical.")
    if len(integer_dims) != len(set(integer_dims)):
        raise ValueError("integer_dims must not contain duplicates.")
    for dim, values in categorical_values.items():
        if not values:
            raise ValueError(f"categorical_values[{dim}] must not be empty.")
        tensor_values = torch.as_tensor(values, device=bounds.device, dtype=bounds.dtype)
        if torch.any(tensor_values < bounds[0, dim]) or torch.any(tensor_values > bounds[1, dim]):
            raise ValueError(f"categorical_values[{dim}] must lie within bounds.")
    if population_size < 2:
        raise ValueError("population_size must be at least 2.")
    if generations < 1:
        raise ValueError("generations must be at least 1.")
    if not 0.0 < elite_fraction < 1.0:
        raise ValueError("elite_fraction must lie strictly between 0 and 1.")
    if tournament_size < 1 or tournament_size > population_size:
        raise ValueError("tournament_size must lie between 1 and population_size.")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must lie between 0 and 1.")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must lie between 0 and 1.")
    if mutation_scale < 0.0:
        raise ValueError("mutation_scale must be non-negative.")


def _expand_fixed_features(
    fixed_features: Mapping[int, float | Tensor] | None,
    q: int,
    d: int,
    bounds: Tensor,
) -> dict[int, Tensor]:
    expanded: dict[int, Tensor] = {}
    for batch in range(q):
        for dim, value in (fixed_features or {}).items():
            if dim < 0 or dim >= d:
                raise ValueError(f"fixed_features dimension {dim} is out of range.")
            expanded[batch * d + dim] = torch.as_tensor(
                value, device=bounds.device, dtype=bounds.dtype
            ).reshape(())
    return expanded


def _apply_fixed_features(population: Tensor, fixed_features: dict[int, Tensor]) -> Tensor:
    if not fixed_features:
        return population
    result = population.clone()
    for dim, value in fixed_features.items():
        result[:, dim] = value
    return result
