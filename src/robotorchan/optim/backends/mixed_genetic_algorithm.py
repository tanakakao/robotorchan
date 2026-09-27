"""Mixed-variable genetic-algorithm acquisition optimizer backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.constraint_evaluation import candidate_constraint_violation
from robotorchan.optim.constraints import CandidateConstraints
from robotorchan.optim.runtime import make_generator, validate_bounds


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
    constraint_penalty: float = 1e6,
    equality_tolerance: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Optimize an acquisition function over continuous, integer, and categorical inputs."""
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
    if constraint_penalty <= 0:
        raise ValueError("constraint_penalty must be positive.")

    generator = make_generator(bounds, seed)
    d = bounds.shape[-1]
    lower = bounds[0].repeat(q)
    upper = bounds[1].repeat(q)
    expanded_integer_dims = _expand_dims(integer_dims, q, d)
    expanded_categories = _expand_categories(categorical_values, q, d, bounds)

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
    elite_count = max(1, int(population_size * elite_fraction))

    best_candidate: Tensor | None = None
    best_value: Tensor | None = None
    for generation in range(generations):
        scores = _evaluate_population(
            acq_function,
            population,
            q,
            d,
            candidate_constraints,
            constraint_penalty,
            equality_tolerance,
        )
        generation_best = scores.argmax()
        if best_value is None or scores[generation_best] > best_value:
            best_value = scores[generation_best].detach().clone()
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
        offspring = _crossover(parents_a, parents_b, crossover_rate, generator)
        offspring = _mutate(offspring, lower, upper, mutation_rate, mutation_scale, generator)
        offspring = _repair_structured(
            offspring,
            lower,
            upper,
            expanded_integer_dims,
            expanded_categories,
            generator,
            resample_categorical=False,
        )
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
    constraint_penalty: float,
    equality_tolerance: float,
) -> Tensor:
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
    return scores - constraint_penalty * violation


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
) -> Tensor:
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
    return torch.where(mask, alpha * parents_a + (1.0 - alpha) * parents_b, parents_a)


def _mutate(
    offspring: Tensor,
    lower: Tensor,
    upper: Tensor,
    mutation_rate: float,
    mutation_scale: float,
    generator: torch.Generator,
) -> Tensor:
    mask = (
        torch.rand(
            offspring.shape,
            dtype=offspring.dtype,
            device=offspring.device,
            generator=generator,
        )
        < mutation_rate
    )
    noise = torch.randn(
        offspring.shape,
        dtype=offspring.dtype,
        device=offspring.device,
        generator=generator,
    )
    return torch.maximum(
        torch.minimum(offspring + mask * noise * mutation_scale * (upper - lower), upper),
        lower,
    )


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
        integer_lower = torch.ceil(lower[index])
        integer_upper = torch.floor(upper[index])
        repaired[:, index] = repaired[:, index].round()
        repaired[:, index] = torch.maximum(
            torch.minimum(repaired[:, index], integer_upper), integer_lower
        )
    for dim, values in categorical_values.items():
        if resample_categorical:
            choices = torch.randint(
                len(values),
                (population.shape[0],),
                device=population.device,
                generator=generator,
            )
            repaired[:, dim] = values[choices]
        else:
            distances = (repaired[:, dim, None] - values[None, :]).abs()
            repaired[:, dim] = values[distances.argmin(dim=-1)]
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
    for dim in integer_dims:
        if torch.ceil(bounds[0, dim]) > torch.floor(bounds[1, dim]):
            raise ValueError(f"integer dimension {dim} has no legal value within bounds.")
    for dim, values in categorical_values.items():
        tensor_values = torch.as_tensor(values, device=bounds.device, dtype=bounds.dtype)
        if tensor_values.ndim != 1 or tensor_values.numel() == 0:
            raise ValueError(
                f"categorical_values[{dim}] must be a non-empty one-dimensional sequence."
            )
        if not torch.isfinite(tensor_values).all():
            raise ValueError(f"categorical_values[{dim}] must contain only finite values.")
        if torch.unique(tensor_values).numel() != tensor_values.numel():
            raise ValueError(f"categorical_values[{dim}] must not contain duplicates.")
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
