"""NSGA-II backend for genuinely vector-valued optimization targets."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

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
) -> tuple[Tensor, Tensor]:
    """Approximate a Pareto set with a native, batched NSGA-II implementation.

    The objective receives [population, d] and must return [population, m].
    All objectives are maximized. This API is intentionally separate from
    scalar BoTorch acquisition optimization: qEHVI/qNEHVI already produce
    scalar acquisition values and do not require NSGA-II.
    """
    _validate(bounds, population_size, generations, crossover_rate, mutation_rate, mutation_scale)
    generator = torch.Generator(device=bounds.device)
    if seed is not None:
        generator.manual_seed(seed)
    lower, upper = bounds
    population = lower + (upper - lower) * torch.rand(
        population_size,
        bounds.shape[-1],
        dtype=bounds.dtype,
        device=bounds.device,
        generator=generator,
    )
    for generation in range(generations):
        values = _evaluate(objective, population)
        if generation == generations - 1:
            break
        ranks, crowding = _rank_and_crowding(values)
        parents_a = _binary_tournament(population, ranks, crowding, generator)
        parents_b = _binary_tournament(population, ranks, crowding, generator)
        offspring = _crossover(parents_a, parents_b, crossover_rate, generator)
        offspring = _mutate(offspring, lower, upper, mutation_rate, mutation_scale, generator)
        combined = torch.cat([population, offspring], dim=0)
        combined_values = _evaluate(objective, combined)
        population = _environmental_select(combined, combined_values, population_size)
    values = _evaluate(objective, population)
    ranks, _ = _rank_and_crowding(values)
    mask = ranks == 0
    return population[mask], values[mask]


def _evaluate(objective: VectorObjective, population: Tensor) -> Tensor:
    with torch.no_grad():
        values = objective(population)
    if values.ndim != 2 or values.shape[0] != population.shape[0]:
        raise ValueError("NSGA-II objective must return shape [population, objectives].")
    if values.shape[1] < 2:
        raise ValueError("NSGA-II requires at least two objective values.")
    return values


def _rank_and_crowding(values: Tensor) -> tuple[Tensor, Tensor]:
    n = values.shape[0]
    dominates = torch.zeros((n, n), dtype=torch.bool, device=values.device)
    for i in range(n):
        no_worse = (values[i] >= values).all(dim=1)
        strictly_better = (values[i] > values).any(dim=1)
        dominates[i] = no_worse & strictly_better
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


def _environmental_select(population: Tensor, values: Tensor, size: int) -> Tensor:
    ranks, crowding = _rank_and_crowding(values)
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
    mask = (\n        torch.rand(\n            parents_a.shape,\n            dtype=parents_a.dtype,\n            device=parents_a.device,\n            generator=generator,\n        )\n        < rate\n    )
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
    mask = (\n        torch.rand(\n            offspring.shape,\n            dtype=offspring.dtype,\n            device=offspring.device,\n            generator=generator,\n        )\n        < rate\n    )
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
