"""Continuous genetic-algorithm acquisition optimizer backend."""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.constraints import CandidateConstraints


def optimize_acqf_ga(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    *,
    population_size: int = 128,
    generations: int = 100,
    elite_fraction: float = 0.1,
    tournament_size: int = 3,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    seed: int | None = None,
    constraints: CandidateConstraints | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize an acquisition function with a real-valued genetic algorithm."""
    _validate_configuration(
        q=q,
        bounds=bounds,
        population_size=population_size,
        generations=generations,
        elite_fraction=elite_fraction,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_scale=mutation_scale,
    )
    candidate_constraints = constraints or CandidateConstraints()
    if candidate_constraints.has_constraints:
        raise ValueError(
            "The Genetic Algorithm backend does not support candidate constraints yet."
        )

    generator = torch.Generator(device=bounds.device)
    if seed is not None:
        generator.manual_seed(seed)
    dimension = q * bounds.shape[-1]
    lower = bounds[0].repeat(q)
    upper = bounds[1].repeat(q)
    population = lower + (upper - lower) * torch.rand(
        population_size,
        dimension,
        dtype=bounds.dtype,
        device=bounds.device,
        generator=generator,
    )
    elite_count = max(1, int(population_size * elite_fraction))

    best_candidate: Tensor | None = None
    best_value: Tensor | None = None
    for generation in range(generations):
        scores = _evaluate_population(acq_function, population, q, bounds.shape[-1])
        generation_best = scores.argmax()
        if best_value is None or scores[generation_best] > best_value:
            best_value = scores[generation_best].detach().clone()
            best_candidate = population[generation_best].detach().clone()
        if generation == generations - 1:
            break

        elite_indices = torch.topk(scores, k=elite_count).indices
        elites = population[elite_indices]
        offspring_count = population_size - elite_count
        parents_a = _tournament_select(
            population, scores, offspring_count, tournament_size, generator
        )
        parents_b = _tournament_select(
            population, scores, offspring_count, tournament_size, generator
        )
        offspring = _crossover(parents_a, parents_b, crossover_rate, generator)
        offspring = _mutate(
            offspring, lower, upper, mutation_rate, mutation_scale, generator
        )
        population = torch.cat([elites, offspring], dim=0)

    if best_candidate is None or best_value is None:
        raise RuntimeError("Genetic Algorithm did not generate any candidate.")
    return best_candidate.reshape(q, bounds.shape[-1]), best_value.reshape(())


def _evaluate_population(
    acq_function: AcquisitionFunction, population: Tensor, q: int, d: int
) -> Tensor:
    with torch.no_grad():
        values = acq_function(population.reshape(population.shape[0], q, d))
    if values.numel() != population.shape[0]:
        raise ValueError(
            "Genetic Algorithm requires one scalar acquisition value per population member."
        )
    return values.reshape(population.shape[0])


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
    mask = torch.rand(
        parents_a.shape,
        dtype=parents_a.dtype,
        device=parents_a.device,
        generator=generator,
    ) < crossover_rate
    alpha = torch.rand(
        parents_a.shape,
        dtype=parents_a.dtype,
        device=parents_a.device,
        generator=generator,
    )
    blended = alpha * parents_a + (1.0 - alpha) * parents_b
    return torch.where(mask, blended, parents_a)

def _mutate(
    offspring: Tensor,
    lower: Tensor,
    upper: Tensor,
    mutation_rate: float,
    mutation_scale: float,
    generator: torch.Generator,
) -> Tensor:
    mask = torch.rand(
        offspring.shape,
        dtype=offspring.dtype,
        device=offspring.device,
        generator=generator,
    ) < mutation_rate
    noise = torch.randn(
        offspring.shape,
        dtype=offspring.dtype,
        device=offspring.device,
        generator=generator,
    )
    mutated = offspring + mask * noise * mutation_scale * (upper - lower)
    return torch.maximum(torch.minimum(mutated, upper), lower)


def _validate_configuration(
    *,
    q: int,
    bounds: Tensor,
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
