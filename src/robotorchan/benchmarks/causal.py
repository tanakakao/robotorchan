"""Seed-paired known-DAG causal intervention benchmark with cost curves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from torch import Tensor

from robotorchan.benchmarks.random_streams import make_stream_generator
from robotorchan.benchmarks.strategy_rng import strategy_rng_scope
from robotorchan.causal import (
    CausalBOStudy,
    CausalGraph,
    CausalObservation,
    HardIntervention,
    StructuralCausalModel,
)
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.optim.dispatch import OptimizerName, optimize_acqf

METHODS = ("cbo", "gp_bo", "random")


@dataclass(frozen=True, slots=True)
class CausalBenchmarkScenario:
    """Deterministic SCM with a known optimum and three allowed intervention sets."""

    graph: CausalGraph
    scm: StructuralCausalModel
    bounds: Mapping[str, tuple[float, float]]
    intervention_sets: tuple[tuple[str, ...], ...]
    costs: Mapping[frozenset[str], float]
    optimal_value: float = 0.0

    def evaluate(self, intervention: HardIntervention) -> CausalObservation:
        """Obtain the interventional ground truth, not observational conditioning."""
        return CausalObservation(self.scm.evaluate(intervention), intervention)


@dataclass(frozen=True, slots=True)
class CausalBenchmarkStep:
    """An actual hard-intervention evaluation, including shared initialization."""

    evaluation: int
    is_initial: bool
    intervention_nodes: tuple[str, ...]
    intervention_values: tuple[float, ...]
    outcome_truth: float
    best_truth: float
    simple_regret: float
    evaluation_cost: float
    cumulative_cost: float


@dataclass(frozen=True, slots=True)
class CausalBenchmarkTrajectory:
    """One strategy/seed trajectory with variable-sized intervention coordinates."""

    method: str
    seed: int
    initial_points: int
    steps: tuple[CausalBenchmarkStep, ...]

    @property
    def best_curve(self) -> Tensor:
        return torch.tensor([step.best_truth for step in self.steps], dtype=torch.double)

    @property
    def regret_curve(self) -> Tensor:
        return torch.tensor([step.simple_regret for step in self.steps], dtype=torch.double)

    @property
    def cost_curve(self) -> Tensor:
        return torch.tensor([step.cumulative_cost for step in self.steps], dtype=torch.double)


def make_causal_synthetic_scenario(
    *, cost_x: float = 1.0, cost_z: float = 1.5, cost_joint: float = 3.0
) -> CausalBenchmarkScenario:
    """Return X -> M -> Y <- Z, with known max Y=0 over the intervention domain."""
    if any(
        isinstance(cost, bool) or not isfinite(float(cost)) or cost <= 0
        for cost in (cost_x, cost_z, cost_joint)
    ):
        raise ValueError("Intervention costs must be finite and positive.")
    graph = CausalGraph({"X": (), "Z": (), "M": ("X",), "Y": ("M", "Z")})
    scm = StructuralCausalModel(
        graph,
        {
            "X": lambda parents: 0.2,
            "Z": lambda parents: 0.3,
            "M": lambda parents: 0.25 + 0.75 * parents["X"],
            "Y": lambda parents: -((parents["M"] - 0.8) ** 2 + 0.5 * (parents["Z"] - 0.65) ** 2),
        },
    )
    return CausalBenchmarkScenario(
        graph=graph,
        scm=scm,
        bounds={"X": (0.0, 1.0), "Z": (0.0, 1.0)},
        intervention_sets=(("X",), ("Z",), ("X", "Z")),
        costs={
            frozenset({"X"}): float(cost_x),
            frozenset({"Z"}): float(cost_z),
            frozenset({"X", "Z"}): float(cost_joint),
        },
    )


def _gp_bo_intervention(
    scenario: CausalBenchmarkScenario,
    records: Sequence[CausalObservation],
    *,
    optimizer: OptimizerName,
    num_restarts: int,
    raw_samples: int,
) -> HardIntervention:
    """Conventional GP-BO restricted to the fixed full-intervention design space."""
    nodes = scenario.intervention_sets[-1]
    key = frozenset(nodes)
    relevant = [record for record in records if frozenset(record.intervention.values) == key]
    train_X = torch.tensor(
        [[record.intervention.values[name] for name in nodes] for record in relevant],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[record.values["Y"]] for record in relevant], dtype=torch.double)
    model = SingleTaskGP(train_X, train_Y)
    model.fit()
    model.eval()
    acquisition = LogExpectedImprovement(model, best_f=float(train_Y.max()))
    bounds = torch.tensor([scenario.bounds[name] for name in nodes], dtype=torch.double).T
    candidate, _ = optimize_acqf(
        acquisition,
        bounds,
        q=1,
        optimizer=optimizer,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return HardIntervention(dict(zip(nodes, candidate[0].tolist(), strict=True)))


def _random_intervention(
    scenario: CausalBenchmarkScenario, generator: torch.Generator
) -> HardIntervention:
    """Sample an intervention set uniformly, then sample all of its values."""
    index = int(torch.randint(len(scenario.intervention_sets), (1,), generator=generator))
    nodes = scenario.intervention_sets[index]
    uniform = torch.rand(len(nodes), generator=generator, dtype=torch.double)
    return HardIntervention(
        {
            node: lower + float(unit) * (upper - lower)
            for node, unit in zip(nodes, uniform, strict=True)
            for lower, upper in (scenario.bounds[node],)
        }
    )


def _make_steps(
    records: Sequence[CausalObservation],
    scenario: CausalBenchmarkScenario,
    initial_points: int,
) -> tuple[CausalBenchmarkStep, ...]:
    steps: list[CausalBenchmarkStep] = []
    best = -float("inf")
    cumulative = 0.0
    for index, record in enumerate(records):
        if record.intervention is None:
            raise ValueError("A causal benchmark requires performed interventions.")
        nodes = tuple(
            node for node in scenario.graph.topological_order if node in record.intervention.values
        )
        outcome = float(record.values["Y"])
        best = max(best, outcome)
        cost = scenario.costs[frozenset(nodes)]
        cumulative += cost
        steps.append(
            CausalBenchmarkStep(
                evaluation=index + 1,
                is_initial=index < initial_points,
                intervention_nodes=nodes,
                intervention_values=tuple(record.intervention.values[node] for node in nodes),
                outcome_truth=outcome,
                best_truth=best,
                simple_regret=max(0.0, scenario.optimal_value - best),
                evaluation_cost=cost,
                cumulative_cost=cumulative,
            )
        )
    return tuple(steps)


def run_causal_benchmark(
    *,
    seeds: Sequence[int] = (0, 1, 2),
    n_initial_per_set: int = 2,
    n_iterations: int = 5,
    scenario: CausalBenchmarkScenario | None = None,
    optimizer: OptimizerName = "botorch",
    num_restarts: int = 3,
    raw_samples: int = 32,
) -> tuple[CausalBenchmarkTrajectory, ...]:
    """Compare CBO, fixed-full-intervention GP-BO and Random Search per seed.

    All three methods begin with exactly identical hard-intervention records.
    Trial counts are paired; total financial cost is intentionally *reported*,
    not fixed to the same ceiling. The conventional GP only trains on the
    full-set subset of shared initial data, avoiding invalid do/conditional
    mixing. Returns actual truth curves, not fabricated performance results.
    """
    seed_list = tuple(seeds)
    if not seed_list or any(type(seed) is not int or seed < 0 for seed in seed_list):
        raise ValueError("seeds must contain nonnegative integers.")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("seeds must be unique.")
    if type(n_initial_per_set) is not int or n_initial_per_set < 2:
        raise ValueError("n_initial_per_set must be at least 2.")
    if type(n_iterations) is not int or n_iterations < 0:
        raise ValueError("n_iterations must be nonnegative.")
    if type(num_restarts) is not int or num_restarts < 1:
        raise ValueError("num_restarts must be positive.")
    if type(raw_samples) is not int or raw_samples < 1:
        raise ValueError("raw_samples must be positive.")

    problem = make_causal_synthetic_scenario() if scenario is None else scenario
    initial_points = n_initial_per_set * len(problem.intervention_sets)
    result: list[CausalBenchmarkTrajectory] = []
    for seed in seed_list:
        study = CausalBOStudy(
            problem.graph,
            "Y",
            problem.intervention_sets,
            problem.bounds,
            intervention_costs=problem.costs,
            n_initial_trials=n_initial_per_set,
            optimizer=optimizer,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            seed=seed,
        )
        for _ in range(initial_points):
            trial = study.ask()
            observation = problem.evaluate(trial.intervention)
            study.tell(trial, observation.values)
        initial_records = study.observations

        for method in METHODS:
            records = list(initial_records)
            generator = make_stream_generator(seed, "causal_random", torch.device("cpu"))
            for iteration in range(n_iterations):
                with strategy_rng_scope(seed, iteration, torch.device("cpu")):
                    if method == "cbo":
                        trial = study.ask()
                        intervention = trial.intervention
                    elif method == "gp_bo":
                        intervention = _gp_bo_intervention(
                            problem,
                            records,
                            optimizer=optimizer,
                            num_restarts=num_restarts,
                            raw_samples=raw_samples,
                        )
                    else:
                        intervention = _random_intervention(problem, generator)
                observation = problem.evaluate(intervention)
                if method == "cbo":
                    study.tell(trial, observation.values)
                records.append(observation)

            result.append(
                CausalBenchmarkTrajectory(
                    method=method,
                    seed=seed,
                    initial_points=initial_points,
                    steps=_make_steps(records, problem, initial_points),
                )
            )
    return tuple(result)
