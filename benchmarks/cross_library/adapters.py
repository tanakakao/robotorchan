"""Public API adapters for the cross-library benchmark, loaded on demand."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from math import ceil, floor
from time import perf_counter
from typing import Any, Protocol

import torch
from torch import Tensor

from robotorchan.benchmarks.candidate_view import CandidateProblemView
from robotorchan.benchmarks.comparative_baselines import sobol_candidates
from robotorchan.benchmarks.runner import random_candidates


class UnsupportedConfiguration(ValueError):
    """The selected adapter does not implement this benchmark contract."""


class Adapter(Protocol):
    algorithm: str
    batch_method: str

    def ask(self, q: int, rng: torch.Generator) -> Tensor: ...

    def tell(self, x: Tensor, y: Tensor, constraints: Tensor, *, initial: bool) -> None: ...


class Space:
    """Exchange tensors with named parameters, preserving nominal categories."""

    def __init__(self, problem: CandidateProblemView) -> None:
        self.problem = problem
        self.names = tuple(f"x{i}" for i in range(problem.dimension))
        self.bounds = list(zip(*problem.bounds.tolist(), strict=True))
        self.objectives = tuple(f"y{i}" for i in range(problem.n_objectives))
        self.constraints = tuple(f"g{i}" for i in range(problem.n_constraints))

    def levels(self, i: int) -> range:
        lo, hi = self.bounds[i]
        return range(ceil(lo), floor(hi) + 1)

    @staticmethod
    def category(value: int) -> str:
        return f"category_{value}"

    def parameters(self, row: list[float]) -> dict:
        params = {}
        for name, kind, value in zip(self.names, self.problem.variable_types, row, strict=True):
            if kind == "categorical":
                value = self.category(int(value))
            elif kind == "integer":
                value = int(value)
            params[name] = value
        return params

    def candidates(self, rows: list[Mapping[str, Any]]) -> Tensor:
        return self.problem.bounds.new_tensor(
            [
                [
                    int(row[name].removeprefix("category_")) if kind == "categorical" else row[name]
                    for name, kind in zip(self.names, self.problem.variable_types, strict=True)
                ]
                for row in rows
            ]
        )

    def measurements(self, x: Tensor, y: Tensor, constraints: Tensor) -> list[dict]:
        return [
            self.parameters(a)
            | dict(zip(self.objectives, b, strict=True))
            | dict(zip(self.constraints, c, strict=True))
            for a, b, c in zip(x.tolist(), y.tolist(), constraints.tolist(), strict=True)
        ]


class Baseline:
    batch_method = "independent"

    def __init__(self, space: Space, name: str) -> None:
        self.problem = space.problem
        self.algorithm = name
        self.initial_points = 0
        self.count = 0

    def ask(self, q: int, rng: torch.Generator) -> Tensor:
        x = self.problem.bounds.new_empty((self.count, self.problem.dimension))
        if self.algorithm == "random":
            return random_candidates(self.problem, x, x, q, rng)
        return sobol_candidates(self.problem, x, x, q, rng, initial_points=self.initial_points)

    def tell(self, x: Tensor, y: Tensor, constraints: Tensor, *, initial: bool) -> None:
        if initial:
            self.initial_points = len(x)
        self.count += len(x)


class GP:
    """Matched low-level model and acquisition paths for BoTorch and robotorchan."""

    def __init__(
        self, space: Space, *, robotorchan: bool, noisy: bool, restarts: int, raw: int, mc: int
    ) -> None:
        if space.problem.n_objectives > 2:
            raise UnsupportedConfiguration("The GP adapter supports one or two objectives.")
        if space.problem.n_objectives == 2 and space.problem.reference_point is None:
            raise UnsupportedConfiguration("Two objectives require a reference point.")
        self.space = space
        self.robotorchan = robotorchan
        self.restarts, self.raw, self.mc = restarts, raw, mc
        self.fit_seconds = 0.0
        self.x: Tensor | None = None
        self.y: Tensor | None = None
        discrete = [
            i for i, kind in enumerate(space.problem.variable_types) if kind != "continuous"
        ]
        self.fixed = (
            [
                dict(zip(discrete, values, strict=True))
                for values in product(*(space.levels(i) for i in discrete))
            ]
            if discrete
            else []
        )
        self.categorical = [
            i for i, kind in enumerate(space.problem.variable_types) if kind == "categorical"
        ]
        self.batch_method = "sequential_pending" if self.fixed else "joint_acquisition"
        self.algorithm = (
            "qLogNEHVI"
            if space.problem.n_objectives == 2
            else "qLogNEI"
            if noisy or space.constraints
            else "qLogEI"
        )

    def tell(self, x: Tensor, y: Tensor, constraints: Tensor, *, initial: bool) -> None:
        signs = y.new_tensor([1 if d == "maximize" else -1 for d in self.space.problem.directions])
        outputs = torch.cat((y * signs, -constraints), dim=-1)
        self.x = x if self.x is None else torch.cat((self.x, x))
        self.y = outputs if self.y is None else torch.cat((self.y, outputs))

    def ask(self, q: int, rng: torch.Generator) -> Tensor:
        from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement,
        )
        from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
        from botorch.acquisition.objective import GenericMCObjective
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import MixedSingleTaskGP, ModelListGP, SingleTaskGP
        from botorch.models.transforms.input import Normalize
        from botorch.optim import optimize_acqf, optimize_acqf_mixed
        from botorch.sampling.normal import SobolQMCNormalSampler
        from gpytorch.mlls import ExactMarginalLogLikelihood

        if self.x is None or self.y is None:
            raise RuntimeError("Initial observations are required.")
        x, y = self.x, self.y
        problem = self.space.problem
        model_cls = MixedSingleTaskGP if self.categorical else SingleTaskGP
        if self.robotorchan:
            from robotorchan.models import MixedSingleTaskGP as RobotorchanMixed
            from robotorchan.models import SingleTaskGP as RobotorchanGP

            model_cls = RobotorchanMixed if self.categorical else RobotorchanGP
        kwargs = {"cat_dims": self.categorical} if self.categorical else {}
        continuous = [i for i in range(problem.dimension) if i not in self.categorical]
        models = []
        for column in y.split(1, dim=-1):
            transform = (
                Normalize(problem.dimension, bounds=problem.bounds, indices=continuous)
                if continuous
                else None
            )
            model = model_cls(x, column, input_transform=transform, **kwargs)
            started = perf_counter()
            try:
                if self.robotorchan:
                    model.fit()
                else:
                    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            finally:
                self.fit_seconds += perf_counter() - started
            models.append(model)
        model = models[0] if len(models) == 1 else ModelListGP(*models)
        sampler = SobolQMCNormalSampler(
            torch.Size([self.mc]), seed=int(torch.randint(0, 2**31 - 1, (), generator=rng))
        )
        width = problem.n_objectives
        constraints = [
            lambda samples, index=index: samples[..., index] for index in range(width, y.shape[-1])
        ] or None
        if self.algorithm == "qLogNEHVI":
            signs = y.new_tensor([1 if d == "maximize" else -1 for d in problem.directions])
            acq = qLogNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=(problem.reference_point * signs).tolist(),
                X_baseline=x,
                sampler=sampler,
                objective=IdentityMCMultiOutputObjective(list(range(width))),
                constraints=constraints,
            )
        elif self.algorithm == "qLogNEI":
            acq = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=x,
                sampler=sampler,
                constraints=constraints,
                objective=GenericMCObjective(lambda samples, X=None: samples[..., 0]),
            )
        else:
            acq = qLogExpectedImprovement(model=model, best_f=y.max(), sampler=sampler)
        args = dict(
            acq_function=acq,
            bounds=problem.bounds,
            q=q,
            num_restarts=self.restarts,
            raw_samples=self.raw,
        )
        if self.fixed:
            return optimize_acqf_mixed(**args, fixed_features_list=self.fixed)[0].detach()
        return optimize_acqf(**args)[0].detach()


class RobotorchanStudy:
    algorithm = "Study:qLogEI"
    batch_method = "joint_acquisition"

    def __init__(self, space: Space, seed: int, restarts: int, raw: int) -> None:
        from robotorchan.study import create_study

        if (
            space.constraints
            or len(space.objectives) != 1
            or any(kind != "continuous" for kind in space.problem.variable_types)
        ):
            raise UnsupportedConfiguration("Scalar Study adapter requires continuous inputs.")
        self.space = space
        self.study = create_study(
            search_space=dict(zip(space.names, space.bounds, strict=True)),
            direction=space.problem.directions[0],
            seed=seed,
            optimizer_kwargs={"num_restarts": restarts, "raw_samples": raw},
        )
        self.pending = ()

    def ask(self, q: int, rng: torch.Generator) -> Tensor:
        self.pending = self.study.ask_batch(q)
        return self.space.candidates([trial.params for trial in self.pending])

    def tell(self, x: Tensor, y: Tensor, constraints: Tensor, *, initial: bool) -> None:
        if initial:
            for row, value in zip(x.tolist(), y[:, 0].tolist(), strict=True):
                trial = self.study.add_trial(self.space.parameters(row))
                self.study.start_trial(trial.number)
                self.study.tell(trial.number, value)
        else:
            for trial, value in zip(self.pending, y[:, 0].tolist(), strict=True):
                self.study.tell(trial, value)
            self.pending = ()


class Optuna:
    batch_method = "sequential_pending"

    def __init__(self, space: Space, seed: int, *, gp: bool = False) -> None:
        import optuna
        from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution

        self.space = space
        self.pending = []
        self.algorithm = "GPSampler" if gp else "TPESampler"
        sampler = (
            optuna.samplers.GPSampler(seed=seed) if gp else optuna.samplers.TPESampler(seed=seed)
        )
        self.study = optuna.create_study(directions=space.problem.directions, sampler=sampler)
        self.distributions = {}
        for i, (name, kind) in enumerate(
            zip(space.names, space.problem.variable_types, strict=True)
        ):
            lo, hi = space.bounds[i]
            if kind == "continuous":
                dist = FloatDistribution(lo, hi)
            elif kind == "integer":
                dist = IntDistribution(ceil(lo), floor(hi))
            else:
                dist = CategoricalDistribution([space.category(j) for j in space.levels(i)])
            self.distributions[name] = dist
        if gp:
            self.batch_method = "repeated_ask"

    def ask(self, q: int, rng: torch.Generator) -> Tensor:
        self.pending = [self.study.ask(self.distributions) for _ in range(q)]
        return self.space.candidates([trial.params for trial in self.pending])

    def tell(self, x: Tensor, y: Tensor, constraints: Tensor, *, initial: bool) -> None:
        from optuna.trial import create_trial

        for i, (row, values, residuals) in enumerate(
            zip(x.tolist(), y.tolist(), (-constraints).tolist(), strict=True)
        ):
            metrics = dict(zip(self.space.constraints, residuals, strict=True))
            if initial:
                self.study.add_trial(
                    create_trial(
                        params=self.space.parameters(row),
                        distributions=self.distributions,
                        values=values,
                        constraints=metrics or None,
                    )
                )
            else:
                trial = self.pending[i]
                for name, residual in metrics.items():
                    trial.set_constraint(name, residual)
                self.study.tell(trial, values)
        self.pending = []


class Ax:
    algorithm = "Client:default_generation_strategy"
    batch_method = "multiple_trials"

    def __init__(self, space: Space, seed: int) -> None:
        from ax.api.client import Client
        from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

        self.space = space
        self.client = Client(random_seed=seed)
        self.pending = []
        parameters = []
        for i, (name, kind) in enumerate(
            zip(space.names, space.problem.variable_types, strict=True)
        ):
            lo, hi = space.bounds[i]
            if kind == "categorical":
                param = ChoiceParameterConfig(
                    name=name,
                    values=[space.category(j) for j in space.levels(i)],
                    parameter_type="str",
                    is_ordered=False,
                )
            elif kind == "integer":
                param = RangeParameterConfig(
                    name=name, bounds=(ceil(lo), floor(hi)), parameter_type="int"
                )
            else:
                param = RangeParameterConfig(name=name, bounds=(lo, hi), parameter_type="float")
            parameters.append(param)
        self.client.configure_experiment(parameters=parameters)
        objective = ", ".join(
            name if direction == "maximize" else f"-{name}"
            for name, direction in zip(space.objectives, space.problem.directions, strict=True)
        )
        self.client.configure_optimization(
            objective=objective, outcome_constraints=[f"{k} >= 0" for k in space.constraints]
        )

    def ask(self, q: int, rng: torch.Generator) -> Tensor:
        candidates = self.client.get_next_trials(max_trials=q)
        self.pending = list(candidates)
        return self.space.candidates(list(candidates.values()))

    def tell(self, x: Tensor, y: Tensor, constraints: Tensor, *, initial: bool) -> None:
        for i, row in enumerate(self.space.measurements(x, y, constraints)):
            trial = (
                self.client.attach_trial(parameters={name: row[name] for name in self.space.names})
                if initial
                else self.pending[i]
            )
            self.client.complete_trial(
                trial,
                raw_data={k: row[k] for k in (*self.space.objectives, *self.space.constraints)},
            )
        self.pending = []


class FrameAdapter:
    """Bridge public table-based recommend/observe APIs."""

    def tell(self, x: Tensor, y: Tensor, constraints: Tensor, *, initial: bool) -> None:
        import pandas as pd

        self.observe(pd.DataFrame(self.space.measurements(x, y, constraints)))

    def ask(self, q: int, rng: torch.Generator) -> Tensor:
        return self.space.candidates(self.recommend(q).to_dict(orient="records"))


class BayBE(FrameAdapter):
    algorithm = "Campaign:default_recommender"

    def __init__(self, space: Space) -> None:
        if space.constraints:
            raise UnsupportedConfiguration("BayBE adapter does not support observed constraints.")

        from baybe import Campaign
        from baybe.objectives import ParetoObjective, SingleTargetObjective
        from baybe.parameters import (
            CategoricalParameter,
            NumericalContinuousParameter,
            NumericalDiscreteParameter,
        )
        from baybe.searchspace import SearchSpace
        from baybe.targets import NumericalTarget

        self.space = space
        self.batch_method = "library_batch"
        parameters = []
        for i, (name, kind) in enumerate(
            zip(space.names, space.problem.variable_types, strict=True)
        ):
            if kind == "categorical":
                param = CategoricalParameter(
                    name=name, values=[space.category(j) for j in space.levels(i)]
                )
            elif kind == "integer":
                param = NumericalDiscreteParameter(name=name, values=list(space.levels(i)))
            else:
                param = NumericalContinuousParameter(name=name, bounds=space.bounds[i])
            parameters.append(param)
        targets = [
            NumericalTarget(name=name, minimize=direction == "minimize")
            for name, direction in zip(space.objectives, space.problem.directions, strict=True)
        ]
        objective = (
            SingleTargetObjective(targets[0]) if len(targets) == 1 else ParetoObjective(targets)
        )
        campaign = Campaign(searchspace=SearchSpace.from_product(parameters), objective=objective)
        self.recommend = campaign.recommend
        self.observe = campaign.add_measurements


class BoFire(FrameAdapter):
    batch_method = "library_batch"

    def __init__(self, space: Space, seed: int) -> None:
        from bofire.data_models.domain.api import Domain
        from bofire.data_models.features.api import (
            CategoricalInput,
            ContinuousInput,
            ContinuousOutput,
            DiscreteInput,
        )
        from bofire.data_models.objectives.api import (
            MaximizeObjective,
            MaximizeSigmoidObjective,
            MinimizeObjective,
        )
        from bofire.data_models.strategies.api import MoboStrategy, SoboStrategy
        from bofire.strategies.api import map as map_strategy

        self.space = space
        inputs = []
        for i, (name, kind) in enumerate(
            zip(space.names, space.problem.variable_types, strict=True)
        ):
            if kind == "categorical":
                feature = CategoricalInput(
                    key=name, categories=[space.category(j) for j in space.levels(i)]
                )
            elif kind == "integer":
                feature = DiscreteInput(key=name, values=list(space.levels(i)))
            else:
                feature = ContinuousInput(key=name, bounds=space.bounds[i])
            inputs.append(feature)
        outputs = [
            ContinuousOutput(
                key=name,
                objective=MaximizeObjective() if direction == "maximize" else MinimizeObjective(),
            )
            for name, direction in zip(space.objectives, space.problem.directions, strict=True)
        ]
        outputs.extend(
            ContinuousOutput(key=name, objective=MaximizeSigmoidObjective(tp=0, steepness=10))
            for name in space.constraints
        )
        cls = SoboStrategy if len(space.objectives) == 1 else MoboStrategy
        strategy = map_strategy(
            cls(domain=Domain.from_lists(inputs=inputs, outputs=outputs), seed=seed)
        )
        self.algorithm = cls.__name__
        self.recommend = strategy.ask
        self.observe = strategy.tell
