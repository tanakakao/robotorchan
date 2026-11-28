"""Decoupled materials search with strength, conductivity and pass outcomes."""

from time import perf_counter

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.acquisition.factory import create_acquisition
from robotorchan.benchmarks.multiobjective_heterogeneous_problems import (
    multiobjective_pass_labels,
    strength_conductivity_pass_tradeoff,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.optim.decoupled import optimize_acqf_decoupled
from robotorchan.workflow.cost_budget import CostBudget


def run_materials_decoupled_benchmark(
    *,
    seed: int = 0,
    budget: int = 60,
    n_initial: int = 4,
    num_samples: int = 8,
    epochs: int = 30,
    decoupled: bool = True,
    fit_regression: bool = True,
    strength_weight: float = 0.5,
) -> list[dict]:
    """Compare masked and full evaluations for a fixed material-property utility.

    The binary quality outcome is observed separately from the two properties.
    Oracle responses at recommendations are used only for scoring.
    """
    if (
        n_initial < 2
        or budget < 9 * n_initial
        or min(num_samples, epochs) < 1
        or not 0 <= strength_weight <= 1
    ):
        raise ValueError("Invalid materials benchmark configuration.")
    problem = strength_conductivity_pass_tradeoff()
    bounds = problem.bounds
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    extra = engine.draw(n_initial - 2).to(bounds) if n_initial > 2 else bounds[:0]
    X = torch.cat((bounds, extra))
    target = torch.cat((bounds, engine.draw(32).to(bounds)))
    Y = problem.evaluate_truth(X)
    labels = multiobjective_pass_labels(X).squeeze(-1).long()
    train_X = [X.clone() for _ in range(3)]
    train_Y = [Y[:, :1], Y[:, 1:], labels]
    costs = bounds.new_tensor([3, 5, 1])
    weights = bounds.new_tensor([strength_weight, 1 - strength_weight])
    full = torch.ones((1, 3), dtype=torch.bool)
    masks = torch.cat((torch.eye(3, dtype=torch.bool), full)) if decoupled else full
    ledger = CostBudget(budget, spent=9 * n_initial)
    records = []
    elapsed = 0.0

    def utility(samples):
        strength, conductivity, probabilities = samples
        value = weights[0] * strength.squeeze(-1) + weights[1] * conductivity.squeeze(-1)
        return probabilities[..., -1] * (value + 1) - 1

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        while True:
            start = perf_counter()
            regressions = [SingleTaskGP(train_X[i], train_Y[i]) for i in range(2)]
            if fit_regression:
                for regression in regressions:
                    fit_gpytorch_mll(ExactMarginalLogLikelihood(regression.likelihood, regression))
            classifier = BinarySingleTaskGPClassifier(
                train_X[2], train_Y[2], inducing_points=min(6, len(train_Y[2]))
            )
            classifier.fit(epochs=epochs, lr=0.01)
            model = HeterogeneousModel(*regressions, classifier)
            model.eval()
            with torch.no_grad():
                means = [regression.posterior(target).mean for regression in regressions]
                probabilities = classifier.probability_posterior(target).mean
                index = int(utility((*means, probabilities)).argmax())
                recommendation = target[index : index + 1]
                truth = problem.evaluate_truth(recommendation).squeeze(0)
                feasible = bool(multiobjective_pass_labels(recommendation).item())
                payoff = float(truth @ weights) if feasible else -1.0
                labels = multiobjective_pass_labels(target).squeeze(-1).long()
                one_hot = torch.nn.functional.one_hot(labels, num_classes=2).to(probabilities)
                brier = float((probabilities - one_hot).square().sum(dim=-1).mean())
            elapsed += perf_counter() - start
            records.append(
                {
                    "seed": seed,
                    "method": "decoupled_particle_kg" if decoupled else "coupled_particle_kg",
                    "strength_weight": strength_weight,
                    "cost": ledger.spent,
                    "strength": float(truth[0]),
                    "conductivity": float(truth[1]),
                    "feasible": feasible,
                    "utility": payoff,
                    "brier_score": brier,
                    "observations": [len(x) for x in train_X],
                    "runtime_seconds": elapsed,
                }
            )
            if not any(ledger.can_afford(float(c)) for c in masks.to(bounds) @ costs):
                break
            start = perf_counter()
            acq = create_acquisition(
                "qDecoupledParticleKnowledgeGradient",
                model=model,
                decision_X=target,
                utility=utility,
                X_evaluation_mask=masks[:1],
                observation_noise=[0.05, 0.05, None],
                num_particles=16,
                num_fantasies=4,
                seed=seed + len(records),
                abstain_utility=-1.0,
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
            ledger.record(float((mask.to(costs) @ costs).sum()))
            values = problem.evaluate_truth(candidate)
            label = multiobjective_pass_labels(candidate).squeeze(-1).long()
            outputs = [values[:, :1], values[:, 1:], label]
            for i in range(3):
                if mask[0, i]:
                    train_X[i] = torch.cat((train_X[i], candidate))
                    train_Y[i] = torch.cat((train_Y[i], outputs[i]))
            elapsed += perf_counter() - start
    return records
