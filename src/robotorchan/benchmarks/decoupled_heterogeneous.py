"""Equal-budget regression and categorical outcome optimization."""

from time import perf_counter
from typing import Literal

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.acquisition import create_acquisition
from robotorchan.models.classification import (
    BinarySingleTaskGPClassifier,
    MulticlassSingleTaskGPClassifier,
    OrdinalSingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.optim import optimize_acqf_decoupled
from robotorchan.workflow.cost_budget import CostBudget


def run_heterogeneous_decoupled_benchmark(
    *,
    kind: Literal["binary", "multiclass", "ordinal"] = "binary",
    seed: int = 0,
    budget: int = 40,
    n_initial: int = 6,
    num_samples: int = 8,
    epochs: int = 30,
    decoupled: bool = True,
    fit_regression: bool = True,
) -> list[dict]:
    """Optimize a synthetic response subject to a categorical quality outcome."""
    classes = {
        "binary": BinarySingleTaskGPClassifier,
        "multiclass": MulticlassSingleTaskGPClassifier,
        "ordinal": OrdinalSingleTaskGPClassifier,
    }
    if (
        kind not in classes
        or n_initial < 3
        or budget < 4 * n_initial
        or min(num_samples, epochs) < 1
    ):
        raise ValueError("Invalid heterogeneous benchmark configuration.")
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    engine = torch.quasirandom.SobolEngine(1, scramble=True, seed=seed)
    X = torch.linspace(0.05, 0.95, n_initial, dtype=bounds.dtype).unsqueeze(-1)
    target = engine.draw(32).to(bounds)
    target = torch.cat((target, bounds))

    def evaluate(x):
        response = 1 - (x - 0.8).square()
        if kind == "binary":
            labels = (x[:, 0] <= 0.6).long()
        elif kind == "multiclass":
            labels = torch.where(x[:, 0] < 0.3, 0, torch.where(x[:, 0] <= 0.6, 2, 1))
        else:
            labels = (x[:, 0] <= 0.6).long() + (x[:, 0] <= 0.3).long()
        return response, labels

    Y, labels = evaluate(X)
    train_X = [X.clone(), X.clone()]
    costs = bounds.new_tensor([3, 1])
    full = torch.ones((1, 2), dtype=torch.bool)
    masks = torch.cat((torch.eye(2, dtype=torch.bool), full)) if decoupled else full
    ledger = CostBudget(budget, spent=4 * n_initial)
    records = []
    elapsed = 0.0

    def utility(samples):
        response, probabilities = samples
        return probabilities[..., -1] * (response.squeeze(-1) + 1) - 1

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        while True:
            start = perf_counter()
            regression = SingleTaskGP(train_X[0], Y)
            if fit_regression:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(regression.likelihood, regression))
            kwargs = {} if kind == "binary" else {"num_classes": 3}
            classifier = classes[kind](
                train_X[1], labels, inducing_points=min(6, len(labels)), **kwargs
            )
            classifier.fit(epochs=epochs, lr=0.01)
            model = HeterogeneousModel(regression, classifier)
            model.eval()
            with torch.no_grad():
                response = regression.posterior(target).mean
                probabilities = classifier.probability_posterior(target).mean
                expected = utility((response, probabilities))
                index = int(expected.argmax())
                truth, actual = evaluate(target[index : index + 1])
                feasible = bool(actual.item() == (1 if kind == "binary" else 2))
                payoff = float(truth.item()) if feasible else -1.0
                _, target_labels = evaluate(target)
                one_hot = torch.nn.functional.one_hot(
                    target_labels, num_classes=probabilities.shape[-1]
                ).to(probabilities)
                brier = float((probabilities - one_hot).square().sum(dim=-1).mean())
            elapsed += perf_counter() - start
            records.append(
                {
                    "seed": seed,
                    "kind": kind,
                    "method": "decoupled_particle_kg" if decoupled else "coupled_particle_kg",
                    "cost": ledger.spent,
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
                observation_noise=[0.05, None],
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
            response, label = evaluate(candidate)
            for i in range(2):
                if mask[0, i]:
                    train_X[i] = torch.cat((train_X[i], candidate))
            if mask[0, 0]:
                Y = torch.cat((Y, response))
            if mask[0, 1]:
                labels = torch.cat((labels, label))
            elapsed += perf_counter() - start
    return records
