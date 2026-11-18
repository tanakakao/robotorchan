"""Compare learned contextual Study and noncontextual Study on matched budgets.

Run: python benchmarks/contextual_study.py
The context-blind baseline optimizes a single decision using observations from
varying contexts; it cannot condition its next proposal on the observed context.
"""

from __future__ import annotations

import time

import torch

from robotorchan.study import create_study


def objective(x: float, context: float) -> float:
    return 1.0 - (x - context) ** 2


def run(seed: int = 7, n_trials: int = 8) -> dict[str, float]:
    """Fit two independent Studies with identical observation context sequences.

    Each Study spends n_trials evaluations. A separate final recommendation is
    generated but not evaluated against the objective during training.
    """
    if n_trials < 3:
        raise ValueError("n_trials must be at least 3.")
    contexts = torch.linspace(0.1, 0.9, n_trials, dtype=torch.double)
    contextual = create_study(
        search_space={"x": (0.0, 1.0), "context": (0.0, 1.0)},
        context_names=("context",),
        seed=seed,
        n_initial_trials=3,
        optimizer_kwargs={"num_restarts": 2, "raw_samples": 16},
    )
    blind = create_study(
        search_space={"x": (0.0, 1.0)},
        seed=seed,
        n_initial_trials=3,
        optimizer_kwargs={"num_restarts": 2, "raw_samples": 16},
    )
    started = time.perf_counter()
    for context in contexts.tolist():
        trial = contextual.ask(context={"context": context})
        contextual.tell(trial, objective(trial.params["x"], context))
        trial = blind.ask()
        blind.tell(trial, objective(trial.params["x"], context))
    training_seconds = time.perf_counter() - started

    # Compare *observed* outcomes only. The blind Study's best observed value
    # is not a context-distribution policy value; report mean training rewards
    # under the same context schedule instead.
    contextual_rewards = [trial.value for trial in contextual.trials]
    blind_rewards = [trial.value for trial in blind.trials]
    return {
        "contextual_mean_observed": sum(contextual_rewards) / n_trials,
        "blind_mean_observed": sum(blind_rewards) / n_trials,
        "contextual_best_observed": max(contextual_rewards),
        "blind_best_observed": max(blind_rewards),
        "training_seconds": training_seconds,
    }


if __name__ == "__main__":
    for trial_seed in (7, 19, 31):
        print(f"seed={trial_seed}")
        for name, value in run(seed=trial_seed).items():
            print(f"  {name}: {value:.6f}")
