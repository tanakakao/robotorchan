# Noisy benchmark problems

Two two-dimensional single-objective maximization problems separate
deterministic ground truth from noisy observations:

| Problem | Ground truth | Observation noise |
| --- | --- | --- |
| noisy_quadratic | negative squared distance from (0.35, 0.35) | Gaussian, standard deviation 0.1 |
| heteroscedastic_quadratic | same quadratic | Gaussian, standard deviation 0.02 + 0.18*x0 |

Both have a known optimum of 0 at (0.35, 0.35).

Use `evaluate_truth(X)` for ground-truth metrics and
`evaluate_observation(X)` for surrogate training. The benchmark runner
records both `Y_truth` and `Y_observed`; acquisition candidate
generators receive only observed responses.

Factories create isolated, locally seeded observation generators;
fresh factory instances reproduce the same noise stream. The runner
instantiates a new problem for each experiment seed to prevent noise
streams from leaking between runs. At present the observation stream
uses a fixed seed for every run; candidate designs still vary by run
seed. Independent per-run observation seeds are a future extension.

Register via `register_noisy_problems(registry)`.
