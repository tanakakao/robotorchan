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

These problem factories create isolated, locally seeded observation
generators. Fresh factory instances reproduce the same noise stream;
the runner instantiates a new problem for each experiment seed to avoid
leaking generator state between runs. These built-in factories use a
fixed observation seed even when candidate designs vary by run seed.

Separately, the benchmark runner supports an explicit
`SeededObservationNoise` policy with run-local noise streams and
`paired` or `independent` strategy pairing. This policy is not
implicitly applied to the built-in noisy problem factories. For
measured noisy comparisons, record which observation mechanism was
used and follow the [Empirical execution contract](empirical-execution-contract.md).
Paired streams do not guarantee pointwise pairing for different
adaptive candidate paths.

Register via `register_noisy_problems(registry)`.
