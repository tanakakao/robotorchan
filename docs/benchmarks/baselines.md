# Benchmark baseline strategies

Phase 22 provides a named factory for four comparison strategies:

| Identifier | Candidate policy | Applicable problems |
| --- | --- | --- |
| `random` | Independent uniform samples | Continuous and mixed |
| `sobol` | Fresh scrambled Sobol block per callback | Continuous and mixed |
| `botorch_qei` | SingleTaskGP with BoTorch qEI | Continuous, unconstrained, single objective |
| `botorch_qnei` | SingleTaskGP with BoTorch qNEI | Continuous, unconstrained, single objective |

Use `make_baseline_strategy(name)` to obtain a callback for
`run_benchmark(config, callback)`. Use `list_baseline_strategies()`
to discover stable identifiers.

The random and Sobol baselines use the runner's seeded generator.
Sobol sampling **does not preserve a single Sobol sequence across
iterations**: each call uses a new scrambled engine. It is a
reproducible randomized quasi-Monte Carlo batch baseline.

The GP strategies use existing BoTorch-native fitting and acquisition
optimization. Their candidate callback consumes only observed X and Y.
They do not support constrained or mixed-variable optimization.

For a fair comparison, use identical problem definitions, initial
designs, seeds, evaluation budgets, batch sizes and dtype. GP fitting
and acquisition optimization add substantial wall-clock overhead;
measure runtime separately from sample efficiency.
