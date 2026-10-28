# Benchmark problem contract

`robotorchan.benchmarks.problem.BenchmarkProblem` defines an independent,
tensor-native optimization problem. The registry and generic runner are
documented in [Problem registry](../../../docs/benchmarks/problem-registry.md) and
[End-to-end benchmark](../../../docs/benchmarks/end_to_end.md), respectively.

- `bounds`: finite floating tensor `(2, d)`, lower bounds strictly below upper.
- `variable_types`: one of continuous, integer, categorical for each dimension.
  Discrete values are integer-coded; this contract does not one-hot encode categories.
  The benchmark author is responsible for valid categorical codes.
- `directions`: one maximize/minimize label per objective.
- `objective(X)`: noiseless ground truth, `(..., q, m)`.
- `observe(X)`: optional noisy observed objectives with the same shape. The
  evaluator owns random-number generation; deterministic replay requires
  explicit RNG management and a recorded observation-noise policy.
- `constraints(X)`: optional deterministic outcome residuals `(..., q, c)`
  where **g(X) >= 0 means feasible**. Candidate/search-space constraints are
  a separate optimizer-layer concept.
- `cost(X)`: optional nonnegative `(..., q, 1)`; defaults to unit cost.
- `optimal_value`: optional known scalar objective optimum, represented as a
  length-one tensor in the original direction. No exact-optimum claim is made
  when absent.
- `reference_front` / `reference_point`: optional multiobjective references
  in original objective directions. A grid-derived front is approximate, not
  an analytical optimum.

`simple_regret(X)` uses **truth**, never noisy observed responses. For a
constrained problem, only feasible evaluated candidates contribute. It returns
positive infinity when no feasible candidate has been evaluated. For unknown
optima or multiobjective problems, the caller must select a different metric.
A known optimum must correspond to the feasible optimum when constraints exist.

This is intentionally a small problem-layer contract. It does not itself
implement a GP, acquisition function, optimizer, registry, persistence format,
or high-level optimization API. Those responsibilities belong to other
benchmark and optimization modules.
