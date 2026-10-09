# Candidate-space constraints and trust-region benchmarks

This benchmark separates **candidate-space constraints** (valid design
points, enforced before evaluation) from **outcome constraints** (observed
feasibility, represented by `BenchmarkProblem.constraints`).

Two benchmark objectives are available:

- `candidate_linear_region`: maximize a quadratic with candidate rule
  `1 - x0 - x1 >= 0`.
- `candidate_nonlinear_region`: maximize the same quadratic with
  candidate rule `0.16 - (x0-0.5)^2 - (x1-0.5)^2 >= 0`.

Both rules use the convention `g(X) >= 0`, with the rule retrieved using
`candidate_constraint(problem_name)`. They are not outcome constraints:
`evaluate_constraints` returns an empty last dimension.

`TrustRegionCandidateGenerator(center, length, constraint)` implements
a **fixed** box trust-region random baseline with bounded rejection
sampling. The region is clipped to global bounds. It raises an explicit
error if no feasible candidate can be found within `max_draws`.

This is a baseline for candidate-constraint and local-search integration,
**not** the adaptive TuRBO algorithm: there is no success/failure state,
trust-region expansion, shrinkage or restart. Future comparisons should
use the library's real TuRBO strategy alongside this baseline. The
initial Sobol design is not filtered by candidate-space constraints;
the generator guarantees feasibility only for newly proposed points.
