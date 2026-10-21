# Empirical optimization eligibility and capability gates

This is a static, conservative eligibility decision for the first measured
empirical cycle. It is not a successful runtime test or performance result.
Use the executable tier validator in
`src/robotorchan/benchmarks/comparative_matrix.py` for configuration
eligibility and the execution checklist in
`comparative-experiment-matrix.md` for runtime acceptance.

## Tier contract

| Tier | Seeds | Initial | Additional | q | dtype | Device |
| --- | ---: | ---: | ---: | --- | --- | --- |
| smoke | 1 | 4 | 4 | 1 | float64 | CPU |
| standard | 5 | 20 | 40 | 1, 3 | float64 | CPU |
| empirical_standard | 10 | 20 | 50 | 1, 3 | float64 | CPU |
| full | 20 | 30 | 100 | 1, 3, 5 | float64 | CPU |
| full_cuda | 20 | 30 | 100 | 1, 3, 5 | float64 | CUDA |

Each q is a separate comparison stratum. For empirical_standard q=3,
the last batch has two points. Seed lists must be 0 through N-1.

## Candidate method matrix

| Problem | Random | Sobol | qEI | qNEI | qEHVI | qNEHVI |
| --- | --- | --- | --- | --- | --- | --- |
| branin | baseline | integration | baseline | baseline | N/A | N/A |
| hartmann6 | baseline | integration | baseline | baseline | N/A | N/A |
| noisy_quadratic | blocked | blocked | blocked | blocked | N/A | N/A |
| branin_currin | baseline | integration | N/A | N/A | integration | integration |
| dtlz2 (3 objectives) | blocked | blocked | N/A | N/A | blocked | blocked |
| strength_pass | baseline | integration | blocked | blocked | N/A | N/A |
| strength_conductivity_pass | baseline | integration | N/A | N/A | blocked | blocked |

**baseline**: generator exists, but oracle-isolation and runtime
validation are still required. **integration**: relevant components
exist but a complete benchmark strategy has not been certified.
**blocked**: known contract gap prevents publishable comparison.
**N/A**: objective dimensionality does not match acquisition.

The validator does not list noisy_quadratic or dtlz2 among its
eligible `_METHODS`; their rows are blocked regardless of method.
It rejects classification acquisition cells via `_BLOCKED`.
It accepts some cells marked integration: passing validation does
not promote a cell to runtime-ready.

## Gates before an empirical comparison

1. **Problem**: registry lookup, dimension, objective direction,
   declared bounds and feasibility semantics are correct.
2. **Strategy**: actual callback construction, q support, optimization
   and final partial batch succeed without oracle access.
3. **Fairness**: paired initial X, budgets, seeds, RNG streams,
   observed-data-only candidate generation and fixed settings.
4. **Metrics**: truth-only regret/HV/feasibility, reference-point
   policy and metric alignment are independently verified.
5. **Reproducibility**: pinned SHA, environment, fit/acquisition/
   optimizer settings, failures and artifact paths are recorded.
6. **Statistics**: paired uncertainty intervals and multiplicity
   policy are defined before comparing results.

No cell is marked **certified** by this document. Certification
requires explicit runtime evidence attached to a subsequent phase.

## Runtime evidence and promotion

Static eligibility does not establish an oracle-safe runnable strategy.
Use the [Comparative fairness protocol](comparative-fairness-protocol.md)
for candidate-view isolation and paired execution requirements, and
the [Empirical evaluation protocol](empirical-evaluation-protocol.md)
for runtime acceptance and measured-result certification. An
integration cell must not be promoted to a validated comparison
without complete candidate-path and seed-level execution evidence.

Do not weaken blocked cells or introduce untested qEHVI adapters
just to increase matrix coverage.
