# Scientific validity audit — Phase 5: empirical reliability evidence inventory

Status: **Phase 5 initiated; evidence triage only** (2026-11-21).

Phase 4 validated selected optimizer and candidate-constraint contracts. Phase 5 asks a different question: do closed-loop Bayesian optimization and active-learning methods show reproducible, statistically interpretable behavior under fair comparisons? Existing measured benchmark evidence and infrastructure should be reused before scheduling any new runs.

## Existing measured evidence (not new measurements)

The archived [four-cell evidence](../verification/benchmarks/phase3-four-cell-evidence.md) covers Branin and Hartmann6, each with q=1 and q=3, comparing Random, Sobol, qEI, and qNEI. Each cell reports ten seeds (0–9), twenty initial observations, fifty additional evaluations, CPU float64, and saved report artifacts with original experiment SHAs.

The measured mean final simple regrets (lower is better) are:

| Problem | q | Random | Sobol | qEI | qNEI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Branin | 1 | 0.755294 | 0.949458 | 0.002399 | 0.002155 |
| Branin | 3 | 0.755294 | 0.949458 | 0.001094 | 0.001751 |
| Hartmann6 | 1 | 1.582809 | 1.390789 | 0.257661 | 0.163215 |
| Hartmann6 | 3 | 1.582809 | 1.390789 | 0.484626 | 0.294021 |

The original evidence reports paired-bootstrap improvements over Random/Sobol *within* each problem/q cell. These observations do not establish superiority for all problems, or a q=3 advantage over q=1. The exploratory q comparisons have intervals spanning zero. Branin q=1 has a different original experiment SHA from the other three cells. Same seed indices do not prove identical initial designs.

## Reuse before implementation

| Existing asset | Phase 5 role |
| --- | --- |
| `docs-internal/verification/benchmarks/phase3-four-cell-evidence.md` | Measured four-cell results, source SHAs, original runs, artifact links, exploratory caveats |
| `docs-internal/contracts/benchmarks/empirical-execution-contract.md` | Fairness, RNG isolation, oracle isolation, paired initial conditions, budget and provenance requirements |
| `docs/benchmarks/multiseed_statistics.md` | Seed-level bootstrap and paired comparisons; avoids pseudoreplication of trajectory points |
| `docs/benchmarks/quickstart.md` | Reproducing saved reports and separating smoke tests from empirical certification |
| `docs/benchmarks/empirical-evaluation-protocol.md` | Frozen experiment and reporting protocol |
| `src/robotorchan/benchmarks/` | Existing runner, strategies, baseline comparisons, metrics, reporting and persistence |

## Evidence gaps and priority

1. **Cross-q comparability — highest priority:** verify paired initial observations from original saved trajectories and review code differences between Branin q=1 SHA and other cells. Until verified, cross-q comparisons remain exploratory.
2. **Durable provenance:** ensure reports, manifests, raw trajectories, environment and source SHAs are retained outside expiring GitHub Actions artifacts. Do not assert archived availability without fetching them.
3. **Active learning:** existing four-cell measurements are BO only; identify an existing AL problem, random baseline, meaningful metric (e.g. held-out log loss or classification accuracy), and a leakage-free evaluation protocol before new multi-seed runs.
4. **Constrained and Mixed:** prioritize a single representative use case with a common evaluation budget, constraint violation metric, failure accounting and baseline rather than a full algorithm Cartesian product.
5. **Statistical interpretation:** use seed as the resampling unit, report per-seed outcomes and uncertainty, and treat failed seeds as outcomes rather than silently dropping them.

## Phase 5 execution gates

- **Gate A: provenance.** Identify exact artifacts, commit SHA, environment and settings; distinguish previously measured evidence from newly executed runs.
- **Gate B: fairness.** Apply the [empirical execution contract](../contracts/benchmarks/empirical-execution-contract.md), including `restrict_candidate_problem=True`, `independent_rng_streams=True`, `isolate_strategy_rng=True`, paired initial observations, and explicit noise policy where relevant.
- **Gate C: reproducibility.** Preserve raw trajectories and manifests; verify initial pairing, per-seed budget, metric direction, q schedule, and any failure.
- **Gate D: inference.** Compare paired seeds with intervals and describe the tested problem/budget only; avoid global superiority claims or treating bootstrap probability as a Bayesian posterior probability.
- **Gate E: scope.** Only introduce a new experiment or test after identifying a specific evidence gap not covered by existing assets.

## Next work item

First, audit cross-q initial-observation pairing and experiment-SHA comparability from the four archived trajectory sets. If the original artifacts are unavailable, record the evidence limitation and plan a *new* properly paired experiment; do not reconstruct or fabricate trajectories from summary statistics.

This inventory makes **no claim of new benchmark execution**, no claim of independently verified archived artifacts, and no Phase 5 completion decision.
