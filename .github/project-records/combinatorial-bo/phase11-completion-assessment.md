# Combinatorial Bayesian Optimization — Phase 11 completion assessment

**Assessment:** Core implementation and documentation complete; full benchmark comparison **not yet complete**.

This record distinguishes implemented APIs from planned research scope. It is not a claim of
original-paper reproduction or statistical superiority.

## Implementation matrix

| Area | Implemented | Input / batch scope | Limitations |
| --- | --- | --- | --- |
| Discrete domain | `MixedVariableSpace`, `hamming_distance`, `hamming_neighbors` | Discrete finite coordinates | Constraints supported where accepted by neighbor generation |
| Hamming search | `HammingLocalSearchStrategy` | Fully discrete, q=1 | Local optimum possible; starting points affect cost |
| Discrete TR | `DiscreteTrustRegionState`, `update_discrete_trust_region`, `DiscreteTrustRegionStrategy` | Fully discrete, q=1 | Caller manages the outer sequential BO loop |
| BOCS surrogate | `BOCSSurrogate`, `bocs_quadratic_features` | Binary | Gaussian-prior quadratic approximation; not original sparse-prior BOCS |
| BOCS BO | `propose_bocs_candidate`, `run_bocs_sequential` | Binary, single objective, q=1 | Enumerates unseen candidates; no general mixed/constraint interface |
| COMBO surrogate | `BinaryGraphDiffusionKernel`, `COMBOSurrogate` | Binary | Graph-diffusion GP; not the entire original COMBO pipeline |
| COMBO BO | `propose_combo_candidate`, `run_combo_sequential` | Binary, single objective, q=1 | Enumerated UCB; no general mixed/constraint interface |
| Benchmark | `examples/combinatorial_bo_benchmark.py` | Small binary, q=1 | Random/Sobol/numeric-GP/Hamming/TR/BOCS/COMBO; missing Mixed GA and actual MixedSingleTaskGP |
| Theory / Guide | `docs/theory/27_combinatorial_bayesian_optimization.md`, `docs/guide/27_combinatorial_bayesian_optimization.md` | Explanations and API mapping | Distinguish research algorithms from implemented approximations |
| Notebook | `examples/notebooks/49_combinatorial_bayesian_optimization.ipynb` | Small binary example | Manual CI execution classification |

## Verification and scope

- The merged Phase 10 head `af7bf1815d424d4a1dc3fc2112fabccf0dd71c5e` passed CI,
  documentation links, and notebook schema validation.
- Notebook schema validation is **not** proof of notebook execution.
- Phase 9 benchmark documentation explicitly records missing Mixed GA and
  `MixedSingleTaskGP` baselines. The numeric GP baseline is **not** the latter.
- No benchmark results or performance ranking are asserted here. Model fitting and
  search wall time should be compared separately from evaluation budget where useful.
- Existing tests cover the introduced feature families; this phase adds no unrelated
  regression-test matrix.

## Completion decision

**Conditionally complete for the binary/q=1 core extension. Not complete for the
original full Phase 9 comparison matrix.**

Before declaring all phases fully closed, either:
1. add the missing Mixed GA and genuine `MixedSingleTaskGP` baselines and run the
   benchmark under a reproducible budget; or
2. explicitly approve a narrowed Phase 9 scope and record that change.

Mixed input for BOCS/COMBO, general constraints, q>1, multiobjective and
large-scale candidate search remain **future extensions**, not silent completion
requirements for this binary core.

## Reference material

- [Japanese theory](../../../docs/theory/27_combinatorial_bayesian_optimization.md)
- [Implementation guide](../../../docs/guide/27_combinatorial_bayesian_optimization.md)
- [Executable notebook](../../../examples/notebooks/49_combinatorial_bayesian_optimization.ipynb)
- [Benchmark protocol and caveats](../../../examples/combinatorial_bo_benchmark.md)
