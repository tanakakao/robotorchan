# Phase 20: Heterogeneous E2E final audit

## Status

**Audit outcome: partial coverage, not release certification.** This phase adds a small cross-phase contract test and a traceable coverage / gap inventory. Passing the tests does not certify all optimizer, classification, or multiobjective workflows.

The audit source is the repository `main` at the time the Phase 20 branch was created. The Phase 19 pull request #1028 passed CI at commit `c605854278cbce6ba8c7e35e0c2b69c125e93a91`. Phase 20 CI and the full-suite result must be verified separately before merging.

## Covered by the cross-phase contract test

- The 3D unit-cube synthetic truth produces two regression targets and a Pass probability in [0, 1].
- With zero observation noise, observed regression targets match the truth; binary labels remain integral.
- A feasible Pareto reference front has consistent input/objective shapes, finite objectives, and satisfies the oracle feasibility threshold.
- Across-seed summary means, unbiased sample standard deviations, and standard errors match PyTorch calculations.
- Notebook seed list, initial-point count, and evaluation budget match the written benchmark report.

## Earlier phase coverage inventory

| Area | Existing validation | Limitation |
| --- | --- | --- |
| Regression + classification model integration | Phase 1-6 synthetic and semantic tests | Not every model family is included |
| qEI / qNEI and classification constraints | Phase 7-8 | Numerical checks are synthetic |
| qEHVI / qNEHVI and mixed constraints | Phase 9-12 | Benchmark quality not established |
| Sequential optimization | Phase 13 | Small evaluation budget |
| Random / Sobol / qEI | Phase 14-15 | Strength-only baseline; no runtime normalization |
| Exact and variational regression | Phase 16 | No model-quality ranking |
| Float32/64, q-batch, fixed features, optional CUDA | Phase 17 | Limited optimizer backends |
| Negative inputs and boundary conditions | Phase 18 | No out-of-memory or optimizer failure injection |
| Notebook and benchmark reporting | Phase 19 | Notebook execution not required by its static test |
| Cross-phase contract checks | Phase 20 | Contract smoke tests, not comprehensive system certification |

## Unresolved gaps

1. **Pending and asynchronous optimization:** no complete heterogeneous E2E run validating pending-point bookkeeping, fantasies, and candidate de-duplication.
2. **Nonlinear candidate constraints:** no full closed-loop comparison including nonlinear search-space constraints and classification outcome constraints together.
3. **Classification-constrained multiobjective benchmark:** individual composition paths exist, but the Phase 19 reproducibility notebook benchmarks only single-objective Strength; feasible hypervolume and PoF calibration are not compared across seeds.
4. **Optimizer backend matrix:** no exhaustive numerical/constraint compatibility test across BoTorch, gradient-based, evolutionary, and mixed-variable optimizers.
5. **Notebook execution:** Phase 19 tests check JSON structure and Python syntax; execute the notebook in a clean environment before claiming end-user reproducibility.
6. **Performance and significance:** three seeds and three additional evaluations are illustrative. No statistically defensible algorithm ranking, wall-time comparison, or convergence guarantee is claimed.

## Recommended verification before merge

```bash
ruff check .
ruff format --check .
pytest -q tests/benchmarks/test_heterogeneous_final_audit_e2e.py
pytest -q tests/benchmarks
pytest -q
```

Record each command's outcome and the exact commit. Notebook execution requires Jupyter and Matplotlib and should be performed separately in an environment with the documented dependencies.

## Completion criterion

Phase 20 is complete as an **audit deliverable** when its CI is green and the gaps above are accepted as explicitly out of scope or tracked for a subsequent cycle. It must not be described as comprehensive heterogeneous BO release readiness until the unresolved end-to-end workflows have been validated.
