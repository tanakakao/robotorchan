# Comparative optimization benchmark — Phase 1 capability audit

Audit date: 2026-10-09. Source: repository default branch `main` (inspect commit SHA again before Phase 2).

## Scope and decision

Reuse the existing problem registry, runner, profiles, trajectory storage, paired statistics and reporting. Do not introduce a second runner. This is a static source audit, not evidence of successful execution or comparative performance.

| Capability | Evidence | Decision |
| --- | --- | --- |
| Standard single-objective | `benchmarks/standard_problems.py` (branin, hartmann6) | Reuse |
| Multi-objective | `benchmarks/multiobjective_problems.py` (branin_currin, dtlz2) | Reuse |
| GP qEI/qNEI | `benchmarks/botorch_strategy.py` | Reuse with configuration audit |
| Heterogeneous constrained cases | `benchmarks/heterogeneous_problems.py`, `docs/benchmarks/multiobjective_heterogeneous.md` | Reuse problem definitions |
| Acquisition composition | `acquisition/composition.py` | Verify semantics and optimizer integration |
| Ground-truth metrics | `docs/benchmarks/metrics.md` | Reuse 2D metrics; address 3D |
| Runner, storage and paired reporting | `docs/benchmarks/end_to_end.md` | Reuse |
| CI | `.github/workflows/ci.yml` | Existing Python 3.11/3.12/3.13 test matrix; fresh run not verified |

## Gaps and risks for later phases

1. The native `make_botorch_gp_strategy` supports unconstrained continuous single-objective qEI/qNEI only. Confirm qEHVI/qNEHVI strategy callbacks and compare their settings with BoTorch before claiming full benchmark support.
2. `hypervolume_curve` is documented as two-objective only, whereas `dtlz2` has three objectives. Implement or reuse tested multi-dimensional hypervolume calculation before DTLZ2 evaluation.
3. `branin_currin` has a reference point but no stored reference front. Define a reproducible reference HV before reporting HV regret.
4. `docs/optimization/heterogeneous-e2e-constrained-multiobjective.md` describes marginal PoF weighting and a qNEHVI path without baseline feasibility correction. Do not label it exact joint-MC constrained hypervolume improvement.
5. `docs/optimization/heterogeneous-e2e-classification-constrained.md` documents marginal PoF weighting and no qNEI baseline feasibility correction. Separate approximation comparisons from exact constrained acquisition comparisons.
6. `docs/benchmarks/end_to_end.md` documents that resume does not capture global RNG, model/optimizer state or stochastic observation noise. Specify deterministic noise and restart protocol.
7. Make acquisition MC sample counts, optimizer restart/raw sample counts, fit time, construction time, optimizer time, and seed handling explicit in Phase 2.
8. Audit named random/Sobol/BoTorch baseline strategy integration and open benchmark PRs before modifying the same files.

## Proposed Phase 2 entry criteria

- Pin the exact `main` commit and environment.
- Define paired initial designs, truth-only metrics, and a common evaluation budget.
- Define exact versus approximate constrained methods and permissible comparisons.
- Distinguish static audit from tests executed in CI or locally.
- Resolve or explicitly exclude unsupported experiment matrix cells.

## Verification status

Static inspection: performed. New benchmark executions: not performed. Fresh CI run: not performed. No optimization superiority claim is supported by this audit.
