# Benchmark final audit — Phase 30

## Scope and evidence

The benchmark work through Phase 29 is integrated on `main`. Phase 24–29
were merged in PRs #1062–#1067. The Phase 29 head commit
`760bce7b6c8d6d8d274c884de395df4bff91ba6c` passed its GitHub
Actions workflow. This is a **release-readiness review**, not a claim that
all possible benchmark strategies, platforms or scientific comparisons
have been independently validated.

## Verified contracts

| Area | Implemented contract | Remaining qualification |
| --- | --- | --- |
| Configuration | Explicit problem, strategy, seeds, q, budget, dtype and device | CUDA support depends on strategy/problem |
| Registry | Explicit named factory registration | No automatic registration by import |
| Runner | Initial design plus strictly budgeted new evaluations | Stateful callbacks need their own state management |
| Metrics | Truth-based regret and feasible 2D hypervolume | Known optimum/reference point required |
| Statistics | Seed-aligned curves, normal/bootstrap intervals | One-seed smoke profile is not statistical evidence |
| Reporting | JSON and CSV summaries | Curve indices include initial evaluations |
| Storage | Validated versioned JSON trajectories | Results are not model checkpoints |
| Resume | Single-seed trajectory prefix and run-local RNG state | No global RNG, model state, mid-batch or async recovery |
| Profiles | Smoke, standard and extended resource presets | Not performance baselines |
| Example | Branin random-search execution and report export | Does not exercise every benchmark family |

## Release acceptance criteria

- [x] Phase 24–29 PRs merged into `main`.
- [x] Phase 29 head GitHub Actions workflow passed.
- [x] Public end-to-end example, smoke test and usage guide present.
- [x] Explicit seed, evaluation-budget and constraint conventions documented.
- [x] Known resume and statistical limitations documented.
- [ ] Execute a fresh full test matrix on the final Phase 30 merge commit.
- [ ] Run independent scientific performance comparisons with enough seeds
      and a fixed hardware/software environment.
- [ ] Validate CUDA and optional-dependency combinations on supported hardware.

## Decision

**Conditionally ready as a benchmark framework for development and internal
evaluation.** Do not claim scientifically validated superiority of any
strategy or unrestricted deterministic resumption. A final release
designation requires the unchecked acceptance criteria appropriate to
its target platforms and published claims.

## Follow-up risks

1. The resume API is intentionally single-seed and does not checkpoint
   model or optimizer state.
2. The smoke profile is intended for integration checks, not uncertainty
   estimates or statistical significance.
3. Different hardware, precision, acquisition budgets and observation
   noise models can invalidate performance comparisons.
4. New benchmark families should be added to the execution and reporting
   regression matrix before being advertised as validated.

See [end-to-end guide](end_to_end.md) for runnable usage and linked
component documentation.
