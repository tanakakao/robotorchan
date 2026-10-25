# Theory–Implementation Mapping: 12-phase development cycle

## Baseline and scope

- Baseline: `main` at `5ad57ed27a0d641e3928dc46b49d1dbcbe7dc5f8` (2026-10-22 UTC).
- Scope: Models, acquisition functions, acquisition optimization, and search strategies.
- This document records verified architectural boundaries and the mapping contract.
  It does **not** certify that individual algorithms are mathematically correct.
- The development cycle has exactly 12 phases; the architecture baseline above is the original Phase 1 artifact.

## Existing ownership (preserve)

| Existing source | Canonical responsibility | Mapping responsibility |
| --- | --- | --- |
| `docs/theory/` | Definitions, assumptions, derivations, references | Link to theory; do not duplicate equations |
| `docs/models/` | Public model usage and capabilities | Link to concrete model contract |
| `docs/optimization/` | Acquisition and optimizer usage | Link to dispatch and supported constraints |
| `docs/development/architecture.md` | Package and dependency boundaries | Link to implementation ownership |
| `docs/development/theory-documentation-boundary.md` | Theory/implementation separation | Enforce the separation |
| `examples/notebooks/` | Executable demonstrations | Record representative example and limits |
| `tests/` | Contracts and regression verification | Link to tests and classify evidence |
| `docs/model_coverage.json` | Generated model guide/theory/notebook coverage | Reuse; do not edit or overload |

## Confirmed source-of-truth boundaries

- `src/robotorchan/models/registry.py`: `MODEL_REGISTRY` is the model coverage source.
- `src/robotorchan/models/classification/registry.py`: classification-specific registry;
  reconcile its entries with model coverage rather than assuming identical membership.
- `src/robotorchan/acquisition/registry.py`: `ACQUISITION_REGISTRY` is acquisition
  capability metadata, not proof of optimizer feasibility.
- `src/robotorchan/acquisition/optimizer_compatibility.py`: acquisition/optimizer
  compatibility; not a substitute for inspection of optimizer implementations.
- `scripts/generate_model_coverage.py`: generates `docs/model_coverage.json`.
  Existing generation and CI checks must remain the authority for this artifact.
- `src/robotorchan/optim/`: candidate generation, constraints, and search strategies.
  Do not infer algorithm support solely from family names.

These were the initial Phase 1 architecture-level observations. Phases 2–3
subsequently established static exported-symbol inventories and concrete
source ownership in the companion mapping documents.

## Reader navigation (Phase 10)

This is the entry point for navigating the existing mapping, not a
replacement for the mathematical theory chapters. Choose a subject,
then follow its theory, implementation, and evidence in that order.

| Subject | Mapping | Canonical theory | Verification evidence |
| --- | --- | --- | --- |
| Surrogate models and classification | [Model mapping](models.md) | [Theory index](../../theory/README.md) | [Model reference checks](verification.md#phase-7-final-exit-reconciliation) |
| Acquisition and active learning | [Acquisition mapping](acquisition.md) | [Theory index](../../theory/README.md) | [Acquisition reference checks](verification.md#phase-8--verification-reconciliation-representative-scope) |
| Optimizers, candidate constraints and TuRBO | [Optimization mapping](optimization.md) | [Theory index](../../theory/README.md) | [Optimizer reference checks](verification.md#phase-9--representative-optimizer-verification-exit-audit) |

**How to interpret the evidence:** start with the mathematical
definition and assumptions in `docs/theory/`, use the relevant mapping
to locate public API and implementation ownership, then consult
`verification.md` for the *specific* property checked. Registry
coverage and smoke tests are not proofs. `Implemented`, `Delegated`,
`Extended`, `Approximated`, `Unsupported`, and `Unverified`
describe implementation relationships, not a universal accuracy grade.

The mapping files retain audit history for traceability. Reorganizing
their large sections into theory-aligned educational chapters and
removing historical phase labels is reserved for a separate workflow
after the original twelve phases; Phase 10 only improves navigation.

## Mapping record contract

A record must identify:

1. Stable method identifier and family.
2. Theory path and specific mathematical object, assumptions, and reference.
3. Public API symbol and concrete implementation module/class/function.
4. Ownership: `Implemented`, `Delegated`, `Extended`, `Approximated`,
   `Unsupported`, or `Unverified`; use multiple labels when appropriate.
5. Upstream library and exact delegated symbol **only when verified in code**.
6. Numerical method and deviations from the paper, including unknowns.
7. Test path and assertion-level evidence category: contract, numerical, or
   mathematical verification; explicitly identify unverified properties.
8. Representative notebook path, or explicit `none` with reason.
9. Limitations and audit provenance (baseline commit).

A path's existence is not mathematical verification. A representative notebook
does not imply every variant has an end-to-end demonstration. A registry entry is
not proof that all upstream implementation paths are exercised.

## Mapping documentation layout

Use `docs/development/theory-implementation/` as the canonical home for
`README.md`, `models.md`, `acquisition.md`, `optimization.md`, and
`verification.md` when those deliverables are populated in Phases 4–9.
Prefer family-level records with links to existing theory and usage documents.
Do not introduce a new wrapper, registry, or runtime API for documentation.

## Verification and remaining work

- Phase 1: inspected architecture, theory boundary, generated model coverage,
  packaging constraints, model/acquisition registries via repository code search.
- Phase 2: enumerate all public models; resolve registry/classification differences,
  implementation modules, theory and test references.
- Phase 3: enumerate public and upstream acquisition functions, optimizers,
  constraints, and strategies, including cases not represented in registries.
- Phases 7–9: independently evaluate mathematical assertions; do not upgrade
  contract-test evidence to mathematical verification.
- Phase 11: add narrowly scoped consistency checks only after mapping records
  exist; reuse existing CI jobs where possible.

### Recorded phase decisions

| Phase | Decision and evidence boundary |
| --- | --- |
| Phase 1 — architectural audit | **Complete**; original baseline above remains historical provenance |
| Phase 2 — model inventory audit | **Complete (static scope)**; registry-wide model audit and documentation path/ID contracts recorded in `models.md`; the final CI-backed exit checks were merged in PR #1361 |
| Phase 3 — acquisition / optimizer / constraint inventory | **Complete (static scope)**; `acquisition.md` and `optimization.md` record public/registry/implementation paths and representative assertions; exit checks merged in PR #1366 at `e1ae6474dca742be2f882ddac7f61c9f77306614` |
| Phase 4 — model theory-to-implementation mapping | **Complete (static mapping scope)**; all 142 regression names and 29 binary registry IDs inventoried in `models.md`, family-level theory/implementation/test boundaries mapped, and PR #1400's Python 3.11 model evidence/documentation audit and coverage check passed. Numerical equivalence remains Phase 7. |
| Phase 5 — acquisition theory-to-implementation mapping | **Complete (static mapping scope)**; regression/classification AL, scalar/multi-objective/constrained BO, information/lookahead families and 21 registry entries reconciled in `acquisition.md`; exit audit merged in PR #1408. Numerical identities remain Phase 8. |
| Phase 6 — optimizer and strategy mapping | **Complete (static mapping scope)**; all eleven named dispatch options, native mixed and one-shot paths, vector NSGA-II, constraint enforcement semantics, TuRBO, embeddings and latent strategies mapped in `optimization.md`; PR #1411 passed CI and documentation links. Numerical correctness remains Phase 9. |
| Phase 7 — model mathematical verification | **Complete (representative property scope)**; CI-backed independent references, with unsupported observation-posterior semantics explicitly retained in `verification.md` |
| Phase 8 — acquisition mathematical verification | **Complete (representative property scope)**; six merged formula/MC reference PRs #1431–#1436; explicit gaps in `verification.md` |
| Phase 9 — optimizer mathematical verification | **Complete (representative property scope)**; five merged reference/contract PRs #1438–#1442; exit audit PR #1443 in `verification.md` |
| Phase 10 — navigation and cross-document links | In progress |
| Phases 11–12 — consistency CI and final audit | Not yet complete |

**Acceptance is scope-specific:** a static source inventory or a passing
contract test does not prove an algorithm's mathematical correctness.
Independent numerical/model/acquisition/optimizer verification remains in
the previously agreed Phases 7–9. Do not add phases or change the plan to
claim such tests were part of the Phase 2/3 static exit gates.

To reproduce static audit evidence, run
`python scripts/audit_model_api.py --include-evidence-summary --check-documentation-paths`
and `python scripts/audit_acquisition_api.py`, along with repository CI.
The scripts report metadata and syntactic evidence, not independent formula
validation.

## Phase 6 exit reconciliation (static scope)

The public `robotorchan.optim.__all__` was inspected
alongside `optim/dispatch.py` and
`optim/capabilities.py`. The mapping separates:

- **Eleven named scalar backends:** one native
  BoTorch, three Torch, two sampling and five
  derivative-free/combined routes.
- **Specialist routes outside named dispatch:**
  native BoTorch mixed optimization, mixed
  augmented one-shot, and vector-valued NSGA-II.
- **Public search strategies:** original, mixed,
  random, Sobol, tree ensemble, TuRBO, REMBO,
  HeSBO, ALEBO, BAxUS, BAxUS Thompson and
  latent-space search.
- **Domain and enforcement contracts:**
  `MixedVariableSpace`, `CandidateConstraints`,
  explicit `ConstraintHandling` modes, and
  unsupported argument combinations.

This reconciles algorithm families and API
ownership, not every exported helper as a separate
algorithm. `NSGA2_OPTIMIZER_CAPABILITIES`
is a public descriptor, not a twelfth dispatcher
value. The explicit native mixed and one-shot
entry points are distinct from the convenience
dispatcher. Search strategies may delegate to
numerical backends without inheriting every
backend capability.

**Exit criterion:** Phase 6 is complete when
this reconciliation passes repository CI and
documentation-link checks. Phase 7 begins model
mathematical verification; Phase 8 acquisition
verification; Phase 9 optimizer verification.
The original twelve-phase scope is unchanged.
