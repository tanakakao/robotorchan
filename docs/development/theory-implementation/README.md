# Theory–Implementation Mapping: Phase 1 architecture audit

## Baseline and scope

- Baseline: `main` at `5ad57ed27a0d641e3928dc46b49d1dbcbe7dc5f8` (2026-10-22 UTC).
- Scope: Models, acquisition functions, acquisition optimization, and search strategies.
- This document records verified architectural boundaries and the mapping contract.
  It does **not** certify that individual algorithms are mathematically correct.
- The development cycle has exactly 12 phases; this is the Phase 1 artifact.

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

These are architecture-level observations, not completed per-symbol audits. Phases 2–3
will establish exhaustive exported-symbol inventories and their exact source paths.

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

No runtime tests, full-tree inventory, or CI execution are claimed by this
document. Phase 1 remains open until PR CI passes, the PR is merged, and
the remaining architecture/inventory review is signed off.
