# Documentation structure contract

This document defines the maintained documentation ownership and validation scope. The current `main` branch is the source of truth.

## Ownership

| Location | Responsibility |
|---|---|
| `docs/getting_started/` | Installation, quickstart, initial model selection |
| `docs/models/` and `docs/models.md` | Public model usage, family guidance, API constraints |
| `docs/optimization/` | Acquisition, candidate generation, search strategies |
| `docs/theory/` | Mathematical foundations, assumptions, derivations |
| `docs/benchmarks/` | Experiment protocols, metrics, evidence, reproducibility |
| `docs/guide/` | Theory-to-source mappings, class structure, internal flow and design decisions |
| `docs/development/` | Japanese developer learning and release guides |
| `docs-internal/` | Normative architecture, API and numerical contracts, maintenance |
| `.github/project-records/` | Historical designs and audit evidence |
| `examples/notebooks/` | Executable usage examples |

The navigation entry point is `docs/README.md`. Each documentation family
maintains its own `README.md`. Model coverage is tracked separately in
`docs/model_coverage.json`; coverage is not inferred from file counts.

## Validation gates

The `Documentation links` workflow runs the regression tests for its checkers and
the following validation commands:

1. `python scripts/check_docs_links.py`
2. `python scripts/check_notebook_links.py`
3. `python scripts/check_docs_references.py`
4. `python scripts/check_theory_implementation_mapping.py`
5. `python scripts/check_theory_guide_quality.py`
6. `python scripts/check_documentation_inventory.py`
7. `python scripts/check_documentation_boundaries.py`

These checks validate supported local Markdown targets, notebook Markdown-cell
targets, reference-style links, Markdown heading fragments, theory-to-implementation
references, the fixed 48-chapter guide layout, inventory coverage, and ownership
boundaries. The workflow
runs on pull requests and pushes to `main` using Python 3.11.

The link checkers scan root Markdown, `docs/`, `docs-internal/`, `.github/`,
`examples/`, `benchmarks/`, and Markdown cells in
`examples/notebooks/*.ipynb` (the current flat notebook layout). They do not
execute notebooks or establish semantic correctness. Notebook execution is a
separate CI responsibility described in [Notebook CI](notebook-ci.md).

Model coverage is generated and checked separately by
`python scripts/generate_model_coverage.py --check` in the main CI workflow;
it is not part of the Documentation links workflow. Notebook JSON schemas are
checked separately by `python scripts/check_notebook_schema.py` in
`.github/workflows/notebook-schema.yml`.

## Maintenance decisions

- **Keep** an entry point for each public, engineering, and record directory.
- **Keep** model, theory, optimization, benchmark, and developer content in their
  respective ownership directories.
- **Keep** executable notebooks under `examples/notebooks/`; do not relocate
  them into `docs/`.
- **Do not delete or merge** files solely because their names or topics overlap.
  Such changes require a content-level comparison and preservation of API,
  mathematical, and experimental contracts.
- **Do not classify** benchmark reports as empirical proof unless actual runs,
  configurations, seeds, and artifacts support the claim.

## Coverage limitations

Uncovered Markdown variants (including HTML anchors), non-Markdown notebook
cells, and notebooks in future nested directories remain known limitations
rather than implied validation coverage. Local link and fragment checks do not
validate external URLs, mathematical claims, API compatibility, or actual
benchmark performance.

## Inventory

[documentation-inventory.json](documentation-inventory.json) records every baseline Markdown file, its responsibility, route and content hash. Generated documents are classified separately. `python scripts/check_documentation_inventory.py` rejects missing destinations, duplicate baseline rows and unclassified current Markdown. It does not certify language quality, mathematical correctness, or semantic preservation.

## Language and ownership boundary

Human-facing learning and usage prose under `docs/` is Japanese. Preserve Python
identifiers, code and mathematical notation. Detailed engineering requirements
may be English under `docs-internal/`. Theory explains mathematical concepts;
Guide explains concrete classes, internal computations, delegation and design decisions.
Usage examples belong in Notebooks or API references, and executable checks belong in Tests.
Historical phase decisions belong in project records or the inventory notes.

`python scripts/check_documentation_boundaries.py` checks concrete local class
names and package instructions in Theory, Guide source paths, and phase records.
Its regression tests run in the existing Documentation links job. Structural
checks do not establish mathematical correctness or replace content review.

## Guide content contract

Guide is technical explanation of how Theory is implemented in source code.
It is not a usage tutorial, an example collection, or a verification report.
The responsibility categories are fixed:

- Theory: mathematics, derivations, assumptions, algorithms.
- Guide: theory-to-implementation mappings, class structure, internal flow, design decisions.
- Notebooks: usage, executable examples, experiments, visualization.
- API Reference: public APIs, arguments, return values, usage contracts.
- Tests: correctness, regressions, boundary conditions.
- Development: development records, audit history, phase management, CI information.

Preserve useful implementation explanations and link to existing material instead of duplicating it.
Check the actual source for classes, methods, file paths, and mathematical computations.
Distinguish BoTorch / GPyTorch delegation from robotorchan implementations.
Document approximations and unsupported combinations where they affect the computation.

Guide code is optional. Only real source excerpts or computations needed to explain implementation
belong there. Data generation, model instantiation, fit/predict calls, plotting, assertions for
verification, and CI/audit outcomes belong in their existing owners.
When removing a usage snippet consumed by a test, preserve its numerical checks in Tests without
requiring Guide to retain the tutorial. Do not create another example unless it fills a real gap.

A chapter is complete when a reader can identify the class, method, and internal operation that
realize the corresponding Theory formula or algorithm. Link checks alone do not establish this.
