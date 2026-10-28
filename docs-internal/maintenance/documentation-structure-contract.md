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
| `docs/guide/` | Japanese theory-to-implementation learning chapters |
| `docs/development/` | Japanese developer learning and release guides |
| `docs-internal/` | Normative architecture, API and numerical contracts, maintenance |
| `.github/project-records/` | Historical designs and audit evidence |
| `examples/notebooks/` | Executable usage examples |

The navigation entry point is `docs/README.md`. Each documentation family
maintains its own `README.md`. Model coverage is tracked separately in
`docs/model_coverage.json`; coverage is not inferred from file counts.

## Validation gates

The `Documentation links` workflow runs:

1. `python -m unittest discover -s tests/docs -p test_docs_links.py`
2. `python scripts/check_docs_links.py`
3. `python scripts/check_notebook_links.py`
4. `python scripts/check_docs_references.py`

These checks validate supported local Markdown targets, notebook Markdown-cell
targets, reference-style links, and Markdown heading fragments. The workflow
runs on pull requests and pushes to `main` using Python 3.11.

The link checkers scan root Markdown, `docs/`, `docs-internal/`, `.github/`,
`examples/`, `benchmarks/`, and Markdown cells in
`examples/notebooks/*.ipynb` (the current flat notebook layout). They do not
execute notebooks or establish semantic correctness. Notebook execution is a
separate CI responsibility described in [Notebook CI](notebook-ci.md).

Model coverage is generated and checked separately by
`python scripts/generate_model_coverage.py --check` in the main CI workflow;
it is not one of the four Documentation links checks.

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
concrete classes, runtime support, code references and examples belong in Guide.
Historical phase decisions belong in project records or the inventory notes.

`python scripts/check_documentation_boundaries.py` checks concrete local class
names and package instructions in Theory, Guide source paths, and phase records.
Its regression tests run in the existing Documentation links job. Structural
checks do not establish mathematical correctness or replace content review.
