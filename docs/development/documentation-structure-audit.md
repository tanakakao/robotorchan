# Documentation structure audit — Phase 5

This audit records the final documentation ownership and validation scope after the
five-phase documentation cleanup. The current `main` branch is the source of truth.

## Ownership

| Location | Responsibility |
|---|---|
| `docs/getting_started/` | Installation, quickstart, initial model selection |
| `docs/models/` and `docs/models.md` | Public model usage, family guidance, API constraints |
| `docs/optimization/` | Acquisition, candidate generation, search strategies |
| `docs/theory/` | Mathematical foundations, assumptions, derivations |
| `docs/benchmarks/` | Experiment protocols, metrics, evidence, reproducibility |
| `docs/development/` | Architecture, maintained contracts, CI, release |
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
targets, reference-style links, and Markdown heading fragments. They do not
establish that documentation is semantically correct or that notebooks execute.

## Cleanup decisions

- **Keep** the seven entry points: the documentation root and six family READMEs.
- **Keep** model, theory, optimization, benchmark, and developer content in their
  respective ownership directories.
- **Keep** executable notebooks under `examples/notebooks/`; do not relocate
  them into `docs/`.
- **Do not delete or merge** files solely because their names or topics overlap.
  Such changes require a content-level comparison and preservation of API,
  mathematical, and experimental contracts.
- **Do not classify** benchmark reports as empirical proof unless actual runs,
  configurations, seeds, and artifacts support the claim.

## Final acceptance

Phase 5 is complete when the final audit PR passes normal CI and the
Documentation links workflow, and the final ownership/navigation decisions
are reviewed and merged. Uncovered Markdown variants (including HTML anchors
and notebook fragments) remain known limitations rather than implied coverage.
