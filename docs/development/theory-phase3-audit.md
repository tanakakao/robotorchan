# Phase 3 — GP/model theory documentation audit

## Scope and acceptance criteria

This audit records the status of the GP/model theory chapters against the Phase 2
ownership contract in [theory-documentation-boundary.md](theory-documentation-boundary.md).
Phase 3 is complete only when (1) mathematics and assumptions are retained,
(2) model-specific API contracts live under `docs/models/`, (3) internal links
and CI pass on the final integrated commit, and (4) important explanations
removed during migration have been preserved or explicitly judged redundant.

## Findings from the current main snapshot (2026-10-17)

| Chapter | Observation | Disposition |
| --- | --- | --- |
| `02_gaussian_process.md`, `03_kernel.md` | No Python code fences or identified fitting API markers. | Retain; review mathematical rendering in final audit. |
| `05_mixed_variables.md`, `07_multitask_multioutput.md` | Entirely English after substantial condensation; originally Japanese chapters. | Review for lost derivations, edge cases, and Japanese-language consistency before Phase 3 sign-off. |
| `08_high_dimensional_gp.md` | Current main still contains six Python code fences and model fitting API references. | Integrate the remaining cleanup from PR #1193; check final CI. |
| `09_variational_gp.md` | No Python code fences or identified fitting API markers. | Retain; verify ELBO conventions and references. |
| `12_hierarchical_contextual_gp.md`, `17_nonstationary_gp.md`, `18_dimensionality_reduction_gp.md`, `24_classification.md` | Entirely English and substantially condensed relative to original Japanese theory material. | Audit preservation of original mathematical examples and explanations, not just links. |
| `19_neural_reduction_gp.md` | No Python code fences; some API names remain in explanatory passages. | Distinguish theoretical model names from API instructions; preserve latent uncertainty discussion. |
| `22_expressive_gp.md` | No Python code fences; a small number of API terms remain. | Preserve NNGP / DKL / DeepGP / spectral distinctions. |

## Explicit blockers

1. **Content preservation:** Large deletions in PRs #1187–#1189 and the classification
   rewrite need a source-to-destination review. A passing link checker does not
   prove mathematical completeness. Do not label the audit complete until unique
   theory and useful implementation instructions have been accounted for.
2. **Language consistency:** The six English-only chapters listed above should
   be reconciled with the Japanese-first theory documentation; translation must
   preserve equations, assumptions, and counterexamples.
3. **Branch integration:** PRs #1193–#1195 have been developed independently.
   Re-check the integrated `main` after merge rather than treating per-PR CI
   as an integration test.
4. **Documentation validation:** Run `python -m unittest discover -s tests/docs -p test_docs_links.py`,
   `python scripts/check_docs_links.py`, `python scripts/check_notebook_links.py`,
   and `python scripts/check_docs_references.py` on the final tree.

## Next actions within Phase 3 (not a new phase)

- Reconcile the high-dimensional chapter's pending edits and confirm all CI checks.
- Compare pre-rewrite content for the English-only chapters with the current
  chapters and the model documentation; restore material that has no owner.
- Recheck the final integrated tree, then issue a Phase 3 completion decision.
