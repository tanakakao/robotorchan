# Phase 1 — Documentation baseline and preservation inventory

> Status: **in progress**. This is an initial, non-destructive inventory, not an approval to migrate or remove existing content.
>
> Baseline: `main` commit `9de071c899dcce5fa2c7c3c0083281d529a3b077` (2026-10-25 23:57 UTC).

## Non-loss migration policy

1. Keep all five original `docs/development/theory-implementation/*.md` files unchanged during inventory and drafting.
2. Treat **every** mapping row, exception, upstream delegation, mathematical assertion, test path, numerical tolerance, limitation, and unsupported/unverified statement as preservation-critical.
3. Build a source-section-to-destination ledger before moving material. Record original heading/line range, new destination, preservation disposition, and review status. No unreviewed content may be discarded as redundant.
4. Separate technical explanation, verification evidence, and historical PR/CI/Phase records without conflating test existence with mathematical correctness.
5. Do not remove or replace legacy files until all entries in the ledger are reconciled and inbound links are checked. Preserve compatibility links or redirects where needed.
6. Record uncertainties explicitly; never infer implementation support from a class name or a mapping assertion alone.

## Baseline inventory (direct file reads)

| File | Characters | Lines | H1–H3 headings | `Phase N` mentions | PR-number mentions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `README.md` | 16,748 | 278 | 13 | 38 | 29 |
| `models.md` | 196,924 | 1,796 | 91 | 103 | 0 |
| `acquisition.md` | 32,345 | 465 | 18 | 29 | 0 |
| `optimization.md` | 26,725 | 355 | 21 | 22 | 0 |
| `verification.md` | 29,283 | 510 | 31 | 46 | 29 |

These counts are descriptive only: mention counts are not proof of historical-only content. All numbers are from direct fetched content, including blank lines.

## Theory-side navigation observed

- `docs/theory/README.md` defines **Part I–VIII** and chapter numbers **01–24**, with chapter 13 (model selection) grouped in Part VIII rather than numeric order.
- `docs/theory/acquisition/` search inventory: `README.md`, `01_foundations.md` through `12_selection_guide.md`, plus `classification-active-learning.md`.
- `docs/theory/optimization/` search inventory: `README.md`, `01_foundations.md` through `12_selection_guide.md`.
- Chapter 24 (classification) is present; the theory README's earlier “23 chapters” prose requires a consistency check before any editing.

## Early migration risk findings

- `models.md` is the largest preservation risk (~197k characters); dividing it by class names alone risks losing cross-family caveats and source/test traceability.
- Existing `verification.md` includes **property-scoped** mathematical references, numerical tolerances and explicit exclusions. Preserve those verbatim or with traceable equivalents; do not reduce them to “tests passed”.
- Existing `README.md` embeds prior 12-phase audit scope and historical baseline; archive the historical claim rather than treating it as the current 18-phase plan.
- Mixed categorical semantics differ by model family (native covariance, internal one-hot, embeddings, passthrough, validation only). Preserve these distinctions.
- Existing docs/theory README distinguishes model latent coordinates, search coordinates, and posterior semantics across surrogate families. Preserve those conceptual boundaries in Japanese guides.

## Phase 1 outstanding gates

- [ ] Obtain authoritative full recursive repository tree at the baseline SHA; verify all theory chapter filenames and docs/models, docs/optimization, notebooks, tests, and CI paths.
- [ ] Build full heading-level inventory and section-level preservation ledger for all five source files.
- [ ] Extract all inbound relative links, anchor references and CI dependencies; assess migration breakage.
- [ ] Inspect duplicate content and missing evidence without guessing.
- [ ] Record source-to-test-to-notebook traceability and explicit verification gaps.
- [ ] Review the completed audit and decide whether Phase 1 exit criteria are satisfied.

**Exit status: NOT YET COMPLETE.** No original file has been modified or deleted.
