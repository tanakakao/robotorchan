# Scientific validity audit — Phase 7: defect remediation and regression-evidence reconciliation

Status: **scoped remediation review complete; pending PR merge** (2026-11-22).

This phase reconciles **concrete defects already found and fixed** during the
scientific-validity audit. It does not start another general-purpose audit,
repeat Phase 2–6 experiments, or equate a green CI with a universal proof.

## Concrete defects and dispositions

| Finding | Fix | Evidence and disposition |
| --- | --- | --- |
| Multiclass classifier exposed an inherited MLL factory with misleading latent ownership | [PR #1954](https://github.com/tanakakao/robotorchan/pull/1954) | Merged; unsupported inherited `make_mll()` now rejects explicitly; PR head CI and documentation checks succeeded |
| Joint heteroskedastic Gaussian log-likelihood applied variance flooring inconsistently | [PR #1955](https://github.com/tanakakao/robotorchan/pull/1955) | Merged; Gaussian normalization and quadratic term aligned; independent seeded numerical reference and regression coverage; PR head CI and documentation checks succeeded |
| Joint variational GP relied on private cache-clearing internals | [PR #1956](https://github.com/tanakakao/robotorchan/pull/1956) | Merged; private cache-clear calls removed and state restoration checked; PR head CI and documentation checks succeeded |

The first two findings were **correctness/contract defects**; the third was
a **maintenance and state-lifecycle risk**. All three are resolved in the
merged changes and covered by targeted regression evidence. These statements
are scoped to the audited paths and do not establish correctness for every
model configuration.

## Other audited areas: verification, not new defect claims

- [Phase 2](scientific-validity-phase2.md): targeted exact/mixed/multifidelity
  posterior references; multiclass posterior covariance and learning-signal
  checks in merged PRs #1957–#1958; ordinal probability tests remain limited.
- [Phase 3](scientific-validity-phase3.md): targeted acquisition references
  including merged PRs #1960–#1963; marginal-versus-joint q semantics and
  observation-model assumptions explicitly documented.
- [Phase 4](scientific-validity-phase4.md): targeted constrained sampling,
  Mixed/q=2 and infeasible joint-constraint behavior in merged PRs
  #1965–#1969; no claim of global optimizer convergence.
- [Phase 5](scientific-validity-phase5-gp-al-epoch-sensitivity-results.md):
  measured GP AL fitting-budget sensitivity; the Accuracy/Brier trade-off
  is **not itself evidence of a software defect**.
- [Phase 6](scientific-validity-phase6.md): numerical reliability and
  reproducibility synthesis, with stated boundaries and operational
  artifact-retention risk.

## Verification performed for this Phase 7 report

- Queried GitHub PR status for #1954–#1958, #1960–#1963,
  #1965–#1969, #1986 and #1988: **all merged**.
- Checked head-commit workflow outcomes for remediation PRs #1954,
  #1955 and #1956: both **CI** and **Documentation links** succeeded
  for each. Also checked #1988: both workflows succeeded and the PR
  was merged.
- Reviewed existing Phase 2–6 audit reports and the recorded benchmark
  provenance; **no new runtime tests or independent mathematical
  calculations were performed in Phase 7**.

## Unresolved risks and release interpretation

- No **unresolved concrete correctness blocker is identified in the
  reviewed audit record**. This is a statement about recorded findings,
  **not** a guarantee that no latent bugs exist.
- Broad multiclass/ordinal probability calibration, extreme numerical
  conditioning, all Mixed/q/constraint combinations, and global optimizer
  convergence remain **unverified outside the targeted scope**.
- Synthetic benchmark evidence does not certify superiority on arbitrary
  real problems. Long-term storage of all GitHub Actions artifacts is
  not independently guaranteed.
- These limitations should appear in the final audit report. Do not
  transform them into new Phase 7 experiments without a reproducible,
  concrete high-severity defect.

## Phase 7 exit decision

**Close Phase 7 for the scoped scientific-validity audit**, subject to this
report's CI and merge. The concrete defects identified by the audit have
merged remediations and regression checks; no additional code changes
are justified by the reviewed evidence.

Next: **Phase 8** — final scientific-validity and release-quality assessment,
including verified scope, justified approximations, limited/unverified
behavior, known risks and explicit release claims.
