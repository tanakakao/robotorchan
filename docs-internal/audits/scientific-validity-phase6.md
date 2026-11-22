# Scientific validity audit — Phase 6: numerical reliability, reproducibility and benchmark synthesis

Status: **scoped evidence review complete, with explicit limitations** (2026-11-22).

This is a synthesis of **existing measured evidence**, not a new numerical experiment.
The purpose is to close the original Phase 6 review without repeating the extensive
Phase 2–5 validation or expanding the audit indefinitely.

## Evidence and dispositions

| Subject | Existing evidence | Phase 6 disposition |
| --- | --- | --- |
| Exact GP covariance and posterior reference arithmetic | [Phase 2](scientific-validity-phase2.md) | Verified in targeted reference cases, not all kernels or numerical regimes |
| Multiclass latent covariance and synthetic fitting | Phase 2, PRs #1957–#1958 | Verified in targeted cases; no universal class-probability calibration claim |
| Ordinal probability posterior | Phase 2 targeted tests | Limited; no broad quadrature/calibration certification |
| Acquisition formulae, Gaussian MI, joint qBALD | [Phase 3](scientific-validity-phase3.md) | Independent targeted numerical references; MC estimation error and model assumptions retained |
| Mixed, q-batch, hard constraints, infeasibility | [Phase 4](scientific-validity-phase4.md) | Feasibility and sampled-pool selection checked in targeted cases; not global convergence |
| BO sample efficiency, Branin and Hartmann6, q=1/3 | [Four-cell measured evidence](../verification/benchmarks/phase3-four-cell-evidence.md) and [all-strategy pairing](scientific-validity-phase5-all-strategy-pairing.md) | Ten seeds per cell, budget and initial design checked for 80 cross-q pairs; limited to these synthetic problems |
| GP binary AL, 8 epochs | [Ten-seed evidence](scientific-validity-phase5-gp-al-full-results.md) | Measured Accuracy/Brier trade-off; no universal ranking |
| GP binary AL, 8 versus 32 epochs | [Sensitivity results](scientific-validity-phase5-gp-al-epoch-sensitivity-results.md) | All 10 seeds completed in both conditions; fitting-budget sensitivity demonstrated, not convergence proof |

## Numerical reliability

- Reference tests provide independent checks of selected covariance, likelihood,
  acquisition and constraint arithmetic. They do **not** imply exhaustive coverage
  of ill-conditioned kernels, extreme lengthscales, near-singular covariance,
  all floating-point dtypes, accelerators or every optimizer.
- In the paired binary GP AL experiment, increasing fitting epochs from 8 to 32
  reduced mean Brier score for all three policies. This demonstrates **material
  sensitivity to training budget**; it does not prove convergence or identify
  an implementation defect. At 32 epochs, PredictiveEntropy still has the
  highest mean Accuracy and the worst mean Brier among the three policies.
- The multiclass and ordinal models are **included** in this review through the
  Phase 2 evidence and limitations. No new classification-only test program
  is required to close Phase 6.

## Reproducibility and statistical interpretation

- The four BO benchmark cells contain ten seeds each, a fixed 20-point initial
  design and 50 additional evaluations, CPU float64, with original source SHAs
  and artifact identifiers recorded. The original Branin q=1 report was
  recovered from saved trajectories after a reporting dtype fix; the source
  comparison found no optimization-code changes.
- Archived trajectory inspection established identical initial X for all 80
  cross-q problem/strategy/seed pairs, with floating-point-level initial
  Hartmann6 Y differences. This supports initial-design comparability but
  **does not** make cross-q differences statistically conclusive; exploratory
  q=1 versus q=3 intervals include zero and model update counts differ.
- The binary GP AL experiments archive per-seed curves, manifest, failures,
  and paired analyses; the 8-versus-32 epoch workflow succeeded with zero
  failed seeds. Single-problem, ten-seed evidence is not generalization
  evidence, and paired SD must not be described as a confidence interval.
- GitHub Actions artifact IDs and checksums identify inspected evidence;
  **permanent independent archival is not established**. Retention/expiry
  remains an operational risk, not a newly discovered mathematical defect.

## Known limits, without expanding this audit

1. No exhaustive numerical stress certification for near-singular kernels,
   arbitrary Mixed/q/constraint combinations or all devices.
2. No global optimizer convergence guarantee.
3. No broad probability calibration certification for multiclass/ordinal or
   other classification variants.
4. No universal performance superiority claim from synthetic benchmarks.
5. No demonstrated permanent archive of every experimental artifact.

These are **documented scope limits**, not automatically release-blocking defects.
They must not be silently reclassified as verified, and they do not trigger
new experiments unless Phase 7 identifies a specific high-severity correctness bug.

## Phase 6 decision

**Phase 6 scoped synthesis: complete with limitations.** Existing numerical
references, archived paired experiments and provenance checks are sufficient
to report the evidence actually obtained. No additional benchmark or test
suite is authorized by this Phase 6 closure.

Next: **Phase 7**, reconcile the already identified and fixed defects with
the current main branch and CI, identify any unresolved *concrete* correctness
blockers from the existing audit record, and avoid speculative re-auditing.
Then **Phase 8** publishes the final audit conclusion and evidence boundaries.
