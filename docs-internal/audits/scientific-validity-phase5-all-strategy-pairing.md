# Scientific validity audit — Phase 5: all-strategy cross-q pairing and budget verification

Status: **archived trajectories directly verified** (2026-11-21). This extends [the qEI-only pairing check](scientific-validity-phase5-pairing.md) to every saved strategy.

## Evidence and procedure

Four downloaded GitHub Actions ZIP archives were inspected directly using Python `zipfile`, `json`, and NumPy. For each problem (Branin, Hartmann6), each strategy (`random`, `sobol`, `qEI`, `qNEI`), and seed 0–9, compare the first 20 rows of `tensors.X.data` and `tensors.Y_truth.data` between q=1 and q=3. Also check all saved trajectory lengths, `completed_batches[-1]`, and that every completed-batch increment is at most q.

| Problem | q=1 artifact ID | q=3 artifact ID | Paired strategy/seed combinations |
| --- | ---: | ---: | ---: |
| Branin | 14334959201 | 14338740901 | 40 |
| Hartmann6 | 14339551141 | 14337814349 | 40 |

**Result:** All **80** cross-q strategy/seed pairs have exactly identical initial X (`max_abs_diff=0`). Branin initial Y_truth is exactly equal; Hartmann6 initial Y_truth differs by at most `1.1102230246251565e-16` (float64 rounding). Each of the 160 trajectory files has 70 evaluated X and Y_truth rows, `completed_batches[-1] == 50`, and no batch increment exceeding its configured q. No violations were found in these checks.

## ZIP SHA-256 checksums

| Artifact ID | SHA-256 |
| --- | --- |
| 14334959201 | `b2877c135c04854b224e5460879621a123a40ed2be470ff7c3a280ad1d316623` |
| 14338740901 | `306dc721c88f09771889c0af004049a5bc40154ed9ccb9b8bd7bf21d2679ba82` |
| 14339551141 | `85252e8633824db21b4280df2f38205016eac47471c763290ea452d8c34906b9` |
| 14337814349 | `424fc7447ad3a18e5af9271ef890d412be948a8d03dc9aeb4761df03400a9df6` |

These checksums identify the inspected ZIP bytes; they are **not** proof of independent long-term archival.

## Interpretation

The initial-design pairing gate is now satisfied **for all four archived strategies** in the two synthetic problems. The inspected files also satisfy the tested evaluation-budget and batch-size invariants. This does not independently verify every observation-noise or runtime property, establish end-to-end correctness, or make q=1 versus q=3 superiority statistically significant. The existing exploratory cross-q intervals include zero; different q values imply different model-update counts.

The [experiment-source comparison](scientific-validity-phase5-provenance.md) found only a report-generation dtype fix and a report-recovery workflow between original experiment SHAs.

## Remaining Phase 5 work

1. Verify observation and metric invariants as needed for stronger statistical claims, and record any per-seed failures explicitly.
2. Preserve ZIPs and environment provenance outside expiring Actions artifacts.
3. Select a **new** Active Learning or constrained-BO multi-seed study with prospective fairness settings, a baseline, and a defined primary metric. Do not reuse this BO-only evidence to claim classification or constraint performance.
4. Retain the uncertainty intervals and avoid universal superiority claims.
