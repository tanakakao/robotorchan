# Scientific validity audit — Phase 5: four-cell provenance follow-up

Status: **source-diff and artifact-metadata verified; trajectory pairing not yet verified** (2026-11-21).

This note follows [Phase 5 evidence inventory](scientific-validity-phase5-inventory.md). No new benchmark was executed.

## Artifact availability checked through GitHub Actions metadata

| Cohort | Run | Artifact ID | Expired at check |
| --- | ---: | ---: | --- |
| Branin q=1 (recovered report and original trajectories) | 44100622002 | 14334959201 | No |
| Branin q=3 | 44105379822 | 14338740901 | No |
| Hartmann6 q=1 | 44106195610 | 14339551141 | No |
| Hartmann6 q=3 | 44107676004 | 14337814349 | No |

The artifact listing confirms metadata and non-expired status **at the check time**, not file integrity, permanent retention, or equality of initial observations. Do not claim the ZIP contents were independently audited in this step.

## Original experiment source comparison

Compared `5fe440a8318a6c9a97e401180379fe16bc13bab3...8b1b8121b5fdd6264e50c902f1facafbf54d4fd9` using GitHub's compare API. The later commit is four commits ahead, with exactly two changed paths:

1. `src/robotorchan/benchmarks/empirical_report.py`: change `torch.tensor([0.025, 0.975])` to `bootstrap.new_tensor([0.025, 0.975])` for `torch.quantile` dtype compatibility. This is a **report-generation** change, not a candidate selection or experiment runner change.
2. `.github/workflows/benchmark-empirical-report-recovery.yml`: add a workflow to regenerate the report from saved trajectories without rerunning BO.

Thus the compare diff contains **no changes to benchmark optimization, problem definitions, or acquisition implementation** between these two SHAs. This narrows the source-code comparability concern, but does not independently prove runtime-environment identity or equal initial observations.

## Still required before confirmatory cross-q comparison

- Retrieve and inspect the four original artifacts, including raw trajectories and manifests.
- Check paired initial X and observed Y **by problem and seed**, across q=1 and q=3, using the persisted data rather than matching seed labels alone.
- Check the reported initial count, evaluation budget, noise policy, metric direction, model/acquisition/optimizer settings, and runtime environment.
- Preserve a durable copy of artifacts and source provenance; Actions artifacts may expire.
- If the paired initial observations or relevant settings differ, retain cross-q results as exploratory or run a new prospectively paired experiment.

## Decision

**Provenance gate partially satisfied; cross-q inference gate remains open.** The source diff does not indicate an optimizer change, and all four artifact records were unexpired at inspection. Initial-design pairing and ZIP contents have **not** been checked, so the existing exploratory q=1 vs q=3 intervals must not be promoted to confirmatory evidence.
