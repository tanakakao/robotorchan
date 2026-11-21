# Scientific validity audit — Phase 5: archived cross-q initial-pairing verification

Status: **archived trajectories inspected; initial design paired** (2026-11-21).

This follow-up inspects the **downloaded ZIP contents**, not merely GitHub artifact metadata. No optimizer was rerun and no new benchmark measurements were generated.

## Sources and inspection

| Problem | q=1 artifact ID | q=3 artifact ID | Seeds |
| --- | ---: | ---: | --- |
| Branin | 14334959201 | 14338740901 | 0–9 |
| Hartmann6 | 14339551141 | 14337814349 | 0–9 |

For each problem, load `qEI_seed{seed}.json` from both archives and compare the first `initial_points=20` rows of `tensors.X.data` and `tensors.Y_truth.data` across q=1 and q=3. The manifest identifies the problem, q, seeds, initial points, evaluation budget, source SHA, Python/PyTorch/BoTorch/GPyTorch versions, and RNG policy.

## Findings

- **Initial X:** exactly equal (maximum absolute difference 0.0) for all 10 paired seeds in **both** Branin and Hartmann6.
- **Initial Y_truth:** Branin exactly equal for all 10 seeds; Hartmann6 differs only at floating-point rounding scale, maximum absolute difference **1.1102230246251565e-16** across seeds. This is numerical equality within a sensible float64 tolerance, **not** bitwise equality.
- **Manifest fields:** Hartmann6 q=1 and q=3 manifests differ only in q. Branin manifests differ in q, source SHA, and Linux platform string; the recorded Python, torch, botorch, gpytorch, dtype, device, budget, restart/raw-sample settings, noise policy, and RNG policy match.
- **Source SHA comparison:** [provenance review](scientific-validity-phase5-provenance.md) establishes that the Branin experiment-SHA diff comprises only a report dtype fix and recovery workflow, not an optimization source change.

This validates **initial design pairing** across q in the inspected qEI trajectories. It does not prove equality of subsequent candidate decisions (nor should it: q changes the decision policy), equal runtime hardware, or identical numerical trajectories. It also does not establish pairing of every strategy independently: only the archived qEI files were directly compared here.

## Reproduction procedure

Download the four artifact ZIPs linked from [four-cell evidence](../verification/benchmarks/phase3-four-cell-evidence.md). Read the ZIP entries directly with Python's standard-library `zipfile` and `json`. For each problem and seed, compare the first 20 rows of `tensors.X.data` and `tensors.Y_truth.data` in `qEI_seed{seed}.json`. Use an explicit float64 tolerance for Hartmann6 objective values, not exact JSON equality.

## Statistical interpretation

The earlier cross-q comparison can now be described as **paired at initial design for qEI**. This removes one specific limitation from the exploratory analysis but does not retroactively make its bootstrap intervals confirmatory: the q comparison was not pre-registered, it has only 10 seeds, and q=1 versus q=3 entails different sequential update counts. The reported intervals include zero. Do not claim superiority of either q.

## Next gates

1. Verify the same initial pairing across Random, Sobol, and qNEI trajectories, plus completed evaluation counts and failed-seed accounting.
2. Archive original ZIPs durably and record checksums; GitHub Actions retention is finite.
3. Choose one targeted Active Learning or constrained-BO empirical gap for a prospectively specified, multi-seed comparison using the existing runner and fairness contract.
