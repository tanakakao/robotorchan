# Phase 5: paired GP active-learning fit-epoch sensitivity (completed)

## Reproducible evidence

- Run: https://github.com/tanakakao/robotorchan/actions/runs/44982094508
- Artifact: `gp-al-epoch-sensitivity-44982094508` (ID `14738233222`).
- Run outcome: success; both manifests `status=complete`, seeds 0–9, zero failures.
- Both cohorts use the same synthetic wave-boundary problem, 20 initial labels,
  256-point candidate pool, fixed 1,024-point reference set, 50 additional labels,
  learning rate 0.03, CPU float64, and Random/Sobol/PredictiveEntropy policies.
- The only intended experimental change is the number of GP fitting epochs per refit:
  **8 versus 32**. Values below are means across ten paired seeds at 50 added labels.
- Raw `curves.csv`, `summary.csv`, `paired_differences.csv`, manifests, logs,
  and `gp-al-epoch-comparison.jsonl` are retained in the workflow artifact.

## End-of-budget means

| Policy | Accuracy (8 epochs) | Accuracy (32 epochs) | Brier (8 epochs) | Brier (32 epochs) |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.871973 | 0.883301 | 0.162987 | 0.082234 |
| Sobol | 0.881250 | 0.883984 | 0.157483 | 0.079394 |
| PredictiveEntropy | 0.889062 | 0.892676 | 0.184099 | 0.092995 |

## Paired changes: 32 minus 8 epochs

| Policy | Accuracy change mean | Accuracy change paired SD | Brier change mean | Brier change paired SD |
| --- | ---: | ---: | ---: | ---: |
| Random | +0.011328 | 0.013006 | -0.080752 | 0.005190 |
| Sobol | +0.002734 | 0.004975 | -0.078089 | 0.003535 |
| PredictiveEntropy | +0.003613 | 0.013295 | -0.091104 | 0.006480 |

Higher accuracy and lower Brier are desirable. Increasing fit epochs improved
mean Brier for **all three policies**, with a particularly large change for
PredictiveEntropy. The result supports **sensitivity to training budget**;
it does not establish full convergence or prove that the original calibration
difference was entirely caused by underfitting.

At 32 epochs, PredictiveEntropy has the highest mean Accuracy (0.892676) but
still the highest (worst) mean Brier (0.092995), compared with Random
(0.082234) and Sobol (0.079394). Thus the original Accuracy/Brier trade-off
**persists**, albeit with markedly improved probability scores.

## Scope and decision

This is a single synthetic problem, ten paired seeds, one fitting-budget contrast,
and one reference design. No statistical significance or universal algorithm
ranking is claimed. No classification-model defect is established by these data.

**Phase 5 empirical follow-up is complete.** Do not extend epoch sweeps or
add further datasets as part of this audit absent a concrete correctness defect.
Use the existing evidence and original audit scope for the final report.
