# Phase 5: binary GP active-learning full-cohort empirical audit

## Provenance and completion

- GitHub Actions: https://github.com/tanakakao/robotorchan/actions/runs/44968637695
- Artifact: `gp-al-full-44968637695` (artifact ID `14732666592`).
- Code revision (recorded `git_sha`): `0380c6cb9d85be8e8d7ca62d24d525cbe9bf4ea9`.
- Workflow result: `success`; manifest `status=complete`; ten seeds `0..9` completed, zero failed.
- Preserved files: `curves.csv`, `paired_differences.csv`, `summary.csv`, `failures.json`,
  `manifest.json`, run log, and paired endpoint/AUC analysis.
- This document records observed results; it does not rerun, modify or embed a fabricated experiment.

## Experimental protocol

- Binary synthetic `wave` boundary in two continuous dimensions; single fixed independent
  1,024-point reference design (Sobol seed `93827`).
- Twenty initial labeled points, a shared 256-point candidate pool per seed, and 50 selected
  new labels, one per step.
- Three policies: Random, Sobol-pool order, and GP `PredictiveEntropy`.
- The **same** `BinarySingleTaskGPClassifier` is refitted each step for all three policies;
  eight epochs per fit, learning rate `0.03`, double precision, CPU.
- Ten paired random seeds; classification Accuracy (larger is better) and Brier score
  (smaller is better), measured on the fixed reference design.
- Per-seed endpoint differences and normalized trapezoidal learning-curve AUC differences
  are relative to the matching Random seed. Reported SD is **sample SD across ten paired
  differences**, not a confidence interval or significance claim.

## End-of-budget absolute performance

| Policy | Accuracy mean | Accuracy SD | Brier mean | Brier SD |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.871973 | 0.012753 | 0.162987 | 0.003906 |
| Sobol | 0.881250 | 0.005604 | 0.157483 | 0.003725 |
| PredictiveEntropy | 0.889062 | 0.005827 | 0.184099 | 0.005112 |

## Paired differences vs. Random

| Policy | Metric | Final mean difference | Final paired SD | Normalized AUC mean difference | AUC paired SD |
| --- | --- | ---: | ---: | ---: | ---: |
| Sobol | Accuracy | +0.009277 | 0.012440 | +0.005821 | 0.033404 |
| PredictiveEntropy | Accuracy | +0.017090 | 0.013144 | +0.024106 | 0.030026 |
| Sobol | Brier | -0.005503 | 0.005325 | -0.002827 | 0.004223 |
| PredictiveEntropy | Brier | +0.021113 | 0.006720 | +0.008191 | 0.004656 |

At the final step, PredictiveEntropy improved Accuracy relative to paired Random in all
10/10 seeds, **but worsened Brier score in all 10/10 seeds**. Sobol improved Accuracy in
7/10 seeds and improved Brier in 9/10 seeds.

**Interpretation:** PredictiveEntropy achieved better classification decisions on this
particular synthetic boundary under this configuration, but worse probability predictions
than Random or Sobol. Accuracy and proper probability scoring are not interchangeable
objectives. This contrast does **not** establish a software defect, a general AL ranking,
or statistical significance.

## Threats to validity and remaining decisions

1. Only one synthetic `wave` boundary, one fixed reference design, one finite candidate
   pool protocol, and ten paired seeds were evaluated; no out-of-distribution or real
   data evidence exists in this run.
2. Eight epochs per refit may be insufficient for calibrated probability estimation;
   do not assume fitted GP classification converged. Check convergence or sensitivity
   **before** interpreting the Brier gap as inherent to entropy sampling.
3. `InputDataWarning` about outcome standardization is emitted by BoTorch. Binary 0/1
   labels must not be blindly standardized like continuous Gaussian regression targets.
   Inspect how the classification model uses its training labels and whether the warning
   is benign; do not suppress warnings without justification.
4. The outcome is synthetic, deterministic, and thresholded; entropy sampling
   prioritizes uncertainty over probability calibration. Consider reliability diagrams,
   log loss, and sensitivity to the number of fit epochs and alternative boundaries in
   follow-up *separate* experiments, while keeping paired budgets fixed.
5. The model refits from scratch at each step; these results assess this
   particular implementation/protocol, not all GP-based active learning methods.

**Phase 5 decision:** full experiment **executed and archived**, numerical comparison
**completed**, superiority/generalization claim **not supported**. Preserve the raw
artifact and its run URL as the primary data source. Next narrow investigation:
validate fitted classification probability quality and training convergence before
claiming or modifying a classification-acquisition behavior.
