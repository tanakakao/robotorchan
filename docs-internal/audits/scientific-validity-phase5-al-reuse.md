# Scientific validity audit — Phase 5: adaptive AL implementation reuse review

Status: **source-level review; no new measurements** (2026-11-21).

This follow-up clarifies the earlier [AL eligibility audit](scientific-validity-phase5-active-learning-eligibility.md): although the passive `empirical_active_learning.py` runner does not adapt queries to labels, **other adaptive AL assets already exist** and must be considered before implementing another runner.

## Three distinct existing pathways

| Pathway | Existing file | Candidate policy | Predictor and limits |
| --- | --- | --- | --- |
| Passive binary baseline | `src/robotorchan/benchmarks/empirical_active_learning.py` | Pre-generated Random or Sobol | Fixed-bandwidth kernel label smoother; not GP; 10-seed full learning curves |
| Adaptive binary heuristic | `src/robotorchan/benchmarks/empirical_active_learning_entropy.py` | Sequential entropy maximization over 256 Sobol candidates | Same fixed kernel smoother; label-adaptive, **not** GP predictive entropy or BALD; 10-seed protocol, but no new execution claimed here |
| Adaptive multiclass GP example | `examples/benchmarks/multiclass_active_learning.py` | Random versus entropy of fitted GP class probabilities on a finite candidate pool | Existing `MulticlassSingleTaskGPClassifier.fit` and `predict_proba`; illustrative defaults 3 seeds, 5 acquired labels, 8 epochs; **not** a certified binary 10-seed benchmark |

The previous audit's claim that *its particular passive runner* lacks GP-adaptive candidate selection remains correct. It must **not** be interpreted as saying the repository has no adaptive AL experiment or GP classification example.

## Reuse and smallest missing link

- The production binary model `BinarySingleTaskGPClassifier` exposes `fit` and `predict_proba`; `PredictiveEntropy` in `src/robotorchan/acquisition/active_learning/classification.py` wraps a classification model and requires q=1.
- Reuse `active_learning_labels`, `classification_accuracy`, `brier_score`, paired initial Sobol seeds, and fixed independent reference set from the existing binary benchmark modules.
- Reuse the multiclass example's train → fit → predict → select → append loop structure, but **do not** silently use the multiclass model, its 3-seed settings, or its entropy reimplementation as a binary performance result.
- For a first binary GP-adaptive smoke test, freeze one boundary, one seed, small acquisition budget, and public model/acquisition calls. Verify prediction shape, finite probabilities, oracle isolation, and update counts. Then use the frozen ten-seed protocol for measured comparisons.
- A fair comparison of *candidate policies* should fit the **same GP classifier** on Random, Sobol, and entropy-acquired observations before computing reference metrics. Comparing a GP classifier to the kernel-smoother baseline confounds candidate policy and predictive model.
- Freeze candidate-pool policy across adaptive and passive methods, or explicitly distinguish pool-based versus direct-space policies. Fix model-training RNG independently of acquisition strategy and record training failures.

## Explicit exclusions

No benchmark is executed in this PR; no GP advantage, convergence rate, or sample-efficiency claim follows from existing source files alone. The kernel entropy heuristic must never be relabeled `PredictiveEntropy` or `BALD` for a GP. A smoke run is not a multi-seed performance evaluation.

## Next implementation slice

Implement a minimal, reusable **binary GP classifier closed-loop comparison** on `wave_boundary` with identical paired initial designs, reference data, label budget, candidate-pool protocol, and predictor for all three candidate policies. Add only targeted smoke/contract coverage, then run the existing ten-seed protocol and archive raw trajectories before making statistical claims.
