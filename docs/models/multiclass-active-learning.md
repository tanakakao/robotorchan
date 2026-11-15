# Multiclass classification: active-learning integration

The multiclass variational GP classifiers expose the same class-probability
prediction interface as the binary classifiers. The following acquisition
functions reuse the existing implementations without multiclass-specific
wrappers.

| Acquisition | Standard multiclass | Mixed multiclass | q > 1 |
| --- | --- | --- | --- |
| PredictiveEntropy | Yes | Yes | Use qPredictiveEntropy |
| MarginUncertainty | Yes | Yes | Use qMarginUncertainty |
| ProbabilityVariance | Yes | Yes | Use qProbabilityVariance |
| BALD | Yes | Yes | Use qBALD |
| qPredictiveEntropy | Yes | Yes | Mean marginal entropy |
| qMarginUncertainty | Yes | Yes | Mean marginal margin |
| qProbabilityVariance | Yes | Yes | Mean marginal variance |
| qBALD | Yes | Yes | Exact joint label enumeration, C**q <= 256 |
| LatentStraddle / qLatentStraddle | No | No | Binary latent-zero boundary only |

**Important:** The three marginal q-acquisitions average per-candidate scores;
they do not account for candidate redundancy. qBALD uses coherent latent
posterior samples and enumerates joint class labels. Its enumeration cost
increases exponentially with q. For binary models, the existing q <= 8
contract remains unchanged.

## Candidate generation and constraints

- `SobolSearchStrategy` accepts the multiclass acquisition functions and
  returns continuous-space q-batches.
- `optimize_acqf_sampling` supports mixed candidates through
  `MixedVariableSpace`, and input constraints through
  `CandidateConstraints`. Linear constraints on continuous dimensions are
  exercised by the multiclass integration test.
- `MixedSpaceStrategy` and the BoTorch mixed backend require explicit
  `fixed_features_list` enumerating discrete assignments. They are separate
  from the sampled mixed-space path and are **not** validated by the
  multiclass Phase 4 integration tests.
- `CandidateConstraints` constrain candidate **inputs**, not predicted class
  probabilities. Safety constraints on model outputs require a different
  acquisition formulation.
- The model's `cat_dims` and the optimizer's `MixedVariableSpace` must
  describe the same categorical coordinates. The classifier does not
  automatically configure the optimizer.

This document describes tested integration paths, not a blanket guarantee
that every optimizer and constraint combination is supported.


## Reproducible benchmark (Phase 5)

Run the standalone comparison with:

```console
python examples/benchmarks/multiclass_active_learning.py --seeds 3 --budget 5 --epochs 8
```

The script writes `multiclass_al_results.csv` with one row per seed,
input type (continuous or Mixed), acquisition method, and labeling step.
It compares random pool sampling against predictive-entropy selection.
Both methods use the same initial 15 labeled examples, 75 candidate
pool points, 30 held-out test points, and label budget for each seed.

The reported metrics are held-out accuracy (higher is better),
negative log likelihood (lower is better), and multiclass Brier score
(lower is better). This is a **synthetic smoke benchmark**, not evidence
that one acquisition is generally superior. Repeat with more seeds,
multiple data-generating processes, class imbalance and realistic
label costs before drawing performance conclusions. Training uses a
small number of epochs for speed; tune this separately for serious use.
