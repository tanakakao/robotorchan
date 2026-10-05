# Classification Active Learning

Classification acquisitions keep latent-function uncertainty, class-probability uncertainty,
and predictive label uncertainty separate.

- `PredictiveEntropy` scores posterior-predictive class entropy.
- `MarginUncertainty` scores ambiguity between the two most probable classes.
- `ProbabilityVariance` estimates posterior variance of class probabilities from samples.
- `BALD` estimates mutual information between the predicted label and latent uncertainty.
- `LatentStraddle` is binary-only and scores uncertainty around the latent boundary `f(x) = 0`.

All five acquisitions currently support `q=1`. The probability-based acquisitions use the
common classification prediction contract so they can extend to multiclass models without
hard-coding Bernoulli labels. `LatentStraddle` remains explicitly binary because its zero
latent boundary has binary-classification semantics.


## Ensemble classification

Classification ensembles use the same acquisition classes through the probability-space model
contract. No ensemble-specific acquisition wrapper is required.

- `PredictiveEntropy` needs only the ensemble predictive mean from `predict_proba(X)`.
- `MarginUncertainty` needs only the predictive class-probability vector.
- `ProbabilityVariance` uses between-member probability variance.
- `BALD` uses predictive entropy minus weighted expected member entropy.

For weighted ensembles, all four scores inherit the normalized model weights from the
`ClassificationEnsemblePosterior`. Thus Bayesian model averaging weights, when supplied as
posterior model probabilities, also propagate into active-learning scores.

`LatentStraddle` is intentionally excluded from heterogeneous ensembles because different
members need not share a latent function or even a latent probabilistic representation.


## Calibration and acquisition semantics

Post-hoc probability calibration composes with probability-space classification acquisitions through
the existing prediction contract. A `CalibratedBinaryClassifier` therefore changes the
probabilities consumed by predictive entropy and margin uncertainty, and it calibrates posterior
probability samples before probability variance and BALD are computed.

The same rule applies to classifier-backed feasibility in Bayesian optimization:
`ClassificationProbabilityOfFeasibility` consumes `predict_proba`, so a calibrated classifier
produces calibrated feasibility probabilities without a separate calibration-specific acquisition
wrapper. The calibration transform remains differentiable for temperature scaling, preserving
gradient-based candidate optimization.

`LatentStraddle` is intentionally different. It operates on the latent GP decision boundary and
therefore delegates to the unchanged latent posterior. Post-hoc probability calibration does not
change its score. This distinction prevents probability calibration from being misrepresented as a
change to the latent GP posterior.

Calibration quality should be checked on held-out data before calibrated probabilities are used for
BO constraints or active-learning decisions. NLL, Brier score, ECE, and MCE are diagnostics rather
than acquisition functions.
