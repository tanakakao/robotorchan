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
