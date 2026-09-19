# Phase 9: robust surrogate coverage audit

## Purpose

Phases 1-8 established the main robustness semantics and the surrogate families
that genuinely require different probabilistic models. Phase 9 is a consolidation
gate before adding more model classes.

The objective is to verify that robotorchan covers distinct statistical
assumptions without creating combinatorial wrappers.

## Current model-level robustness coverage

### Sparse gross outliers

Use `RobustRelevancePursuitSingleTaskGP`.

This family assumes a relatively small set of observations need explicit sparse
correction. It is not equivalent to globally heavy-tailed residuals.

### Input-dependent Gaussian observation variance

Two models intentionally remain available:

- `HeteroskedasticSingleTaskGP`: iterative two-GP approximation;
- `JointHeteroskedasticSingleTaskGP`: joint variational response/log-noise model.

The first is a practical baseline. The second models both latent processes in a
joint training objective.

### Globally heavy-tailed observation residuals

Use `StudentTSingleTaskGP`.

This changes the observation likelihood and is distinct from both sparse
relevance-pursuit corrections and heteroskedastic Gaussian noise.

### Uncertain observed training inputs

Use `UncertainInputSingleTaskGP`.

This integrates diagonal or full-covariance Gaussian uncertainty in the observed input locations
into the covariance. It is distinct from perturbing a candidate during robust
decision evaluation.

## Robustness that should remain compositional

The following do not justify new surrogate cross-product classes by themselves:

- candidate-time input perturbation;
- explicit environmental/scenario factors;
- expectation, mean-variance, worst-case, VaR, CVaR, and SN aggregation;
- PCA/PLS/random-projection/neural dimensionality reduction when uncertainty is
  defined in raw design coordinates;
- ALEBO strategy-level robustness.

The preferred architecture remains:

```text
surrogate model
    × uncertainty/scenario generator
    × risk aggregation
    -> acquisition
```

## Deliberately unsupported cross-products

Do not add classes such as:

- `RobustPCAGP`;
- `StudentTPCAGP`;
- `StudentTHeteroskedasticGP`;
- `UncertainInputPLSGP`;
- `RobustALEBOGP`.

A cross-product model is justified only when the additional concern changes the
likelihood, kernel, posterior, or inference procedure in a way that cannot be
composed externally.

## Remaining gaps worth evaluating

The next additions should be selected by a concrete statistical gap rather than
by model-count growth. Candidates are:

1. replicated-observation noise models when replicate structure is known;
2. uncertain categorical training inputs, only if a defensible probability
   model and kernel expectation are defined;
3. explicit contamination/mixture likelihoods if Student-t and relevance
   pursuit do not cover the required outlier mechanism;
4. nonstationary observation/process models only when a use case requires
   local behavior that current kernels cannot represent.

## Exit criteria

Phase 9 is complete when:

1. every current robust surrogate has a distinct documented assumption;
2. candidate/environment/risk robustness is clearly separated from surrogate
   modeling;
3. no stale documentation still describes implemented models as missing;
4. future work is prioritized by statistical semantics rather than wrapper
   combinations;
5. public model contracts and CI remain unchanged by the documentation audit.
