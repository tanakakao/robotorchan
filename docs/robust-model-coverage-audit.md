# Phase 9: robust surrogate coverage audit

> **文書種別: audit / closeout record.** 現在の利用者向け選択ガイドは [robust-model-support.md](robust-model-support.md) を参照してください。


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

### Replicate-derived observation noise

Use `ReplicateNoiseSingleTaskGP` when repeated measurements are available at
identical design conditions. It estimates empirical within-group variance and
uses variance of the group mean as fixed likelihood noise.

### Explicit observation contamination

Use `ContaminatedSingleTaskGP` when observations are assumed to arise from a
mixture of nominal and broader gross-error Gaussian components. This is distinct
from global heavy tails and sparse relevance-pursuit corrections.

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

### Nonstationary latent-process behavior

Use `NonstationarySingleTaskGP` when the latent response smoothness itself changes
across the design space. Its Gibbs covariance is distinct from input-dependent
observation noise.

### Uncertain categorical training inputs

Use `UncertainCategoricalSingleTaskGP` when an observed category is uncertain but
an explicit probability vector over the finite category set is available. The
model marginalizes that uncertainty through a positive-semidefinite expected
categorical covariance. It does not jitter integer category codes or impose an
ordinal distance.

## Remaining gaps worth evaluating

No major surrogate-level robustness gap identified by this audit remains open.

Further model classes should require a new, concrete statistical assumption that
cannot be represented by the existing surrogate, scenario-generator, and risk
aggregation layers. In particular, do not create cross-product classes merely
to combine already-supported concerns.

## Exit criteria

Phase 9 is complete when:

1. every current robust surrogate has a distinct documented assumption;
2. candidate/environment/risk robustness is clearly separated from surrogate
   modeling;
3. no stale documentation still describes implemented models as missing;
4. future work is prioritized by statistical semantics rather than wrapper
   combinations;
5. public model contracts and CI remain unchanged by the documentation audit.

## Phase 15 closeout

The robust-surrogate audit is closed after implementation of:

- sparse relevance-pursuit robustness;
- iterative and joint heteroskedastic Gaussian noise;
- Student-t observation likelihood;
- diagonal/full-covariance uncertain continuous training inputs;
- replicate-derived fixed observation noise;
- contamination-mixture observation likelihood;
- nonstationary Gibbs covariance;
- probability-valued uncertain categorical training inputs.

Candidate perturbation, environmental scenarios, and risk measures remain
compositional concerns rather than reasons to multiply surrogate classes.
