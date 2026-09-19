# Phase 16: Robust × Mixed coverage audit

## Purpose

The robust-surrogate roadmap added statistically distinct observation and input
models after the general mixed-model work. This audit determines which robust
concerns need a native mixed surrogate, which should be composed with existing
mixed infrastructure, and which are already categorical by construction.

The rule is unchanged: do not create a `MixedXxxGP` merely to multiply class
names. Add a mixed implementation only when categorical design variables must
participate in the surrogate covariance or inference procedure.

## Current baseline

`MixedSingleTaskGP` is the standard exact-GP mixed surrogate. It uses BoTorch's
mixed covariance and retains robotorchan raw data / `make_mll()` conventions.

The robust models added in Phases 6-14 are not automatically mixed merely
because a tensor can contain integer-valued category codes. Integer codes must
never be passed to an ordinary continuous kernel as if ordinal distances were
meaningful.

## Classification

| Robust concern | Mixed status | Decision |
| --- | --- | --- |
| sparse relevance-pursuit outliers | audit existing implementation | Prefer shared mixed covariance / base model rather than duplicating the robust inference algorithm. |
| iterative heteroskedastic noise | missing | Add mixed response and mixed log-noise processes; categorical variables affect both latent functions. |
| joint heteroskedastic noise | missing | Add mixed-capable variational covariance or factor the mixed covariance into the existing joint variational implementation. |
| Student-t residuals | missing | Share Student-t likelihood/training objective; use a mixed latent response GP. |
| Gaussian uncertain continuous training inputs | conditional | Do not combine naively. A valid model needs expected continuous covariance multiplied/composed with a categorical covariance. |
| replicate-derived noise | compositional | Replicate aggregation is input-type agnostic. Refactor aggregation so the resulting fixed-noise data can feed `MixedSingleTaskGP`; avoid a separate statistical algorithm. |
| contamination-mixture residuals | missing | Share contamination likelihood/training objective; use a mixed latent response GP. |
| nonstationary Gibbs covariance | missing | Build a mixed covariance with Gibbs behavior on continuous dimensions and a categorical covariance on categorical dimensions. |
| uncertain categorical training inputs | already categorical-specific | Do not add a second ordinary Mixed wrapper. Extend only if additional deterministic categorical columns are required by a concrete use case. |

## Implementation priorities

### Priority A: reusable mixed latent-GP infrastructure

Student-t, contamination mixture, and joint heteroskedastic models currently
depend on variational latent response models. Implement one reusable mixed
variational latent GP / covariance path first. Then reuse the existing
likelihood and training objectives rather than copying them into parallel
classes.

This is the highest-leverage step because it prevents three independent mixed
implementations from drifting.

### Priority B: exact mixed covariance injection

Iterative heteroskedastic and nonstationary models are exact-GP families.
Provide mixed covariance construction through shared kernel factories:

- stationary mixed covariance for response/noise processes;
- Gibbs continuous covariance × categorical covariance for nonstationarity.

Do not treat categorical integer codes as continuous Gibbs inputs.

### Priority C: preprocessing composition

Refactor replicate aggregation into a reusable helper returning:

- unique design rows;
- group means;
- variance of each group mean;
- replicate counts;
- within-group variance.

The mixed exact GP can then consume the aggregated rows with fixed observation
variance. A dedicated `MixedReplicateNoiseSingleTaskGP` is optional API sugar
only if it adds meaningful validation; it must not duplicate aggregation logic.

### Priority D: uncertain continuous × categorical inputs

Only implement after defining the probability model. For deterministic
categorical variables and Gaussian-uncertain continuous coordinates, the
natural covariance is an expected continuous kernel composed with a categorical
kernel. This is not equivalent to sending integer category codes through
`UncertainInputSingleTaskGP`.

## Explicit non-goals

- no integer-category jitter;
- no ordinal Euclidean distance for nominal categories;
- no `MixedXxxGP` class solely for naming symmetry;
- no compatibility aliases or deprecated wrappers;
- no cross-product with PCA/PLS/ALEBO unless the statistical model itself
  requires it;
- no simultaneous implementation of every row before shared infrastructure is
  established.

## Proposed implementation sequence

1. audit/refactor reusable mixed covariance and mixed variational infrastructure;
2. mixed Student-t and contamination models using the shared variational path;
3. mixed iterative/joint heteroskedastic models;
4. mixed nonstationary Gibbs model;
5. replicate-noise mixed composition;
6. expected continuous-uncertainty × categorical covariance, if retained after
   design review;
7. public-contract, acquisition, dtype/device, state-dict, and documentation
   coverage;
8. final Robust × Mixed matrix audit.

## Exit criteria

Phase 16 is complete when every robust family is explicitly classified as
native mixed, compositional, categorical-specific, or deliberately unsupported,
and the implementation roadmap avoids duplicated inference code.
