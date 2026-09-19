# Phase 10: ALEBO and specialized high-dimensional Mixed audit

## Decision

Phase 10 does **not** introduce a nominal `MixedALEBOGP`.

The current ALEBO implementation has two coupled pieces:

1. `ALEBOGP` models coordinates in the continuous embedded space with a full
   Mahalanobis RBF metric.
2. `ALEBOStrategy` samples and optimizes a continuous polytope, then maps the
   embedded point back to the ambient box with a linear pseudoinverse.

A raw categorical coordinate cannot pass through this projection without
losing discreteness. Adding a categorical kernel only to the surrogate would
therefore advertise Mixed support while the search layer can still generate
invalid fractional categories. That violates robotorchan's rule that ALEBO
Mixed support must include embedding, feasibility, acquisition optimization,
and candidate reconstruction.

## Current capability boundary

| Component | Continuous ALEBO | Raw categorical design variables |
| --- | --- | --- |
| Surrogate geometry | Mahalanobis RBF | not supported |
| Embedding | continuous linear projection | not categorical-aware |
| Feasibility | continuous polytope | no category feasibility |
| Acquisition optimization | continuous `optimize_acqf` | no discrete enumeration |
| Candidate reconstruction | linear pseudoinverse | does not preserve categories |

Consequently, `ALEBOGP` and `ALEBOStrategy` remain continuous-only.

## PairwiseGP

`PairwiseGP` is not a high-dimensional embedding method, but it was left as
an investigation item in the repository-wide Mixed audit. Phase 10 keeps it
without a nominal Mixed wrapper. Pairwise preference inference includes
datapoint consolidation and a Laplace approximation; a Mixed variant requires
an explicit design and tests for categorical covariance together with
duplicate/consolidation semantics. It must not be inferred from the existence
of `MixedSingleTaskGP`.

## What a future Mixed ALEBO implementation requires

A future implementation may be added only as an end-to-end search-space
feature. A viable design is to partition raw dimensions into continuous and
categorical design variables, embed only the continuous subspace, keep
categorical choices outside the projection, and optimize acquisition over
continuous embedded coordinates conditional on discrete category assignments.

Before exposing a public Mixed ALEBO API it must provide:

- raw-space `cat_dims` normalization including negative indices;
- a continuous-only ALEBO embedding;
- explicit categorical values / feasibility;
- acquisition optimization that jointly handles the embedded continuous
  coordinates and categorical assignments;
- reconstruction that exactly preserves a valid raw categorical assignment;
- a surrogate whose covariance semantics match that search representation;
- q-batch behavior and state/dtype/device tests;
- end-to-end tests proving returned candidates are inside bounds and every
  categorical coordinate belongs to its allowed set.

Until those conditions are met, absence of `MixedALEBOGP` is an intentional
semantic boundary rather than a missing wrapper.
