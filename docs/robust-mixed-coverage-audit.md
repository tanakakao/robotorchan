# Robust × Mixed coverage closeout

## Purpose

This document closes the Robust × Mixed implementation sequence started by the
Phase 16 audit. The matrix reflects the current implementation on `main`, not
the original gap analysis.

A mixed surrogate is public only when categorical design variables change the
surrogate covariance or inference. Candidate perturbation, environmental
scenarios, and risk aggregation remain compositional layers.

## Final coverage matrix

| Robust concern | Continuous model | Mixed status | Implementation decision |
| --- | --- | --- | --- |
| sparse relevance-pursuit outliers | `RobustRelevancePursuitSingleTaskGP` | implemented | `MixedRobustRelevancePursuitSingleTaskGP` injects native mixed covariance while reusing relevance-pursuit inference. |
| iterative heteroskedastic noise | `HeteroskedasticSingleTaskGP` | implemented | `MixedHeteroskedasticSingleTaskGP` uses mixed response and mixed log-noise GPs and exposes fitted-state guarded noise diagnostics. |
| joint heteroskedastic noise | `JointHeteroskedasticSingleTaskGP` | implemented | `MixedJointHeteroskedasticSingleTaskGP` uses shared mixed variational latent-GP infrastructure for response and noise. |
| Student-t residuals | `StudentTSingleTaskGP` | implemented | `MixedStudentTSingleTaskGP` reuses the Student-t likelihood/training objective with a mixed variational latent response GP. |
| contamination-mixture residuals | `ContaminatedSingleTaskGP` | implemented | `MixedContaminatedSingleTaskGP` reuses contamination inference with a mixed variational latent response GP. |
| replicate-derived observation noise | `ReplicateNoiseSingleTaskGP` | implemented composition | `MixedReplicateNoiseSingleTaskGP` reuses the same replicate aggregation helper and feeds fixed mean variances to a mixed exact GP. |
| nonstationary latent smoothness | `NonstationarySingleTaskGP` | implemented | `MixedNonstationarySingleTaskGP` applies the Gibbs covariance only to continuous dimensions and composes native categorical covariance. |
| Gaussian-uncertain continuous training inputs | `UncertainInputSingleTaskGP` | implemented | `MixedUncertainInputSingleTaskGP` analytically marginalizes Gaussian uncertainty over continuous coordinates and multiplies by deterministic categorical covariance. |
| uncertain categorical training inputs | `UncertainCategoricalSingleTaskGP` | categorical-specific | No ordinary Mixed wrapper. Probability-valued category uncertainty is already modeled directly; additional deterministic categorical columns require a separate concrete statistical design. |

## Shared infrastructure and invariants

The mixed implementations intentionally reuse existing statistical algorithms:

- `make_mixed_covar_module` supplies native mixed covariance where the ordinary
  mixed kernel is appropriate.
- `MixedSingleTaskVariationalGP` is the shared variational latent-GP path for
  robust likelihood models.
- replicate aggregation is shared between continuous and mixed wrappers.
- Gaussian uncertain-input validation and expected-RBF mathematics are shared;
  categorical integer codes never enter Gaussian perturbation or continuous
  distance calculations.
- `MixedSingleTaskGP` retains normalized categorical-dimension metadata for
  downstream model composition.

The following invariants remain mandatory:

1. nominal category codes are never jittered;
2. nominal category codes are never interpreted through Euclidean/ordinal
   distance unless a model explicitly defines an ordinal variable;
3. no compatibility aliases, deprecated wrappers, or legacy API shims are
   introduced;
4. no Robust × reduced-model or Robust × ALEBO class multiplication is added
   when scenario/risk composition already represents the concern;
5. public exact models expose the standard MLL contract, while variational
   robust-likelihood models explicitly declare that exact MLL is unsupported.

## Phase 22 hardening result

The final audit found one semantic gap left by the earlier mixed
heteroskedastic MRO repair: the mixed iterative model no longer inherited
`noise_posterior()` and `predicted_noise()`. Phase 22 restored those methods
and added a real one-iteration fit test. That test also exposed missing
`cat_dims` retention in the base `MixedSingleTaskGP` wrapper, which is now
part of the base mixed-model contract.

This demonstrates why closeout tests cover behavior, not only construction and
posterior shape.

## Deliberately compositional robustness

These concerns remain outside surrogate cross-product classes:

- candidate-time input perturbation;
- environmental/scenario factors;
- expectation, mean-variance, worst-case, VaR, CVaR, and SN aggregation;
- reduced-coordinate models when robustness is defined in raw design space;
- ALEBO strategy-level robustness.

The intended architecture remains:

```text
surrogate model
    × uncertainty/scenario generator
    × risk aggregation
    -> acquisition
```

## Closeout

The Robust × Mixed roadmap is complete for the statistical concerns identified
by the robust-surrogate audit. No major missing mixed surrogate remains in that
scope.

Future mixed robust models should be added only when a new statistical
assumption changes the likelihood, covariance, posterior, or inference in a way
that cannot be represented by the current surrogate plus scenario/risk
composition.
