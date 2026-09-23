# Robust relevance pursuit × multi-fidelity audit

Phase 13 audits whether robust relevance pursuit (RRP) can be combined with multi-fidelity
regression without inventing new statistical semantics.

## Decision

**Do not implement a public `RobustRelevancePursuitMultiFidelityGP` in this phase.**

The combination is plausible, but the current robotorchan and BoTorch contracts do not provide a
thin, semantics-preserving composition that has been demonstrated by executable evidence.

## Existing contracts

robotorchan currently uses two different upstream-backed contracts:

- `RobustRelevancePursuitSingleTaskGP` wraps BoTorch's dedicated
  `RobustRelevancePursuitSingleTaskGP`;
- `SingleTaskMultiFidelityGP` wraps BoTorch's dedicated multi-fidelity GP and its fidelity
  covariance construction.

The RRP wrapper accepts a replacement `covar_module`, but that alone is not evidence that
BoTorch's multi-fidelity covariance can be reconstructed and injected without changing the
upstream model's intended initialization, likelihood, support dimension, fitting workflow, or
future compatibility.

## Why multiple inheritance is rejected

A class such as

```python
class RobustRelevancePursuitMultiFidelityGP(
    SingleTaskMultiFidelityGP,
    RobustRelevancePursuitMixin,
):
    ...
```

is not accepted merely because both parents are exact single-task GPs.

The RRP mixin owns sparse outlier-noise behavior and expects a compatible base likelihood.
The multi-fidelity model owns fidelity-aware response covariance. Cooperative initialization,
likelihood replacement, relevance-pursuit support bookkeeping, and model conversion / fitting
behavior must all remain valid.

The Phase 6 Kronecker investigation already showed that neighboring RRP support does not imply
likelihood compatibility.

## Acceptable future implementation paths

### Preferred: upstream composition point

If BoTorch exposes a supported way to combine its RRP likelihood / mixin with the native
multi-fidelity covariance while preserving both public contracts, robotorchan should add only a
thin wrapper and runtime tests.

### Acceptable: robotorchan composition with upstream components

A robotorchan-owned model is acceptable only if all of the following are explicit:

1. the fidelity covariance is built from the same BoTorch components and parameterization used by
   `SingleTaskMultiFidelityGP`;
2. RRP sparse-outlier support has the same observation-level meaning as the BoTorch RRP model;
3. no fidelity coordinate is treated as an outlier indicator or ordinary categorical variable;
4. the likelihood contract is verified directly rather than inferred from inheritance;
5. fitting, posterior sampling, and acquisition evaluation have executable tests;
6. the reason a thin BoTorch wrapper is impossible is documented.

This would be a deliberate composition model, not an alias or compatibility shim.

## Required executable evidence

Before public registration, a future implementation must demonstrate:

- raw training-data preservation;
- negative and multiple fidelity-index handling;
- fidelity covariance remains active after RRP initialization;
- sparse-outlier support dimension matches observations;
- RRP fitting / support update path executes;
- finite posterior mean and variance;
- finite posterior samples;
- representative MC acquisition;
- representative multi-fidelity acquisition or candidate-generation path;
- no regression in the ordinary RRP and multi-fidelity wrappers.

## Phase 13 classification

| Combination | Status |
| --- | --- |
| ReplicateNoise × MultiFidelity | implemented |
| Heteroskedastic × MultiFidelity | implemented |
| RRP × MultiFidelity | research-gated |
| Student-t × MultiFidelity | deferred |
| Contaminated × MultiFidelity | deferred |

RRP × MultiFidelity is therefore **not a missing correctness fix**. It is a genuine future model
candidate whose implementation requires a verified composition contract.

## Next regression priority

With the two statistically clear robust multi-fidelity paths implemented, development should move
to the next planned regression capability rather than forcing RRP support. Revisit this model when
upstream BoTorch provides a clearer composition point or when a concrete use case justifies the
additional likelihood/covariance integration work.
