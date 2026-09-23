# Additive / ensemble MAP-SAAS × multi-fidelity follow-up audit

## Scope

Phase 34 re-evaluates whether the existing `MapSaasMultiFidelityGP` contract is sufficient evidence
for public additive or ensemble MAP-SAAS multi-fidelity variants. It is not.

## Source-of-truth findings

The current wrappers delegate directly to BoTorch's `AdditiveMapSaasSingleTaskGP` and
`EnsembleMapSaasSingleTaskGP`. Their semantics are materially different from the design-only
Matern kernel used by `MapSaasMultiFidelityGP`.

### Additive MAP-SAAS

The additive model is not a thin change of prior. Its design covariance is an additive construction
with its own component semantics. A multi-fidelity version must define how the additive design
components compose with iteration/data fidelity kernels. Reusing `MapSaasMultiFidelityGP` and
renaming it would erase that distinction.

Required evidence before implementation:

1. additive components operate only on raw design dimensions;
2. fidelity dimensions are excluded from SAAS relevance parameters;
3. the complete covariance composes additive design structure with BoTorch-native fidelity kernels;
4. raw-space posterior, conditioning, fantasize, and native MF acquisition paths remain valid;
5. additive hyperparameter ownership is explicit and testable.

### Ensemble MAP-SAAS

The ensemble model has an additional boundary: its posterior represents an ensemble over MAP-SAAS
fits. A multi-fidelity implementation must preserve that ensemble dimension and sampling semantics
after fidelity covariance is introduced.

Required evidence before implementation:

1. each ensemble member uses the same declared fidelity structure;
2. fidelity parameters are not accidentally interpreted as ensemble SAAS design parameters;
3. posterior sampling preserves ensemble semantics;
4. acquisition compatibility is established from the resulting posterior type, not inherited from
   the non-ensemble MAP-SAAS model;
5. conditioning/fantasize semantics are validated independently.

## Decision

| Candidate | Phase 34 status | Reason |
| --- | --- | --- |
| Additive MAP-SAAS × MF | design/prototype required | additive covariance semantics are distinct |
| Ensemble MAP-SAAS × MF | research-gated behind additive/prototype work | ensemble posterior semantics add another boundary |
| `MapSaasMultiFidelityGP` | implemented reference | design-only SAAS + native MF covariance already validated |
| Fully Bayesian SAAS × MF | research-gated | requires a fidelity-aware probabilistic model |

No public model is added in this phase. This is intentional: the existing reference model is not
evidence that either specialized BoTorch MAP-SAAS family can be represented by a renamed or aliased
wrapper.

## Next implementation gate

The next code phase may prototype additive design covariance × native fidelity covariance, but it
must remain non-public until covariance ownership, raw-space posterior behavior, and native MF
acquisition execution are demonstrated. Ensemble MAP-SAAS × MF should not be implemented by
inheritance from that prototype unless ensemble posterior semantics are independently preserved.

## Phase 35 follow-up

The additive branch has now been reduced to a concrete covariance prototype contract. Current
public wrappers do not expose an additive design-kernel factory, so no implementation is promoted
by copying private BoTorch construction details. See
`additive_map_saas_multifidelity_prototype.md` for the required ownership and executable gates.
