# Classification structured-family audit

Phase 30 audits regression structured-model roles against classification semantics. This is
development guidance, not a runtime model-selection API.

| Regression family | Decision | Classification rationale |
|---|---|---|
| OrthogonalAdditiveGP | DEFER | Additive latent structure is meaningful, but requires a Bernoulli variational model rather than the exact Gaussian regression class. |
| SACGP / LCEAGP | DEFER | Context decomposition is meaningful on a latent classifier, but requires classification-native variational inference. |
| LCEMGP / HeterogeneousMTGP | DEFER | Task/context covariance can be reused only through the multitask classification posterior contract. |
| HierarchicalConditionalKernelGP | DEFER | Hierarchical input covariance is meaningful, but should be implemented as a variational classifier preserving structural parent dimensions. |
| HigherOrderGP | UNSUPPORTED | Tensor-valued Gaussian responses are not binary or multiclass class-probability outputs. |
| LatentKroneckerGP | UNSUPPORTED | Gaussian responses over an explicit output-coordinate axis are not classification-label posteriors. |

## Decision rules

`DEFER` means that the structural idea is meaningful for classification, but the current regression
class must not be wrapped or inherited merely for name parity. A future implementation must attach
the relevant covariance or decomposition to the classification-native likelihood, variational
inference, posterior, sampling, calibration, and acquisition contracts.

`UNSUPPORTED` means direct family parity is semantically incorrect. HigherOrderGP and
LatentKroneckerGP model structured continuous Gaussian outputs. Changing their likelihood name does
not turn those output semantics into class probabilities.

This audit deliberately creates no runtime `structured_audit` module. Development audits remain
documentation artifacts; runtime capability metadata is reserved for capabilities that actually
exist in the library.
