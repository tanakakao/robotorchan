# Robust modeling support audit

## Scope

This audit records the robust-modeling surface on current `main` before new
robust functionality is implemented. It separates surrogate robustness from
input perturbation and decision-risk handling so that later phases do not create
unnecessary cross-product model wrappers.

## Current implementation

### Robust surrogate models

The current robust surrogate family contains:

- `RobustRelevancePursuitSingleTaskGP`
- `MixedRobustRelevancePursuitSingleTaskGP`

Both live in `robotorchan.models.robust` and are exported from
`robotorchan.models`. The continuous wrapper follows the standard exact-GP
contract, including constructor-level raw-data snapshots and `make_mll()`.
The mixed wrapper reuses the repository's canonical mixed covariance helpers and
stores normalized `cat_dims`.

The existing robust model addresses sparse gross observation outliers through
relevance pursuit. It is not a general input-uncertainty or risk-aware BO layer.

### Input perturbation and decision risk

No robotorchan-owned public abstraction for input perturbations, VaR, CVaR,
worst-case aggregation, or other decision-risk measures is currently exposed
from `robotorchan.acquisition` or `robotorchan.objectives`.

Therefore later phases should add these as composable BO-layer capabilities
rather than encoding them into every surrogate class.

## Robustness taxonomy

Future work should keep four concerns separate.

| Concern | Meaning | Current support |
|---|---|---|
| Observation robustness | Outliers or non-Gaussian / input-dependent observation noise | Relevance pursuit only |
| Input robustness | Uncertain realized control input, such as `x + delta` | No robotorchan public layer |
| Environmental robustness | Explicit uncontrollable noise factors `w`, such as humidity, lot, or equipment | No robotorchan public layer |
| Decision robustness | Optimize a risk functional over uncertain outcomes or environments | No robotorchan public layer |

This distinction is architectural. A model name containing `Robust` must not
be used as a catch-all for all four concerns.

## Control factors and environmental noise factors

For manufacturing and materials applications, robust optimization should support
the quality-engineering distinction between controllable factors `x` and
uncontrollable or scenario factors `w`:

```text
y = f(x, w) + observation noise
```

The optimizer chooses `x`; `w` represents operating environment, raw-material
lot, equipment, ambient conditions, or other factors whose realized value is
not the design decision. This differs from perturbing a control setting itself.

Environmental robustness should marginalize or aggregate over `w` using a
decision-risk measure. Candidate measures include expectation, mean-variance,
worst case, VaR, CVaR, and quality-engineering signal-to-noise ratios.

SN ratio support should be a risk / aggregation capability, not a dedicated GP
class. Later design must allow the characteristic type to be explicit, such as
larger-is-better, smaller-is-better, or nominal-is-best, rather than assuming a
single SN formula.

Environmental scenarios may be continuous, categorical, empirical, or
correlated. Categorical noise factors such as material lot or equipment must
not be represented by accidental continuous jitter.

## High-dimensional integration audit

The high-dimensional model family is now broad enough that robust support must
prefer composition over model-name cross products.

### Reduced-input models

PCA, PLS, random projection, autoencoder, VAE, supervised neural reduction, and
joint neural models should not receive nominal `Robust*` subclasses merely to
apply an input perturbation or a risk measure.

For physical or process uncertainty, perturbations should normally be defined in
the original design space before reduction. This preserves the interpretation
of tolerances and measurement / actuation uncertainty. A latent-space
perturbation is a different modeling assumption and must be explicit if added.

### Multi-task reduced models

Long-format and Kronecker reduced multi-task models should consume the same
composable perturbation / risk layer where their posterior shape is compatible.
Do not introduce names such as `RobustPCAMultiTaskGP` solely for composition.

### Mixed reduced models

Mixed reduced models preserve categorical design variables outside the reducer.
A future perturbation layer must therefore distinguish perturbable continuous
dimensions from categorical and structural dimensions. Categorical perturbation
must be an explicit model of uncertainty, not an accidental continuous jitter.

### SAAS

SAAS remains sparse modeling in the original coordinate system rather than a
dimensionality-reduction transform. Robust decision layers should compose with
SAAS where BoTorch posterior semantics permit it. Do not add `RobustSAAS*`
wrappers without a distinct surrogate model.

### ALEBO

ALEBO is an end-to-end embedded search strategy, not a generic reducer. Robust
ALEBO requires a separate strategy-level feasibility audit. Do not create a
nominal `RobustALEBOGP` by mechanically combining names.

## Candidate surrogate gaps

The following surrogate capabilities remain materially distinct from the
existing relevance-pursuit model and are candidates for later phases:

1. Heteroskedastic observation modeling for input-dependent noise. Phase 6 found no maintained\n   BoTorch 0.18.1 exact-GP wrapper suitable for a truthful thin wrapper; see\n   `docs/models/heteroskedastic_gp_feasibility.md`.
2. Heavy-tailed / Student-t observation modeling.
3. Uncertain-input GP modeling when training inputs themselves are uncertain.

These candidates require separate feasibility checks against current
BoTorch / GPyTorch inference semantics before public classes are added.

## Composition policy

Prefer this architecture:

```text
surrogate model
    |
input perturbation or environmental scenarios
    |
risk measure / SN aggregation
    |
acquisition function and optimizer
```

A dedicated cross-product model is justified only when robustness changes the
surrogate likelihood, kernel, posterior, or inference procedure itself.

Consequently:

- relevance pursuit is a surrogate model;
- heteroskedastic and heavy-tailed likelihoods are surrogate concerns;
- uncertain training inputs may require a surrogate-level model;
- candidate-time perturbation is not a new surrogate model;
- explicit environmental factors should normally enter the surrogate as `f(x, w)`;
- environmental marginalization / scenario aggregation is not a new surrogate model;
- VaR / CVaR / worst-case / SN aggregation is not a new surrogate model;
- high-dimensional, mixed, multi-task, and robust names must not be multiplied
  unless the mathematics requires a dedicated implementation.

## Planned follow-up

Phase 2 should turn this audit into explicit design rules and decide the exact
package boundary for perturbation and risk abstractions before implementation.
Phase 3 can then implement the first perturbation primitives against that
contract.

The audit intentionally adds no compatibility alias, deprecated wrapper, or
placeholder public API.


## Phase 6C — joint heteroskedastic inference

The iterative model remains the practical baseline. The next heteroskedastic
surrogate is a two-latent variational model with jointly optimized response and
log-noise processes. See `docs/models/joint_heteroskedastic_gp.md` for the
inference objective, public API target, compatibility gates, and cross-product
boundaries.
