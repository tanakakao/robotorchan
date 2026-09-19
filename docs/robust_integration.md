# Robust composition integration audit

## Scope

This phase checks whether the uncertainty scenarios from Phase 3 and the risk
aggregations from Phase 4 can be composed with the existing surrogate families
without creating cross-product model classes.

The robust evaluation path is:

```text
raw candidate X
    |
uncertainty / environmental scenarios
    |
surrogate posterior at scenario points
    |
scenario outcomes
    |
risk or SN aggregation
```

## Integration matrix

| Surrogate family | Scenario composition | Decision aggregation | Boundary |
|---|---|---|---|
| SingleTaskGP | Direct | Direct | Baseline supported path |
| PCA / PLS / random projection GP | Raw-space first | Direct | Perturb before reduction |
| Autoencoder / VAE GP | Raw-space first | Direct | Perturb before learned encoder |
| Joint encoder / hybrid / joint VAE | Raw-space first | Direct | Keep physical uncertainty outside latent space |
| SAAS single-task | Direct | Direct | Sparse original-coordinate model; no robust subclass |
| Reduced multi-task | Conditional | Conditional | Preserve task feature / task structure |
| Mixed reduced GP | Conditional | Direct | Never jitter categorical dimensions implicitly |
| ALEBO | Strategy-level audit required | Potentially composable | Do not treat ALEBO as a generic reducer |

## Reduced models

Reduced models already accept original-space candidates at their public
posterior surface and apply their reducer internally. Therefore an uncertainty
scenario must be generated before the model call:

```python
scenario_X = perturbation.sample(X, n_w=64)
posterior = model.posterior(scenario_X)
```

This is the desired semantics for PCA, PLS, random projection, autoencoder,
VAE, supervised neural reduction, and joint neural reduction. It keeps
tolerances, process errors, and environmental factors interpretable in physical
coordinates.

No `RobustPCAGP`, `RobustPLSGP`, `RobustAutoEncoderGP`, or equivalent
cross-product class is justified by this composition.

## SAAS

SAAS models operate in the original coordinate system and use sparsity priors
rather than a learned lower-dimensional candidate space. Scenario generation
therefore remains outside the surrogate. Risk aggregation can consume scenario
outcomes without a dedicated `RobustSAASGP`.

Fully Bayesian posterior sampling cost must be considered when `n_w` is
large, but that is a computational concern rather than a new model contract.

## Mixed models

For mixed models, continuous input perturbation must name the dimensions that
may change. Categorical dimensions must remain untouched unless categorical
uncertainty is modeled explicitly through empirical environmental scenarios.

This means:

- Gaussian and uniform jitter require explicit safe dimensions in mixed use;
- empirical scenarios may replace categorical environmental factors explicitly;
- a categorical factor such as material lot or equipment is not approximated by
  adding floating-point noise to its encoded value.

## Multi-task models

Long-format multi-task models contain task identity in the input representation.
A generic perturbation must not alter the task feature. The scenario generator
can be used only when its selected dimensions exclude task identity.

Kronecker multi-task models keep task structure outside the ordinary feature
axis, so feature perturbations are structurally cleaner. Risk aggregation still
needs an explicit output / task policy before a generic acquisition wrapper is
declared fully supported.

No robust multi-task cross-product model is introduced in this phase.

## Environmental factors

Explicit environmental variables use the same raw-space scenario contract but
have different semantics from actuator error:

```text
control factors x + environmental factors w -> f(x, w)
```

`EmpiricalScenarios` can replace environmental dimensions with observed
scenario rows while leaving control factors unchanged. This supports material
lot, equipment, ambient condition, and similar quality-engineering noise
factors.

The resulting scenario outcomes may be aggregated by expectation,
mean-variance, worst case, VaR, CVaR, or SN ratio.

## ALEBO boundary

ALEBO remains a special case. It is an embedded search strategy with its own
projection and optimizer geometry rather than a generic reducer wrapper.

A raw-space environmental or perturbation scenario may be mathematically
meaningful, but robust candidate generation must respect ALEBO's embedded search
constraints. Phase 5 therefore does not declare generic ALEBO robust composition
supported and does not add `RobustALEBOGP`.

A later strategy-level integration may explicitly map uncertainty through the
ALEBO search geometry.

## Phase 5 conclusion

The composition architecture is sufficient for ordinary single-task, reduced,
neural-reduced, and SAAS surrogate families. Mixed and multi-task use requires
dimension / task protection rather than new surrogate classes. ALEBO requires a
strategy-level integration.

The next surrogate-level robust work should therefore remain focused on cases
where the statistical model itself changes: heteroskedastic observation noise,
heavy-tailed likelihoods, and uncertain training inputs.
