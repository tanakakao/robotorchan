# Objective and PosteriorTransform

This guide defines the runtime responsibility boundary for objectives, posterior transforms,
constraints, and robust scenario aggregation.

## Canonical pipeline

```text
X
-> candidate feasibility
-> model.posterior(X)
-> PosteriorTransform when required
-> posterior sampling when required
-> MC Objective when required
-> outcome feasibility / acquisition utility
-> acquisition optimization
```

For decision-time input perturbation, scenarios are introduced before posterior evaluation:

```text
X -> perturbation scenarios -> posterior -> sampling -> objective -> scenario risk aggregation
```

Input perturbation is not a generic objective responsibility.

## PosteriorTransform

A `PosteriorTransform` changes the posterior distribution before analytic or Monte Carlo
acquisition evaluation. Use BoTorch's native transform objects directly when the model posterior
supports them.

Affine scalarization should normally use `ScalarizedPosteriorTransform`. For
`Z = offset + w^T Y`, a compatible Gaussian posterior must preserve both transformed mean and
cross-output covariance:

```text
E[Z] = offset + w^T E[Y]
Var[Z] = w^T Cov[Y] w
```

A model must not manufacture this covariance from marginal variances alone. Specialized posterior
types may therefore explicitly reject transforms whose required distribution structure they do
not expose.

## MC Objective

An `MCAcquisitionObjective` transforms posterior samples rather than the posterior distribution.
Its conceptual shape contract is:

```text
sample_shape x batch_shape x q x m
-> sample_shape x batch_shape x q
```

Use native BoTorch objectives where they express the required semantics. `GenericMCObjective`
is the ordinary extension point for custom nonlinear sample objectives.

Posterior transforms and MC objectives are not interchangeable abstractions.

## Multi-objective acquisition

Hypervolume acquisitions such as qLogEHVI and qLogNEHVI retain the objective vector required for
hypervolume calculations. Scalarization is therefore not the default multi-objective contract.
Scalarization-based strategies such as qLogNParEGO are separate workflows.

## Constraints

Outcome constraints and candidate constraints are distinct.

Outcome constraints represent unknown feasibility through modeled outcomes and are composed with
compatible acquisitions. Candidate constraints are known functions of `X` and belong to
`robotorchan.optim`.

Do not combine these concepts into a shared constraints abstraction.

## Robust scenario aggregation

Decision-time perturbation creates candidate scenarios before posterior evaluation. Posterior
sampling and scenario sampling remain separate axes. Risk aggregation reduces the scenario axis
after modeled outcomes have been evaluated.

The classes in `robotorchan.objectives.risk` are scenario aggregators; they must not be presented
as drop-in `MCAcquisitionObjective` instances unless they actually implement that native
contract. When BoTorch provides a risk-measure objective with the required semantics, prefer the
native path rather than adding a parallel wrapper.

## Model-specific transform support

Uniform `posterior_transform=` syntax does not imply universal transform compatibility.

- Standard and mixed Gaussian exact-GP paths normally delegate native transforms to BoTorch.
- Reduced-input models apply input reduction before posterior construction; transforms still act
  on the public posterior.
- Output-reduced models must preserve original-output semantics before claiming transform support.
- Empirical ensemble and stochastic posterior types may reject transforms requiring Gaussian
  distribution structure.
- Kronecker or other specialized BoTorch posterior paths follow their upstream transform
  limitations unless robotorchan has a mathematically justified implementation.

Unsupported combinations should fail explicitly rather than silently changing semantics.

## Extension policy

Add a robotorchan-specific objective or transform only when a demonstrated semantic gap remains
after considering native BoTorch APIs. New abstractions must define tensor axes, gradient
behavior, composition with acquisitions, and model/posterior compatibility through executable
tests.
