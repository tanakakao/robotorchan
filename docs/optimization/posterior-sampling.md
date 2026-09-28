# Posterior sampling

Robotorchan follows a BoTorch-first sampling policy. Sampling behavior is selected from the
posterior contract; model family names do not determine the sampler.

## Posterior-to-sampler mapping

| Posterior semantics | Preferred BoTorch sampler | Representative models |
| --- | --- | --- |
| Gaussian | `SobolQMCNormalSampler` | SingleTask, MultiTask, Kronecker, mixed, multi-fidelity, reduced, robust Gaussian, NGBoost adapter |
| Empirical ensemble | `IndexSampler` | RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting |
| Stochastic / empirical trajectory | `StochasticSampler` | DeepGP families |
| Actual `PosteriorList` | `ListSampler` | only when the returned posterior is a `PosteriorList` |

`ModelListGP` must not be mapped to `ListSampler` by model name. For homogeneous Gaussian
submodels, BoTorch can return a joint Gaussian multi-output posterior, so normal samplers are
the correct path.

`make_model_sampler` is an integration helper driven by model capability metadata. It is not
part of the top-level acquisition public API and does not replace direct BoTorch sampler use.

## Joint sampling

Multi-output, MultiTask, and Kronecker posteriors are sampled jointly. Do not split outputs or
tasks into independent samplers when the posterior already represents their covariance.

For DeepGP, `DeepGPPosterior` stores complete predictive trajectories. `StochasticSampler`
delegates to `posterior.rsample()`, preserving trajectory-level resampling without pretending
that the empirical posterior is Gaussian or QMC-normal.

## Shape and reproducibility

Sampling follows the BoTorch shape contract:

```text
sample_shape x batch_shape x q x output_shape
```

Gaussian normal samplers support base-sample reuse for common random numbers. Empirical
ensemble and stochastic posteriors follow their own BoTorch sampler contracts; identical
behavior must not be inferred from the word "sampling" alone.

## Sampling and candidate gradients

Sampler compatibility does not imply gradient-based optimization compatibility.

Gaussian GP, mixed continuous dimensions, MultiTask, Kronecker, reduced-space GP, robust
Gaussian GP, and current DeepGP trajectories preserve candidate gradients in their tested
paths. Scikit-learn tree ensembles and the NGBoost adapter cross a NumPy / external-estimator
boundary, so their posterior samples are not differentiable with respect to candidate inputs.
Use an optimization strategy compatible with the model boundary.

## MC acquisition sampling

MC acquisition sampling integrates over uncertain model outcomes:

```text
model -> posterior -> MCSampler -> objective / outcome constraint -> acquisition
```

Outcome constraints are evaluated from sampled model outputs. They are separate from known
candidate/input-space constraints, which belong to acquisition optimization.

## Input perturbation is a separate uncertainty layer

Candidate perturbation is not a responsibility of the posterior sampler:

```text
candidate X
-> input perturbation scenarios
-> model posterior
-> posterior sampling
-> risk aggregation across scenarios
```

The scenario axis and posterior MC sample axis remain distinct. Risk measures such as
Expectation, MeanVariance, and CVaR reduce the scenario axis after posterior sampling.

## Thompson sampling is candidate generation

Finite-pool Thompson sampling is also distinct from MC integration. Robotorchan delegates to
BoTorch `MaxPosteriorSampling` through `select_thompson_candidates`:

```text
model -> posterior over choices -> MaxPosteriorSampling -> selected candidates
```

This path works with compatible Gaussian and empirical ensemble posteriors. Multi-output
posterior sampling requires an explicit scalarizing objective.

## Custom sampler policy

A robotorchan-specific sampler should be added only when a current posterior cannot be sampled
correctly by BoTorch's existing sampler ecosystem and the missing statistical semantics are
well defined and testable.

The current audited model set does not require a custom sampler. Gaussian, empirical ensemble,
DeepGP stochastic trajectory, and finite-pool Thompson workflows are covered by BoTorch-native
sampling primitives.
