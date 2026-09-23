# Fully Bayesian SAAS x Kronecker design audit

## Decision

Phase 9 classifies a fully Bayesian SAAS Kronecker model as **Hold**, not Implement.

The statistical target is meaningful: a block-design multi-task GP with SAAS shrinkage on the
input/data covariance and a separate task covariance. The current BoTorch fully Bayesian APIs,
however, do not provide a Kronecker block-design Pyro model or a compatible MCMC sample-loading
contract. Implementing one in robotorchan now would require ownership of a substantial custom
fully Bayesian model rather than a thin BoTorch-first extension.

Phase 10 must therefore not create a public `SaasFullyBayesianKroneckerMultiTaskGP` unless the
upstream contract changes or a later prototype demonstrates a small, maintainable integration.

## Desired statistical model

For aligned observations

- `train_X[..., n, d]`;
- `train_Y[..., n, m]`;

the latent covariance should remain

`K = K_data^SAAS(X, X') ⊗ K_task`.

SAAS shrinkage belongs only to the `d` input dimensions. Output tasks are not input dimensions
and must not receive SAAS lengthscales.

A suitable fully Bayesian model would sample at least:

- the SAAS global shrinkage parameter;
- data-kernel inverse lengthscales / lengthscales for the `d` data dimensions;
- data outputscale and observation-noise parameters as required by the chosen likelihood;
- task covariance parameters, including the requested task rank.

The posterior must integrate these samples without flattening task identity into `train_X`.

## Existing robotorchan / BoTorch paths

robotorchan's current fully Bayesian wrappers deliberately delegate inference to BoTorch:

- `SaasFullyBayesianSingleTaskGP` delegates to BoTorch's single-task SAAS Pyro model;
- `SaasFullyBayesianMultiTaskGP` delegates to BoTorch's long-format
  `SaasFullyBayesianMultiTaskGP` and `MultitaskSaasPyroModel`;
- Mixed variants only own categorical encoding and still delegate fully Bayesian inference.

The available multi-task SAAS contract is task-feature / long-format. It is not a block-design
Kronecker model. Reformatting `n × m` observations into `nm × 1` rows would change the model
semantics and is explicitly rejected.

## Pyro ownership boundary

A genuine SAAS Kronecker implementation would currently need robotorchan to own a custom Pyro
model that understands all of the following simultaneously:

1. block-design `n × m` observations;
2. SAAS priors over data dimensions only;
3. a sampled task covariance factor;
4. multitask likelihood/noise parameters;
5. MCMC batch dimensions after NUTS;
6. conversion of Pyro samples into GPyTorch modules used by the Kronecker posterior.

This is substantially more than configuring an upstream kernel. It creates a second fully
Bayesian implementation that must track BoTorch's internal sample naming and loading behavior.
That cost is not justified while the library's rule is BoTorch-first.

## MCMC sample dimensions

The required distinction is:

- MCMC sample dimension: posterior hyperparameter draws;
- observation dimension: `n`;
- output task dimension: `m`;
- candidate batch / q dimensions: introduced at posterior evaluation.

The MCMC dimension must never be interpreted as an output task dimension. Likewise, task outputs
must not be batched as independent SAAS models, because that would remove `K_task`.

Any future prototype must test these dimensions explicitly after sample loading and posterior
evaluation.

## Posterior sampling

A future implementation must prove that a posterior call at `X[..., q, d]` preserves the output
task axis and correctly mixes over MCMC hyperparameter samples using BoTorch's fully Bayesian
posterior contract.

Merely constructing a Kronecker kernel after fitting independent SAAS models is insufficient:
hyperparameter samples must correspond to one coherent joint block-design model.

## Acquisition compatibility

Before public registration, a future implementation must run at least:

- scalarized MC `qLogExpectedImprovement`;
- `qUpperConfidenceBound`;
- posterior sampling with the standard BoTorch sampler path;
- continuous `optimize_acqf`.

Compatibility cannot be inferred from the class name or registry metadata.

## Fantasize / lookahead boundary

Fantasize support is a separate gate. A fully Bayesian Kronecker model must preserve the MCMC
sample dimension when conditioning on fantasy observations. Current long-format fully Bayesian
support does not establish that this works for a custom block-design Pyro model.

Therefore `fantasize`, KG, and lookahead acquisitions must remain unsupported until demonstrated
by runtime tests. Phase 10 must not advertise them speculatively.

## Mixed SAAS x Kronecker

Mixed support is also deferred. If a future continuous SAAS Kronecker model becomes viable,
categorical treatment should follow the established raw-input contract:

- `cat_dims` refers only to columns of raw `train_X`;
- task identity remains the output axis;
- SAAS shrinkage applies to continuous data coordinates, not one-hot category indicators unless
  that prior is explicitly justified;
- categorical covariance remains separate from `K_task`.

Implementing Mixed first would compound unresolved Pyro and Kronecker ownership issues.

## Revisit triggers

Change the status from Hold to Prototype only when at least one of these becomes true:

1. BoTorch provides a block-design fully Bayesian Kronecker multi-task model;
2. BoTorch exposes a stable public Pyro/sample-loading interface sufficient to implement the model
   without copying private fully Bayesian internals;
3. robotorchan adopts an explicit policy to own custom fully Bayesian inference implementations,
   with maintenance and version-compatibility tests.

## Phase 10 action

Under the current source of truth, Phase 10 should record the Hold decision in the extension
matrix and registry/documentation audit surfaces. It should **not** add a runtime model merely to
satisfy the original phase numbering.

This is a deliberate implementation-cost decision, not a rejection of SAAS x Kronecker as a
statistical model.
