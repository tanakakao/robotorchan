# Classification and ordinal model audit

## Scope

Phase 39 reopens model development only for the classification / ordinal family.
The regression model matrix is not reopened. The purpose is to decide which
missing public model contracts are real gaps and which acquisition functions
must wait for those contracts.

## Current source-of-truth

The current public model registry contains regression, multi-task,
multi-fidelity, robust, high-dimensional, structured-output, preference, and
non-GP regression families. It does not expose a probabilistic binary GP
classifier or an ordinal GP model.

The acquisition package already has regression active-learning primitives such
as `PosteriorVariance`, `PosteriorStd`, `Straddle`,
`BoundaryVariance`, and `ExpectedPredictiveInformationGain`. Those operate
on posterior moments or regression level-set semantics. They must not be
relabelled as classification entropy or BALD.

No current public model provides the explicit class-probability and latent
posterior contract required to implement classification AL correctly.

## Required statistical contracts

### Binary GP classification

A public binary classifier needs two distinct prediction surfaces:

1. latent function posterior `p(f | D, x)`;
2. class probability `p(y=1 | D, x)` after the Bernoulli link.

The API must make that distinction explicit. Returning a latent Gaussian
posterior and calling its mean a probability is invalid.

The first implementation should support:

- raw `train_X` retention;
- binary `train_Y` validation;
- a Bernoulli likelihood with a documented link;
- variational inference, because the Bernoulli likelihood is non-Gaussian;
- latent posterior sampling;
- class probability prediction;
- a training objective / `make_mll` equivalent appropriate to variational
  classification;
- CPU double-precision smoke tests;
- finite gradients and optimizer smoke tests.

### Ordinal GP

Ordinal observations require ordered thresholds / cut-points and a categorical
probability vector whose ordering is part of the model contract. Treating
ordinal labels as continuous regression targets is not an ordinal model.

Required semantics include:

- integer ordered labels with contiguous class mapping;
- monotone cut-points;
- latent posterior;
- per-class probabilities that sum to one;
- explicit class ordering metadata;
- variational or other justified non-Gaussian inference;
- tests for finite probabilities and monotonic threshold ownership.

Ordinal support should follow binary classification rather than being built
first, because it needs the same latent/non-Gaussian infrastructure plus
threshold semantics.

## Acquisition dependencies

Classification acquisition functions should be introduced only after the model
contract exists.

### Predictive entropy

For binary probability `p`:

```text
H[y | x, D] = -p log p - (1-p) log(1-p)
```

This requires predictive class probabilities, not only latent variance.

### Margin uncertainty

For binary classification it can be expressed from class probability distance
to 0.5. Multi-class generalization requires the top-two probability margin.

### Probability variance

This name must be defined carefully. Variance of a Bernoulli predictive label
`p(1-p)` is aleatoric-plus-predictive uncertainty and is not the same object
as posterior variance of the latent GP.

### BALD

BALD requires mutual information between predictions and model uncertainty:

```text
I[y, theta | x, D]
  = H[y | x, D] - E_theta[H[y | x, theta]]
```

A single marginal predictive probability is insufficient. The implementation
needs posterior samples that preserve epistemic variation before the Bernoulli
link. This is why plain NGBoost predictive distributions and regression
posterior variance are not substitutes for BALD.

### Latent straddle

A latent-function level-set acquisition is possible for GP classification, but
its threshold must be defined on the latent link scale. It should not silently
mix latent zero with a probability threshold unless the link makes that
correspondence explicit.

## Mixed and multi-task boundary

Do not start with the cross product.

The implementation order should be:

1. binary single-task GP classifier;
2. classification AL on that validated contract;
3. mixed-input binary classifier if raw categorical semantics are required;
4. multi-class / multi-task classification only with an explicit output
   probability contract;
5. ordinal GP and ordinal-specific acquisitions.

This avoids duplicating an unstable base contract across Mixed and MultiTask
variants.

## BoTorch-first assessment

BoTorch's standard exact and variational GP model APIs are useful foundations,
but a Bernoulli classification likelihood changes the inference and prediction
contract. The implementation should reuse public BoTorch / GPyTorch
variational components wherever they express the required semantics.

A new robotorchan public classifier is justified only as a thin integration
layer around those public primitives: raw-data retention, unified training API,
probability prediction, capability metadata, and acquisition compatibility.

No compatibility aliases or regression-shaped fake classifier APIs should be
introduced.

## Phase 39 decision

The missing binary probabilistic GP classifier is a genuine **future model
capability**, not a regression correctness defect.

It is the next model implementation candidate because it unlocks a coherent
classification Active Learning stack: predictive entropy, margin uncertainty,
probability-space uncertainty, BALD, and latent straddle.

Ordinal GP remains a subsequent candidate after the binary latent/probability
contract is validated.

### Proposed next phase

Phase 40 should design the concrete binary variational GP classifier API before
implementation. It should resolve:

- class name and constructor;
- label shape and accepted values;
- latent `posterior` versus `predict_proba`;
- Bernoulli link;
- variational strategy / inducing-point ownership;
- training objective and `make_mll` contract;
- registry capability fields;
- exact tests required before any classification acquisition is exported.
