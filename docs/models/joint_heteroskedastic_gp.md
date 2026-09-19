# Phase 6C: Joint variational heteroskedastic GP design

## Goal

Phase 6A/6B established the practical iterative two-GP model. Phase 6C defines
the statistically stronger joint variational model before adding a second
public implementation.

For latent response and log-noise processes,

```text
f(x) ~ GP
g(x) ~ GP
y | f, g ~ Normal(f(x), exp(g(x)))
```

the inference objective must optimize both latent processes together rather
than turning fitted residuals into fixed targets.

## Required architecture

The joint model should use two variational latent GPs, `f` and `g`, with
separate inducing points and variational distributions. Training minimizes a
Monte Carlo estimate of the negative ELBO:

```text
-E_q(f,g)[log p(y | f, g)]
+ beta_f KL[q(f) || p(f)]
+ beta_g KL[q(g) || p(g)]
```

The observation log likelihood is evaluated directly with
`variance = exp(g).clamp_min(noise_floor)`.

## Public API target

```python
model = JointHeteroskedasticSingleTaskGP(
    train_X,
    train_Y,
    num_inducing=32,
    noise_floor=1e-6,
)

loss = model.training_loss()
posterior = model.posterior(test_X)
noise = model.predicted_noise(test_X)
```

The class must not pretend to support `make_mll()`: the joint ELBO is not an
ExactMarginalLogLikelihood. It should follow robotorchan's existing
variational-model training contract.

## Compatibility requirements

Before the class becomes public, tests must prove:

- gradients reach response and noise variational parameters;
- `training_loss()` is finite;
- `posterior(X)` satisfies the BoTorch scalar-model contract;
- `predicted_noise(X)` is positive and supports batch/q dimensions;
- dtype/device migration invalidates or avoids stale caches;
- state_dict round trips both latent processes;
- MC acquisitions accept the response posterior;
- scenario generation and risk aggregation remain external composition.

## Boundaries

Do not create Mixed/Reduced/MultiTask variants in Phase 6C. Those combinations
are considered only after the base joint model is numerically validated.

Do not replace the iterative `HeteroskedasticSingleTaskGP` silently. The
iterative and joint models have different inference contracts. If a later
decision removes the iterative model, perform a complete API migration rather
than adding aliases or deprecated wrappers.

## Implementation gate

Implementation should reuse the repository's variational training conventions
rather than introducing a one-off optimizer API. The next subphase first
audits `SingleTaskVariationalGP` and its training tests, then implements the
two-latent ELBO against that contract.
