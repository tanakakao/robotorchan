# Heteroskedastic GP

## Implementation

robotorchan provides an iterative two-GP heteroskedastic surrogate. The mean
process models the response while a second GP models log residual variance:

```text
y = f(x) + epsilon(x)
epsilon(x) ~ N(0, sigma(x)^2)
log sigma(x)^2 ~ GP
```

`HeteroskedasticSingleTaskGP.fit_heteroskedastic()` alternates between fitting
the response GP, fitting the log-noise GP from squared residuals, and updating
the response likelihood with the predicted input-dependent variance.

This is intentionally distinct from supplying known `train_Yvar`: the latter
is fixed-noise modeling, while this model learns a noise surface that can be
queried at unseen inputs.

## API

```python
model = HeteroskedasticSingleTaskGP(train_X, train_Y, noise_floor=1e-6)
model.fit_heteroskedastic(iterations=3)

response = model.posterior(test_X)
noise_variance = model.predicted_noise(test_X)
log_noise_posterior = model.noise_posterior(test_X)
```

The current implementation is the practical iterative two-process version. A
future joint variational implementation may replace it only through a complete
API migration if it provides a materially better inference contract; no
compatibility wrapper should be retained.

Mixed, reduced, and multi-task cross-products remain deferred until this base
model is validated by CI and robust benchmarks.


## BoTorch integration

Phase 6B validates the iterative model as a BoTorch surrogate rather than only
as a standalone regression model. Integration coverage includes MC acquisition,
batched noise prediction, raw-space scenario generation followed by risk
aggregation, and dtype migration of the registered noise GP submodule.

The noise GP is a registered PyTorch submodule after fitting, so normal model
`.to(...)`, `.float()`, and `.double()` operations migrate both processes.
The response posterior remains the standard BoTorch posterior used by
acquisition functions; `predicted_noise` is an additional diagnostic and
robust-design signal.
