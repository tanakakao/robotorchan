# Phase 8: uncertain training-input GP feasibility

## Purpose

Phase 8 addresses uncertainty in the **training inputs themselves**. This is
different from candidate-time robust optimization, where a proposed control
setting is perturbed only while evaluating a decision.

The statistical problem is

```text
x_observed = x_true + delta
y = f(x_true) + epsilon
```

where the latent or uncertain `x_true` affects surrogate inference.

## Boundary with existing robust layers

Keep these cases separate:

- `GaussianPerturbation` and other scenario generators describe uncertainty
  when evaluating candidate decisions;
- environmental scenarios represent explicit uncontrollable factors `w`;
- heteroskedastic and Student-t models alter observation-noise assumptions;
- uncertain training inputs alter the surrogate because covariance must account
  for uncertainty in observed input locations.

Therefore Phase 8 may justify a dedicated surrogate implementation. It must not
be implemented by simply jittering `train_X` once before fitting.

## Initial scope

Start with scalar-output continuous inputs and caller-supplied per-observation
input uncertainty. The first supported representation should be diagonal
Gaussian uncertainty:

```text
train_X_std.shape == train_X.shape
```

with nonnegative standard deviations.

A later phase may consider full covariance per observation. Mixed categorical
input uncertainty, multi-task cross-products, and reduced-space variants are
out of scope until the continuous base model is validated.

## Candidate inference approaches

Phase 8 implementation must audit current BoTorch / GPyTorch capabilities before
choosing an API. Acceptable approaches include:

1. an uncertainty-aware kernel that analytically integrates Gaussian input
   uncertainty when the kernel permits it;
2. Monte Carlo marginalization over latent training inputs during variational
   inference;
3. another maintained upstream primitive with equivalent semantics.

A one-time augmented data set or random jitter is not sufficient because it
does not preserve the uncertainty model during inference.

## API target

The public surface should remain explicit about the uncertainty source:

```python
model = UncertainInputSingleTaskGP(
    train_X,
    train_Y,
    train_X_std=train_X_std,
)
```

The model must retain raw `train_X`, `train_Y`, and the supplied uncertainty
metadata. If inference is non-exact, it must expose `training_loss()` and must
not claim exact MLL support.

## Validation gates

Before implementation is accepted:

1. zero input uncertainty recovers or closely approaches the corresponding
   ordinary GP behavior;
2. increasing input uncertainty changes posterior uncertainty in the expected
   direction on a controlled synthetic case;
3. uncertainty participates in training rather than being discarded after
   preprocessing;
4. gradients are finite for every learned component;
5. posterior batch/q shapes are BoTorch-compatible;
6. dtype/device migration and state serialization are covered;
7. MC acquisition compatibility is tested;
8. candidate-time perturbation remains independently composable;
9. no `RobustPCAGP`-style cross-product classes are introduced.

## Phase 8 sequence

1. Audit upstream BoTorch / GPyTorch support and repository kernel conventions.
2. Select analytic-kernel or variational marginalization inference based on that
   audit.
3. Implement the smallest scalar continuous model.
4. Add recovery, uncertainty-sensitivity, lifecycle, acquisition, and
   composition tests.
5. Only after the base model is green, evaluate full covariance and other model
   families.
