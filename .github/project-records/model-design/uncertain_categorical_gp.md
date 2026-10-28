# Phase 14: uncertain categorical training inputs

## Decision

Uncertain categorical training inputs are statistically defensible when the
uncertainty is represented explicitly as a probability distribution over a
finite category set. They must not be represented by jittering integer category
codes.

Phase 14 therefore models each uncertain categorical observation by a
probability vector on the simplex.

## Problem

For an observation with continuous features `x` and categorical variable
`c`, the true training category may be uncertain:

```text
p_i = [P(c_i=1), ..., P(c_i=K)]
```

where `p_i >= 0` and `sum(p_i)=1`.

One-hot vectors recover known categories exactly.

## Expected categorical kernel

Use a learnable positive-semidefinite category covariance matrix `K_cat`.
For probability vectors `p_i` and `p_j`:

```text
E[k_cat(c_i,c_j)] = p_i^T K_cat p_j
```

This is positive semidefinite because it is a bilinear expectation of a
positive-semidefinite category kernel.

For mixed continuous/categorical inputs, compose it with a continuous kernel:

```text
k((x_i,p_i),(x_j,p_j))
= k_cont(x_i,x_j) * p_i^T K_cat p_j
```

The first implementation should support one uncertain categorical feature with
`K` categories. Multiple uncertain categorical features can later be composed
as products if a concrete use case requires them.

## Category covariance parameterization

Parameterize

```text
K_cat = L L^T + jitter I
```

with learnable `L`. This guarantees positive semidefiniteness.

Do not use ordinal distance between category indices.

## Proposed API

```python
model = UncertainCategoricalSingleTaskGP(
    train_X_cont,
    train_category_probabilities,
    train_Y,
)
```

where:

- `train_X_cont.shape == (..., n, d_cont)`;
- `train_category_probabilities.shape == (..., n, K)`;
- probabilities are finite, nonnegative, and row-normalized.

For prediction:

```python
posterior = model.posterior_with_category_probabilities(
    X_cont,
    category_probabilities,
)
```

A deterministic category is passed as a one-hot probability vector.

## BoTorch optimization boundary

Standard BoTorch acquisition functions expect a tensor-valued candidate
`X`. The uncertain-category probability vector is therefore part of the
surrogate input representation, not a hidden side channel.

The implementation should internally concatenate:

```text
[X_cont, category_probabilities]
```

and expose metadata identifying the continuous and probability blocks.

Optimization over the probability simplex is not automatically equivalent to
choosing a physical category. The model is primarily intended for uncertain
observed training categories. Candidate generation should normally use one-hot
category vectors unless the application genuinely controls a mixture.

## Exact-GP contract

The expected kernel remains Gaussian and positive semidefinite, so the model
should preserve:

- exact GP inference;
- `supports_mll = True`;
- `make_mll()`;
- raw continuous inputs, category probabilities, and targets;
- BoTorch posterior/acquisition compatibility on the augmented tensor.

## Validation

Reject:

- negative probabilities;
- non-finite probabilities;
- rows that do not sum to one within a documented tolerance;
- fewer than two categories;
- feature-width mismatches.

Do not silently renormalize invalid probabilities.

## Required tests

1. one-hot probabilities recover the deterministic category kernel;
2. expected categorical covariance is symmetric;
3. covariance is positive semidefinite;
4. probability mixtures interpolate category covariance;
5. invalid simplex rows are rejected;
6. gradients reach category covariance parameters;
7. finite exact MLL;
8. raw data retention;
9. dtype/device migration and state-dict roundtrip;
10. posterior batch/q shapes;
11. qMC acquisition smoke test;
12. no integer-category jitter or ordinal-distance behavior.

## Scope boundaries

Phase 14 does not add:

- uncertain category labels without probability information;
- automatic inference of category probabilities;
- categorical input perturbation by integer jitter;
- multiple uncertain categorical columns;
- mixed/reduced/multitask cross-product classes.

## Exit criterion

Phase 14 is complete when uncertain observed categories can be represented by
explicit simplex probabilities and marginalized through a positive-semidefinite
expected categorical kernel without changing the standard exact-GP contract.
