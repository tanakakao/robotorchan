# Mixed-variable search-space contract

`MixedVariableSpace` describes variable semantics in public/raw candidate
coordinates. It is separate from surrogate-model categorical kernels, candidate
constraints, and numerical optimizer backends.

A dimension has exactly one owner:

- dimensions omitted from structured metadata are continuous;
- `integer_dims` use every integer in `ceil(lower) .. floor(upper)`;
- `categorical_values` define an explicit finite unordered domain.

The contract rejects duplicate ownership, out-of-range dimensions, integer
bounds containing no legal integer, empty or duplicate categorical domains,
non-finite values, and categorical values outside the public bounds.

```python
space = MixedVariableSpace(
    bounds,
    integer_dims=(1,),
    categorical_values={2: (0.0, 1.0, 2.0)},
)
```

## q-batch semantics

Variable metadata is defined per public feature dimension, not per flattened
q-batch coordinate. An optimizer handling `q > 1` applies the same variable
semantics independently to each candidate row. Inter-point relationships remain
the responsibility of `CandidateConstraints`.

## Fixed features

`validate_fixed_features` checks fixed values in the same public coordinates.
Integer fixed values must be integral and categorical fixed values must belong
to the declared category set. This prevents a backend from silently repairing
an invalid fixed feature.

## Responsibility boundary

`MixedVariableSpace` says which values are legal. It does not define how they
are searched. A GA may use categorical mutation, DE may support only continuous
and integer subsets, and gradient optimizers may reject all structured
dimensions. Those choices belong to optimizer capabilities and runtime
validation.

The contract also does not replace BoTorch `fixed_features_list`.
`MixedSpaceStrategy` remains the BoTorch-first enumerated mixed path. Later
integration may derive or validate backend-specific representations from the
shared variable-space contract without changing BoTorch constraint semantics.
