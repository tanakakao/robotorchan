# Mixed-variable Genetic Algorithm backend

Phase 9 extends evolutionary acquisition optimization to heterogeneous input spaces.

The backend accepts ordinary BoTorch-style numeric bounds together with explicit
structured dimensions:

- continuous dimensions remain real-valued;
- `integer_dims` are rounded and clipped after initialization and variation;
- `categorical_values` define the legal encoded values for categorical dimensions.

Categorical variables are never treated as ordered continuous values. Crossover
inherits a categorical value from one parent, and mutation samples directly from
the declared category set. Integer crossover also inherits parent values, while
integer mutation uses discrete one-step moves followed by legal-domain repair.

The preferred search-space contract is `MixedVariableSpace`, shared with the
sampling backends. The legacy low-level `integer_dims` / `categorical_values`
arguments remain available to this backend, but they cannot be supplied together
with `variable_space`. Integer and categorical declarations may not overlap.

`fixed_features` are validated by `MixedVariableSpace` when that contract is
used, expanded across the joint q-batch, and restored after every variation step.

A population member represents the complete joint `q x d` candidate, so the
backend preserves the joint semantics of q-batch acquisition functions. Population
evaluation is batched and stays on the tensor device, including CUDA when the
acquisition function and bounds are CUDA-resident.

```python
from robotorchan.optim.backends import optimize_acqf_mixed_ga

candidates, value = optimize_acqf_mixed_ga(
    acq_function=acq,
    bounds=bounds,
    q=3,
    integer_dims=[1, 4],
    categorical_values={2: [0.0, 1.0, 2.0]},
    population_size=128,
    generations=100,
    seed=123,
)
```

This is deliberately a numerical optimizer backend, not another search-space
strategy. It can therefore be reused later by higher-level mixed or tree search
strategies.

Candidate constraints use the common penalty implementation described in
`candidate-constraints.md`. Phase 7 separately evaluates penalty ranking against
feasibility-first selection.
