# Mixed-variable Genetic Algorithm backend

Phase 9 extends evolutionary acquisition optimization to heterogeneous input spaces.

The backend accepts ordinary BoTorch-style numeric bounds together with explicit
structured dimensions:

- continuous dimensions remain real-valued;
- `integer_dims` are rounded and clipped after initialization and variation;
- `categorical_values` define the legal encoded values for categorical dimensions.

Categorical variables are not treated as ordered continuous values after variation.
They are repaired by sampling from their declared legal domain. Integer and
categorical declarations may not overlap.

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

Candidate constraints remain disabled until the cross-optimizer constraint phase.
