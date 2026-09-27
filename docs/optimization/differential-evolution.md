# Differential Evolution optimizer backend

Phase 6 adds Differential Evolution (DE) as a derivative-free acquisition
optimizer. It uses SciPy `differential_evolution` and keeps the BoTorch
`AcquisitionFunction`, tensor bounds, and q-batch concepts.

A joint `q x d` candidate is flattened to a `q*d` DE decision vector. Each
objective evaluation reconstructs the tensor on the acquisition function device
and dtype, evaluates the acquisition, and minimizes its negative value.

The backend is intentionally CPU-orchestrated because SciPy owns the evolutionary
loop. Acquisition evaluation can still execute on the model tensor device, but
Phase 6 does not advertise GPU capability or vectorized population evaluation.

Candidate constraints are rejected explicitly in Phase 6. Constraint translation
is handled in the dedicated cross-optimizer constraint phase. SciPy polishing is
disabled by default so the DE backend remains derivative-free and its semantics
do not silently change to a local optimizer.

```python
from robotorchan.optim.backends import optimize_acqf_de

candidates, value = optimize_acqf_de(
    acq_function=acq,
    bounds=bounds,
    q=3,
    seed=123,
    options={"maxiter": 200, "popsize": 15},
)
```


## Integer and mixed continuous/integer variables

Differential Evolution now supports continuous plus integer search spaces. Integer
coordinates remain part of the DE decision vector, but every acquisition and
constraint evaluation repairs those coordinates to the nearest legal integer and
clips them to the integer domain defined by the bounds. The returned candidate is
repaired with the same rule.

This is intentionally limited to continuous and integer variables in this phase.
Categorical variables are rejected rather than treating category labels as a
numeric metric. Categorical-aware DE mutation is evaluated separately before it
can be advertised as supported.

Both the direct `integer_dims` backend argument and a `MixedVariableSpace`
containing only continuous/integer variables are supported. Supplying both is
rejected to keep a single source of search-space semantics.
