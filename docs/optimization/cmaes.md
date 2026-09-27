# CMA-ES optimizer backend

Phase 7 adds CMA-ES as an optional derivative-free acquisition optimizer.
The backend consumes a BoTorch `AcquisitionFunction`, tensor bounds, and `q`,
while the `cmaes` package owns the evolutionary update loop.

The joint `q x d` candidate batch is flattened to one CMA-ES vector and restored
before every acquisition evaluation. The initial mean is the center of the box.
The public `sigma` is relative to the largest box width, which avoids silently
treating the same absolute sigma as meaningful across differently scaled domains.

The dependency is optional: install `robotorchan[cmaes]` when this backend is
needed. Importing robotorchan does not require `cmaes`.

Candidate constraints are explicitly rejected until the cross-optimizer
constraint phase. Population orchestration is CPU-side; acquisition evaluation
uses the tensor device and dtype supplied by the caller.

```python
from robotorchan.optim.backends import optimize_acqf_cmaes

candidates, value = optimize_acqf_cmaes(
    acq_function=acq,
    bounds=bounds,
    q=3,
    sigma=0.25,
    max_generations=200,
    seed=123,
)
```
