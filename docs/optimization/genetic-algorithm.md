# Genetic Algorithm optimizer backend

Phase 8 adds a native real-valued Genetic Algorithm (GA) backend for acquisition
optimization. It has no additional package dependency and uses PyTorch tensors for
population generation, selection, crossover, mutation, and batched acquisition
evaluation.

A population member represents the complete joint `q x d` candidate. Internally
it is flattened to `q*d`; the acquisition function receives `[population, q, d]`.
This makes q-batch optimization joint rather than optimizing each candidate
independently.

The implementation uses elitism, tournament selection, arithmetic crossover, and
Gaussian mutation scaled by each variable range. Mutation is clipped to the box.
A dedicated local `torch.Generator` provides reproducibility without mutating the
global random stream.

Because the population remains on the bounds tensor device, this backend supports
GPU-resident evolutionary operations and batched GPU acquisition evaluation.

Phase 8 is intentionally continuous-only. Integer and categorical genes belong to
the mixed-variable evolutionary phase. Candidate constraints are also rejected
until the dedicated cross-optimizer constraint phase.

```python
from robotorchan.optim.backends import optimize_acqf_ga

candidates, value = optimize_acqf_ga(
    acq_function=acq,
    bounds=bounds,
    q=3,
    population_size=128,
    generations=100,
    seed=123,
)
```
