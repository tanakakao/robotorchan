# Optimizer benchmark suite

`robotorchan.benchmarks` provides a common measurement harness for scalar acquisition
optimizers.

```python
from robotorchan.benchmarks import benchmark_optimizer, benchmark_optimizers
```

The benchmark layer is deliberately separate from optimizer implementations. It does not choose
a preferred optimizer and does not alter optimizer budgets.

## Common measurements

Each `BenchmarkResult` records:

- returned candidate,
- raw acquisition value at that candidate,
- wall-clock runtime,
- number of q-batches evaluated by the acquisition,
- candidate-space feasibility when constraints are supplied,
- optimizer seed.

Acquisition evaluations are counted at the q-batch level. A vectorized call
with shape `[128, q, d]` therefore counts as 128 evaluations rather than one
Python call.

## Fair comparisons

Optimizer-specific budgets are not directly interchangeable. For example,
`raw_samples`, DE population/generations, GA population/generations, and PSO
swarm/iterations imply different acquisition evaluation counts. Reports should
therefore show both wall time and measured acquisition evaluations.

For stochastic methods, use several seeds and summarize the distribution of
candidate quality rather than relying on one run.

## Constraints

The harness reports final candidate feasibility independently of the
optimizer's internal penalty or exact-constraint mechanism. This makes
constraint failures visible instead of treating a high penalized internal
score as successful optimization.

## Intended benchmark problems

The reusable harness can support common problem suites including smooth continuous,
multimodal, q-batch, mixed/discrete, constrained, noisy, and higher-dimensional cases.

Optional backends such as CMA-ES should only enter a benchmark matrix when
their optional dependency is installed.
