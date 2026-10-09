# Phase 5 gap closure: BoTorch GP benchmark strategy

`make_botorch_gp_strategy` plugs native BoTorch `SingleTaskGP`,
`fit_gpytorch_mll`, `qExpectedImprovement` or
`qNoisyExpectedImprovement`, and `optimize_acqf` into the existing
`run_benchmark` candidate callback.

The strategy is deliberately restricted to unconstrained, continuous,
single-objective benchmarks. It trains on observed values only, converts
minimization to maximization internally, and does not access truth or
known optimal values during candidate selection.

```python
from robotorchan.benchmarks import (
    BenchmarkExperimentConfig,
    BenchmarkProblemRegistry,
    make_botorch_gp_strategy,
    run_benchmark,
)
from robotorchan.benchmarks.standard_problems import register_standard_problems

registry = BenchmarkProblemRegistry()
register_standard_problems(registry)
config = BenchmarkExperimentConfig(
    problem="branin",
    strategy="botorch-qNEI",
    seeds=(0,),
    initial_points=8,
    evaluation_budget=4,
)
strategy = make_botorch_gp_strategy(
    acquisition="qNEI", num_restarts=2, raw_samples=32
)
trajectory = run_benchmark(config, strategy, registry=registry)[0]
print(trajectory.X.shape)
```

The benchmark runner's Sobol initial design depends only on problem,
seed, dtype and device, so Random and GP strategies can use matched
initial conditions. This is a smoke-test baseline, not a performance
claim. Fair multi-seed comparisons and noisy-objective experiments
belong to later phases.
