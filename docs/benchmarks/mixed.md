# Mixed-variable benchmark problems

The benchmark suite represents categorical variables as **integer-coded
nominal labels**, not as ordinal numeric distances. Candidate generators
must return valid integer codes within bounds. No one-hot encoding is
required by the problem contract.

| Problem | Inputs | Objective | Constraints |
| --- | --- | --- | --- |
| mixed_quadratic | continuous, integer, category | minimize; optimum 0 | none |
| categorical_switch | continuous, category | maximize; optimum 1 | none |
| mixed_process_yield | temperature, integer cycle count, recipe category | maximize; optimum 1 | recipe-dependent temperature limit; cycles <= 4 |
| mixed_category_interaction | continuous, two categorical choices | minimize; optimum 0 | none |

**Process yield** has recipe-dependent response centers and upper
temperature limits. The feasible optimum is (0.7, 3, 1).
This exercises the interaction between categorical choices and
continuous outcome constraints.

**Category interaction** has a non-additive lookup table for the
continuous response center and categorical penalty. Its optimum
is (0.85, 1, 1), demonstrating a case where two nominal variables
jointly determine the landscape.

The existing `sobol_initial_design` and `random_candidates` functions
round discrete coordinates. For more sophisticated mixed-variable
optimization, use a compatible candidate generator with the same
`(q, d)` tensor contract.

Register all problems using `register_mixed_problems(registry)`.

## Executable mixed-variable smoke example

```python
from robotorchan.benchmarks import (
    BenchmarkExperimentConfig,
    BenchmarkProblemRegistry,
    random_candidates,
    register_mixed_problems,
    run_benchmark,
)

registry = BenchmarkProblemRegistry()
register_mixed_problems(registry)
config = BenchmarkExperimentConfig(
    problem="mixed_quadratic",
    strategy="random",
    seeds=(0, 1, 2),
    initial_points=8,
    evaluation_budget=24,
    q=3,
)
trajectories = run_benchmark(config, random_candidates, registry=registry)
```

The `mixed_quadratic` optimum is `(0.8, 2, 1)` and the
`categorical_switch` optimum is `(0.7, 1)`. Both have optimal value
`0` (minimize) and `1` (maximize), respectively. Categorical codes
such as `0`, `1`, and `2` are nominal labels without an ordinal distance.
Rounding by the baseline Sobol and random candidate generators supports
an end-to-end smoke test, but does **not** establish uniform sampling
across discrete levels or the quality of specialized mixed-variable
acquisition optimization.
