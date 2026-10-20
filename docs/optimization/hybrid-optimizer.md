# Hybrid global-to-local acquisition optimization

Phase 11 adds a two-stage optimizer that combines global derivative-free search
with BoTorch local refinement.

```text
DE or CMA-ES
    -> promising joint q x d candidate
    -> BoTorch optimize_acqf with explicit batch_initial_conditions
    -> refined candidate
```

The global stage is useful when the acquisition landscape is multimodal or a
local gradient-based optimizer is sensitive to initialization. The local stage
retains BoTorch's native constrained optimization behavior.

```python
from robotorchan.optim.backends import optimize_acqf_hybrid

candidates, value = optimize_acqf_hybrid(
    acq_function=acq,
    bounds=bounds,
    q=3,
    global_optimizer="de",
    global_options={"maxiter": 100},
    local_options={"maxiter": 200},
    constraints=constraints,
    seed=123,
)
```

The implemented global stage supports `de`, `cmaes`, and `mixed_ga` by name,
plus a compatible custom optimizer callable. Candidate constraints are supplied
to both stages. Each global backend retains its own constraint-handling semantics;
see [Candidate constraints](candidate-constraints.md).

Only one local restart is currently accepted. Duplicating the same global candidate
across multiple restarts provides no additional initialization diversity, so
`num_restarts > 1` is rejected rather than pretending to perform independent
multi-start refinement. Distinct global restart generation can be added later.

`fixed_features` is supported and forwarded to the global and local stages.
When `variable_space` declares integer or categorical dimensions, the caller
must also supply `mixed_fixed_features_list` for the BoTorch mixed local stage.
Every configuration must fix each structured dimension, either directly or
through common `fixed_features`. The global optimizer must support the declared
variable types; `cmaes` does not accept `variable_space` through this path.
Unsupported combinations fail explicitly rather than silently dropping
structured coordinates.
