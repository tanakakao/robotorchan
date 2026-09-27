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

The initial implementation supports DE and CMA-ES by name and accepts a compatible
custom global optimizer callable. Candidate constraints are supplied to both stages.

Only one local restart is currently accepted. Duplicating the same global candidate
across multiple restarts provides no additional initialization diversity, so
`num_restarts > 1` is rejected rather than pretending to perform independent
multi-start refinement. Distinct global restart generation can be added later.

`fixed_features` is also rejected for now because the global stage must honor the
same fixed coordinates before the local refinement can claim correct semantics.
