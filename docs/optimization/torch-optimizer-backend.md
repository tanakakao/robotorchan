# PyTorch acquisition optimizer backend

Phase 4 adds a BoTorch-native PyTorch optimizer path without replacing the
BoTorch acquisition-function API. The backend calls `botorch.optim.optimize_acqf`
for restart initialization and candidate selection, while
`botorch.generation.gen.gen_candidates_torch` performs local optimization.

Supported optimizer names are `adam`, `adamw`, and `sgd`. A
`torch.optim.Optimizer` subclass can also be supplied directly. Adam is the
default. LBFGS is intentionally not advertised because BoTorch
`gen_candidates_torch` manages its own closure and stopping loop; treating
LBFGS as interchangeable would require optimizer-specific semantics and tests.

The backend supports continuous box bounds, q-batches, fixed features, tensor
device/dtype preservation, and GPU execution when the acquisition function and
tensors are on GPU. Candidate constraints are rejected explicitly in Phase 4;
they are handled in the cross-optimizer constraint phase rather than by a
silent penalty approximation.

Example:

```python
from robotorchan.optim.backends import optimize_acqf_torch

candidates, value = optimize_acqf_torch(
    acq_function=acq,
    bounds=bounds,
    q=3,
    num_restarts=10,
    raw_samples=512,
    optimizer="adam",
    options={
        "optimizer_options": {"lr": 0.025},
        "stopping_criterion_options": {"maxiter": 200},
    },
)
```
