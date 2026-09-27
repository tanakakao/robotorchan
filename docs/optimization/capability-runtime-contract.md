# Optimizer capability and runtime contract

Phase 19 makes optimizer metadata executable rather than documentation-only.

`OPTIMIZER_CAPABILITIES` is the canonical registry for scalar public optimizer names, and `get_optimizer_capabilities(name)` exposes the corresponding contract. The `optimize_acqf` dispatcher validates requested fixed features, sequential behavior, and candidate-constraint families before numerical optimization starts.

This prevents metadata and runtime behavior from drifting silently. Unsupported combinations raise before entering the backend.

The runtime RNG contract is also explicit: a supplied seed creates a backend-local `torch.Generator`; `seed=None` returns `None` so normal PyTorch global RNG behavior is preserved instead of accidentally repeating an implicit default generator seed. Bounds must contain at least one finite floating-point dimension with strict lower/upper ordering.
