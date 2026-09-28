# Optimizer capability and runtime contract

`OPTIMIZER_CAPABILITIES` is the canonical registry for the scalar optimizer
names accepted by the high-level `optimize_acqf` dispatcher.
`get_optimizer_capabilities(name)` exposes the corresponding runtime contract.

The mapping is intentionally not an inventory of every optimizer capability
descriptor in robotorchan. Specialist APIs such as BoTorch mixed optimization,
Mixed GA, and NSGA-II expose their own public `OptimizerCapabilities` constants,
but they are not scalar `optimizer=...` names accepted by the dispatcher.

The dispatcher validates requested fixed features, sequential behavior, and
candidate-constraint families before numerical optimization starts. Unsupported
combinations raise before entering the backend.

The runtime RNG contract is also explicit: a supplied seed creates a backend-local
`torch.Generator`; `seed=None` returns `None` so normal PyTorch global RNG
behavior is preserved instead of accidentally repeating an implicit default
generator seed. Bounds must contain at least one finite floating-point dimension
with strict lower/upper ordering.
