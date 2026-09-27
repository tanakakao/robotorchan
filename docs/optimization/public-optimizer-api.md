# Optimizer public API

robotorchan keeps BoTorch's acquisition-function vocabulary and adds an optional named optimizer selector.

The public entry point is `from robotorchan.optim import optimize_acqf`. The default `optimizer="botorch"` delegates to the BoTorch-native backend. Existing direct use of `botorch.optim.optimize_acqf` remains valid.

## Named backends

The convenience dispatcher exposes `botorch`, `torch_adam`, `torch_adamw`, `torch_sgd`, `random`, `sobol`, `de`, `cmaes`, `ga`, `pso`, and `hybrid`.

Backend-specific controls belong in `optimizer_options`. Common BoTorch-shaped arguments stay explicit when they have the same semantics. Unsupported combinations fail explicitly rather than silently dropping an argument.

Mixed-variable GA and vector-valued NSGA-II remain explicit specialist APIs under `robotorchan.optim.backends`; they are not disguised as ordinary scalar continuous `optimize_acqf` calls.

CMA-ES is the only optimizer backend in this set with an additional optional runtime dependency. Install the `cmaes` extra when that backend is required.
