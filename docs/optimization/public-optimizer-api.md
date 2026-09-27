# Optimizer public API

robotorchan keeps BoTorch's acquisition-function vocabulary and adds an optional named optimizer selector.

The public entry point is `from robotorchan.optim import optimize_acqf`. The default `optimizer="botorch"` delegates to the BoTorch-native backend. Existing direct use of `botorch.optim.optimize_acqf` remains valid.

## Named backends

The convenience dispatcher exposes `botorch`, `torch_adam`, `torch_adamw`, `torch_sgd`, `random`, `sobol`, `de`, `cmaes`, `ga`, `pso`, and `hybrid`.

Backend-specific controls belong in `optimizer_options`. Common BoTorch-shaped arguments stay explicit when they have the same semantics. Unsupported combinations fail explicitly rather than silently dropping an argument.

Mixed-variable GA and vector-valued NSGA-II remain explicit specialist APIs under `robotorchan.optim.backends`; they are not disguised as ordinary scalar continuous `optimize_acqf` calls.

CMA-ES is the only optimizer backend in this set with an additional optional runtime dependency. Install the `cmaes` extra when that backend is required.


## Structured variable spaces

The public dispatcher accepts an optional `MixedVariableSpace` alongside the usual
BoTorch-shaped `bounds`, `q`, constraints, and fixed features. This is the unified
robotorchan layer for declaring integer and categorical coordinates; continuous
coordinates remain implicit.

The dispatcher validates the requested variable types against optimizer capability
metadata before backend execution. Unsupported combinations therefore fail explicitly
instead of silently rounding or ignoring structured coordinates. Common fixed features
are also validated against the declared integer and categorical domains.

Backend functions remain public for direct BoTorch-style use. `MixedVariableSpace`
does not replace backend-specific low-level arguments; it provides a reusable contract
for the unified dispatcher. Supplying both representations to a backend that forbids
ambiguous duplicate declarations remains an error.

Random and Sobol sampling, DE, GA, PSO, and Hybrid receive the declared variable space
when supported. Gradient Torch optimizers reject integer or categorical spaces. PSO
rejects categorical spaces. The plain `optimizer="botorch"` dispatch path remains the
continuous BoTorch optimizer; callers needing BoTorch mixed enumeration should use the
explicit mixed backend where its `fixed_features_list` semantics are visible.
