# Optimizer runtime contract

Phase 15 defines runtime behavior shared by robotorchan optimizer backends.

## dtype and device

Optimizer bounds must be floating-point, finite tensors with shape `[2, d]`
and strictly ordered lower/upper limits. Tensor-native backends create
candidates with the bounds dtype and device. Returned candidates preserve that
dtype/device.

CPU-orchestrated SciPy DE and optional CMA-ES may move decision vectors to CPU
for their optimizer internals, but acquisition evaluation and returned
candidates are reconstructed on the bounds dtype/device. Their capability
metadata therefore does not claim native GPU optimization.

## random seeds

A supplied seed controls the optimizer's own stochastic search. Native PyTorch
backends use a backend-local `torch.Generator`; they do not call
`torch.manual_seed` and therefore do not intentionally mutate the process-wide
PyTorch RNG state.

Sobol sampling uses a locally constructed `SobolEngine`. SciPy DE and CMA-ES
receive their seed through their own APIs.

A seed is a reproducibility control for the optimizer backend. It does not make
an acquisition deterministic if the acquisition itself performs uncontrolled
stochastic sampling.

## reproducibility scope

Reproducibility is defined for the same robotorchan/BoTorch/PyTorch dependency
versions, dtype/device, optimizer options, acquisition state, and seed. Exact
cross-device or cross-version bitwise identity is not promised.

## GPU

Tensor-native population methods (sampling, GA, mixed GA, PSO, NSGA-II) can
batch objective/acquisition evaluation on the bounds device. Torch local
optimization is also device-native. SciPy DE, CMA-ES, and the current hybrid
orchestration remain CPU-oriented at the numerical optimizer layer.
