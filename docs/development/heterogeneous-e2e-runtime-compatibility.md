# Phase 17: Runtime and shape compatibility

Exercise the existing heterogeneous qEI pipeline with `SingleTaskGP` across float32/float64 and q=1/2. Tests fit the GP, verify native BoTorch posterior shapes `(batch=2, q, output=1)`, evaluate batched acquisition values, and generate bounded candidates through the Sobol optimizer with feature 2 fixed at 0.5. The selected candidates must retain the input dtype and satisfy fixed-feature and bound contracts.

A separate CUDA-only test validates posterior and acquisition evaluation on GPU when CUDA is available. It skips on CPU-only CI and does not exercise GPU optimization.

**Scope limitations:** This phase tests fixed features, not general nonlinear candidate constraints, pending-point asynchronous selection, or every optimizer backend. These remain distinct compatibility scenarios. Float32 may trigger BoTorch precision warnings. No accuracy or convergence guarantee is implied.

Run `pytest -q tests/benchmarks/test_heterogeneous_runtime_compatibility_e2e.py`.
