# Phase 18: Edge cases and failure handling

This phase adds negative-path tests around the existing heterogeneous synthetic benchmark and its multi-seed evaluation helpers, without changing the model or optimization API.

Covered contracts:

- Reject invalid input shape, integer input dtype, out-of-bounds design points, and nonfinite values in `evaluate_truth`.
- Reject negative, NaN, or infinite observation noise and noninteger seeds in `observe`.
- Verify reproducible noisy regression outputs and Bernoulli labels under fixed seeds.
- Reject invalid reference-front thresholds and degenerate grids; return stable empty tensor shapes when the feasibility threshold is unattainable.
- Reject unknown baseline strategy identifiers, duplicate/insufficient seeds, invalid evaluation budgets, and nonfloating/nonfinite statistical trajectories.

**Limitations:** These tests cover benchmark-facing failure contracts, not optimizer numerical failures, invalid candidate constraint metadata, GPU out-of-memory, asynchronous pending-point state, or malformed model training data. They do not imply all failure modes are handled gracefully.

Run `pytest -q tests/benchmarks/test_heterogeneous_edge_failures_e2e.py`.
