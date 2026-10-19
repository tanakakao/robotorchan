# Heterogeneous E2E synthetic benchmark (Phase 2)

The fixed benchmark is implemented in `src/robotorchan/benchmarks/heterogeneous_synthetic.py`.
All three design variables are continuous in `[0, 1]`:
temperature (`x0`), pressure (`x1`), and composition (`x2`).
These are illustrative normalized coordinates, not calibrated physical quantities.

- Strength: `1.3 - 2(x0-.78)^2 - 1.5(x1-.28)^2 - .7(x2-.68)^2`.
- Conductivity: `1.2 - 1.8(x0-.22)^2 - 1.6(x1-.76)^2 - .9(x2-.35)^2`.
- Pass probability: `sigmoid(9(.52 - (x0-.50)^2 - 1.2(x1-.53)^2 - .7(x2-.50)^2) - 2.5)`.

Strength and Conductivity are **maximized**. Their distinct maxima produce
a trade-off. The true probability is not an observed binary label:
`Pass ~ Bernoulli(p(x))`. The deterministic reference feasibility rule is
`p(x) >= 0.5`, whereas observed Pass is a stochastic training label.
Neither an observed Pass=1 nor a model-predicted probability is a substitute
for ground-truth reference feasibility.

`observe(X, seed, noise_std=0.02)` draws independent Gaussian noise for
both regression targets and Bernoulli labels from a local seeded generator.
Reusing the same seed with the same input tensor and shape reproduces the
observation; seed independence must be managed at the experiment-loop level.

`reference_front(grid_size=25, probability_threshold=0.5)` constructs
a deterministic 3D Cartesian grid, filters by **true** feasibility, and
returns its nondominated objective pairs. This is a **discrete numerical
approximation**, not an analytical continuous Pareto front. Report the grid
resolution alongside any reference hypervolume, and assess convergence with
increasing grid sizes. Hypervolume reference points must be chosen below
the feasible objectives in maximization space and recorded explicitly.

Phase 2 establishes the fixed data-generating process only. It does not
claim trained-model calibration, constrained acquisition correctness,
closed-loop superiority, or CUDA compatibility. Those are later phases.

Run the contract tests with:

```bash
pytest -q tests/benchmarks/test_heterogeneous_synthetic.py
```
