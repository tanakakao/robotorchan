# Cross-library benchmark

Phase 3 establishes an executable CPU comparison for robotorchan, BoTorch, Ax,
Optuna, BayBE and BoFire. It also provides Random and Sobol baselines.
Comparisons use the existing benchmark problems, trajectory format and metrics.

## Reproduce the smoke run

From the repository root on Linux x86_64, create a separate CPython 3.12 environment.

```sh
python3.12 -m venv .venv-benchmark
. .venv-benchmark/bin/activate
python -m pip install --require-hashes -r benchmarks/cross_library/requirements-linux-x86_64-py312.lock
python -m pip install --no-deps -e .
python -m pip check
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/cross_library/run.py --seeds 0 \
  --output results/cross-library-smoke.jsonl --summary results/cross-library-smoke.csv
```

Use new output paths on each run. Select cells with `--libraries` and
`--problems`, or supply a JSON array of cells with `--config`. Each cell has
`problem`, `library`, `q`, `initial`, `total` and `mode`. The `total` budget
includes initial evaluations. The runner shortens the last batch to the
remaining evaluation budget. `--profile standard` increases restart and
Monte Carlo budgets in the low-level GP adapters. Use a frozen experiment
configuration before running longer comparisons.

The smoke configuration covers Branin, Mixed Quadratic, Branin–Currin, the
constrained annulus, Noisy Quadratic, Hartmann6 and C2-DTLZ2. It exercises
the matched GP path and the Optuna GPSampler entry point separately. Ax uses
the default public generation strategy. Its initialization trials count
against the budget, and its multiple-trial request is reported as such.
BayBE and BoFire use library-managed batch recommendations. The evidence
shows the public batch count, while the result metadata identifies the batch
method. The initial experiments may include repeated settings when permitted.
For Phase 4, fix the allowed repetition rules per problem and account for
any upstream rejection in the result report.

## Data contract

For a given problem, initial count and seed, the Runner generates the same
Sobol initial design across adapters, with equal-width bins on discrete axes.
The initial observed data and its SHA-256 ID are stored in every result.
Paired Gaussian noise in Noisy Quadratic uses a separate seeded stream
shared across strategies with the same batch schedule. Integer and nominal
categorical parameters use their respective public types. Candidate creation
sees the restricted search-space metadata and evaluated observations. An
entire batch is evaluated before the adapter receives its feedback.

The common constraint convention is feasible when all residuals are
nonnegative. The BoTorch acquisitions and Optuna trials receive the
corresponding nonpositive residual convention. Hypervolume uses the fixture
reference point across all participants. Each library's native acquisition
reference handling is part of its selected algorithm. Native objective
directions are retained in logged observations. Regret and hypervolume are
evaluated on the truth trajectory. Empty feasible history yields null regret
and zero hypervolume when defined.

JSONL output uses [schema version 1](result.schema.json). Every record contains the cell, seed,
distribution version, commit, environment, algorithm, batch method, initial
data and ID, evaluation count and cost, observations, metrics, timing, status
and error. The nested `trajectory` reuses
`robotorchan.benchmarks.persistence.trajectory_from_record`. Fast diagnostic
runs have the same record format as longer cells. Fit time is measured in the
low-level GP route and otherwise left null. Candidate wall time includes fit
when performed during candidate generation. The CSV contains the final
indicators, timing and status. Success counts and failure records are retained.
The CLI exits with a nonzero status if a cell fails, after writing its report.

## Scope and version pins

The lock pins the actual smoke distributions and their archive hashes. The
robotorchan code is identified by the checkout commit and installed separately.
The Phase 1 robotorchan reference was `3b0ed04`. This implementation is based on
`3627bae`, the main head at the start of Phase 3. External core versions remain
BoTorch 0.18.1, Ax 1.3.1, Optuna 5.1.0, BayBE 0.15.2 and BoFire 0.5.0.

The optional comparison dependencies are confined to this environment.
Additional execution adapters for isolated or specialist libraries, including
Vizier, Trieste, SMAC3, OpenBox and pymoo, need separate validation before their
Phase 4 cells are used. The current adapter contract covers synchronous BO.
Pool AL, asynchronous evaluation and cost-budget stopping have their own
benchmark contracts. Statistical performance evaluation is covered in Phase 4.
