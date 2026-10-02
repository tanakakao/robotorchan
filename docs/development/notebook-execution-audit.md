# Notebook Execution Audit

## Scope

Phase 25 audits whether the executable-documentation suite is actually runnable from a clean
kernel and whether every committed notebook has an explicit execution policy.

The primary execution path is `examples/run_notebooks.py`, which uses `nbclient.NotebookClient`
to execute cells from top to bottom in a fresh kernel.

## Execution contract

A notebook in `examples/notebooks/` must belong to exactly one manifest group:

- `DEFAULT_NOTEBOOKS`: executed in normal notebook CI;
- `SLOW_NOTEBOOKS`: executed only with `--include-slow`;
- `EXCLUDED_NOTEBOOKS`: intentionally not executed by this runner.

The runner validates that:

1. every `.ipynb` file is declared;
2. no declared notebook is missing;
3. no notebook belongs to multiple execution groups.

This prevents a new notebook from silently bypassing executable-documentation CI.

## Current intentionally excluded notebooks

The current excluded group is:

- `23_high_dimensional_benchmark.ipynb`;
- `24_expressive_surrogate_gp.ipynb`;
- `25_non_gp_surrogates.ipynb`.

Exclusion is an execution-policy statement, not a correctness claim. Phase 26 may move suitable
notebooks into scheduled or optional CI tiers.

## Clean-kernel semantics

`NotebookClient` executes each notebook from the first cell in a fresh kernel. Therefore a default
or slow notebook cannot rely on:

- variables created by a previous notebook;
- manually executed cells;
- stored cell outputs;
- an interactive working directory outside `examples/notebooks/`.

The runner sets the notebook working directory explicitly and fails on cell exceptions or timeout.

## Runtime coverage

The default manifest includes the cross-layer notebooks added through the executable-documentation
program, including acquisition, multi-objective, active learning, objective / transform, sampling,
optimization, batch / async, initialization, TuRBO, robust input perturbation, and the integrated
end-to-end workflow.

Slow Fully Bayesian SAAS and structured-output examples remain opt-in because normal PR CI should
stay bounded.

## Audit conclusion

The notebook execution path already provided strong clean-kernel coverage. The material gap was
manifest completeness: files could exist without belonging to any execution tier. The manifest
validator closes that gap without changing notebook semantics or adding another execution
framework.
