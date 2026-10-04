# Executable Notebook Documentation

## Purpose

The notebook suite connects robotorchan theory, public APIs, and executable Bayesian optimization
or active-learning workflows.

The canonical cross-layer pipeline is:

```text
Surrogate Model
    -> Posterior
    -> Sampling when required
    -> Objective / PosteriorTransform when required
    -> Acquisition Function
    -> Acquisition Optimization / Search Strategy
    -> Candidate
    -> Evaluation / Update
```

Not every notebook demonstrates every layer. Model and specialist notebooks preserve their real
semantics instead of forcing an artificial end-to-end Gaussian BO workflow.

## Where to start

Use these entry points for common tasks:

| Goal | Notebook |
| --- | --- |
| First end-to-end BO | `01_single_task_gp.ipynb` |
| Mixed-variable BO | `02_mixed_single_task_gp.ipynb` |
| Multi-Fidelity / cost-aware BO | `03_multi_fidelity_gp.ipynb` |
| MultiTask / Kronecker | `04_multitask_gp.ipynb` |
| High-dimensional surrogate reduction | `19_reduced_gp.ipynb` |
| Standard acquisition functions | `26_standard_acquisition.ipynb` |
| Multi-objective BO | `27_multiobjective_acquisition.ipynb` |
| Regression active learning | `28_active_learning_acquisition.ipynb` |
| Objective / PosteriorTransform | `29_objective_posterior_transform.ipynb` |
| Posterior sampling | `30_posterior_sampling.ipynb` |
| Acquisition optimization / known candidate constraints | `31_acquisition_optimization.ipynb` |
| Batch / async / fantasization | `32_batch_async_fantasization.ipynb` |
| Acquisition initialization | `33_acquisition_initialization.ipynb` |
| TuRBO / trust region | `34_turbo_trust_region.ipynb` |
| Robust input perturbation | `35_robust_input_perturbation.ipynb` |
| Sequential -> batch -> async integrated workflow | `36_end_to_end_workflows.ipynb` |
| Unknown outcome / black-box constraints | `37_outcome_constrained_bo.ipynb` |
| Binary classification active learning | `39_classification_active_learning.ipynb` |

The complete model-family notebook list remains in `examples/README.md`.

## Documentation ownership

Use the documentation layers for different questions:

- `docs/theory/`: mathematical assumptions, derivations, and conceptual boundaries;
- `docs/models/`: model selection, public model behavior, and model-specific limitations;
- `docs/optimization/`: acquisition and candidate-search runtime contracts;
- `examples/notebooks/`: executable user workflows;
- `tests/`: focused correctness and compatibility contracts.

A notebook is integration documentation, not a substitute for theory or tests.

## Authoring and coverage contract

The authoring contract is
[`notebook-authoring-standard.md`](notebook-authoring-standard.md).

The information architecture is
[`notebook-executable-documentation-architecture.md`](notebook-executable-documentation-architecture.md).

Notebook metadata records what the notebook actually demonstrates. A missing layer does not mean
the library lacks that capability.

Representative workflows are preferred over a Cartesian product of model x acquisition x optimizer
combinations. Add a dedicated notebook when a combination introduces a distinct user-facing
semantic contract.

## Execution contract

The CI policy is defined in [`notebook-ci.md`](notebook-ci.md).

Normal pull requests execute the default deterministic suite. The full suite additionally executes
slow and specialist notebooks through the scheduled / manually dispatched workflow.

Every committed notebook must belong to exactly one execution-policy group in
`examples/run_notebooks.py`.

## Important responsibility boundaries

The suite intentionally keeps these concepts separate:

- model-side dimensionality reduction vs candidate-search geometry;
- robust observation modeling vs input-perturbation risk;
- known candidate constraints vs unknown outcome constraints;
- MultiTask structure vs generic multi-output utility;
- batch BO vs asynchronous pending points vs model fantasization;
- posterior transforms vs MC objectives;
- surrogate models vs stateful trust-region search strategies.

These boundaries are part of the executable-documentation contract because collapsing them would
produce examples that run while teaching the wrong API semantics.

## Maintenance rule

When a public runtime contract changes:

1. update focused tests;
2. update the relevant theory or runtime guide when semantics changed;
3. update notebook metadata only for layers actually demonstrated;
4. execute the affected notebook from a clean kernel;
5. keep the notebook on a public robotorchan / BoTorch API path;
6. update navigation when a new distinct workflow is added.

Historical phase records belong in Git and pull-request history. This document describes the
current durable notebook system.
