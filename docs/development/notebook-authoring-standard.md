# Notebook authoring standard

This document defines the authoring contract for executable documentation under
`examples/notebooks/`. The information architecture is defined separately in
[notebook-executable-documentation-architecture.md](notebook-executable-documentation-architecture.md).

## Required notebook metadata

Every new or substantially revised notebook must define these keys in top-level notebook metadata:

```json
{
  "robotorchan": {
    "tier": "A",
    "kind": "getting_started",
    "theory": [
      "docs/theory/01_bayesian_optimization.md"
    ],
    "runtime_guides": [
      "docs/getting_started/model_selection.md"
    ],
    "ci": "default",
    "layers": [
      "model",
      "fit",
      "posterior",
      "acquisition",
      "optimization",
      "candidate",
      "evaluation"
    ]
  }
}
```

Allowed values:

- `tier`: `A`, `B`, or `C`
- `kind`: `getting_started`, `model`, `acquisition`, `optimization`, or `workflow`
- `ci`: `default`, `slow`, or `manual`

`theory` and `runtime_guides` contain repository-relative paths. `layers` records what the
notebook actually demonstrates; it is not a claim about library-wide support.

Supported layer labels are:

```text
data
model
fit
posterior
sampling
posterior_transform
objective
acquisition
initialization
optimization
constraints
pending
fantasy
candidate
evaluation
loop
visualization
```

Existing notebooks may be migrated incrementally. The metadata contract becomes mandatory when a
notebook is substantially revised under the executable-documentation program.

## Common opening

Every notebook starts with:

1. a descriptive title,
2. a short statement of the problem being solved,
3. "このNotebookで確認すること",
4. links to the smallest relevant theory and runtime guide,
5. an explicit statement of the expected pipeline.

For a Tier A notebook, show the pipeline near the top:

```text
Data
 -> Model
 -> Fit
 -> Posterior
 -> Acquisition
 -> Optimization
 -> Candidate
 -> Evaluation
```

Add Sampling, Objective, PosteriorTransform, constraints, pending points, or fantasization only
when they are actually part of that example. Do not imply that every acquisition requires an
explicit sampler or objective.

## Standard section order

Use the following order when the sections apply:

1. Overview
2. Theory and documentation
3. Imports and reproducibility
4. Problem setup
5. Training data
6. Model construction
7. Model fitting / inference
8. Posterior inspection
9. Sampling
10. Objective / PosteriorTransform
11. Acquisition function
12. Acquisition initialization
13. Acquisition optimization
14. Candidate evaluation
15. Visualization
16. Full BO / AL loop
17. When to use this method
18. Limitations and caveats
19. Related robotorchan components
20. References

Sections that are semantically irrelevant should be omitted rather than filled with boilerplate.

## Tier-specific requirements

### Tier A

Tier A is an end-to-end executable example.

Required:

- problem and optimization direction are explicit,
- training data are reproducible,
- model construction and fitting/inference are shown,
- posterior behavior is inspected,
- acquisition semantics are stated,
- candidate generation is executed,
- the generated candidate is evaluated or its next-step meaning is demonstrated,
- theory and runtime-guide links are present.

A short full loop is preferred when runtime permits. If a loop is omitted, state why.

### Tier B

Tier B teaches a model family.

Required:

- model-specific motivation,
- input/output shape and raw-data contract,
- fitting/inference contract,
- posterior semantics,
- important model-specific caveats,
- link to the nearest compatible Tier A workflow.

Candidate generation may be included, but should not duplicate a workflow notebook solely to satisfy
a checklist.

### Tier C

Tier C covers specialist semantics.

Required:

- explain why the ordinary Tier A contract is incomplete or inappropriate,
- demonstrate the model's real inference/posterior contract,
- state intentionally omitted layers,
- link to relevant theory and compatibility documentation.

Do not fabricate a standard Gaussian-regression workflow for preference, structured-output,
non-Gaussian, or external posterior models.

## Code conventions

Notebook code follows repository code quality conventions.

- Prefer public robotorchan and BoTorch APIs.
- Use `torch.double` for ordinary GP examples unless the model requires otherwise.
- Set deterministic seeds near the imports.
- Keep examples network-independent.
- Keep default-CI examples CPU-friendly.
- Avoid hidden execution state: Restart Kernel + Run All must work.
- Do not define a large notebook helper framework.
- Keep helper functions local and small when they improve readability.
- Keep code lines at or below 100 characters where practical.
- Use real newlines; never encode a line break as the literal text `/n`.
- Avoid unnecessary blank lines.
- Use Matplotlib for lightweight plots unless another dependency is justified.

## Public API and BoTorch-first rule

Notebook code is user-facing API documentation.

Use robotorchan public APIs when robotorchan owns the contract. Use standard BoTorch APIs directly
when robotorchan intentionally delegates the contract to BoTorch. A notebook must not introduce a
wrapper merely to make an example look uniform.

Examples include using BoTorch acquisition optimization directly when that is the supported public
path, while using robotorchan model constructors and model-owned contracts where robotorchan adds
behavior.

Private imports are prohibited unless a notebook is explicitly a development/debugging notebook,
which user-facing examples are not.

## Theory discipline

Do not duplicate a theory chapter inside a notebook.

The notebook should contain only enough mathematics to explain the executable step. Link to the
canonical theory chapter for derivations, assumptions, and references.

Use the same terminology and optimization direction as the theory documentation. If implementation
and theory disagree, fix the source-of-truth mismatch instead of documenting a notebook workaround.

## Candidate-generation discipline

Candidate generation is not equivalent to constructing an acquisition function.

When a notebook claims candidate-generation coverage, it must execute the relevant optimizer or
search strategy and expose the resulting candidate shape/value.

For mixed, constrained, multi-fidelity, high-dimensional, trust-region, or derivative-free
workflows, use the optimizer appropriate to that search space. Do not force `optimize_acqf` into
an incompatible workflow.

## Full-loop discipline

A full loop must make these state transitions visible:

```text
fit / update model
 -> build acquisition
 -> generate candidate
 -> evaluate candidate
 -> append observation
 -> repeat
```

A tiny number of iterations is sufficient for documentation. Benchmark-quality performance
comparisons belong in `benchmarks/`.

## Outputs and execution state

Committed notebooks should avoid noisy or machine-specific outputs. Examples must not depend on
previously stored outputs for correctness.

Execution validation is performed by `examples/run_notebooks.py`; the exact committed
`execution_count` policy may remain unchanged until notebook migration tooling is introduced.

## Review checklist

Before merging a new or substantially revised notebook, verify:

- metadata matches the demonstrated layers,
- theory and runtime-guide paths exist,
- imports are public,
- Restart Kernel + Run All succeeds,
- seed/dtype/device choices are explicit where relevant,
- tensor shapes are understandable,
- optimization direction is unambiguous,
- candidate generation is real when claimed,
- special model semantics are not hidden,
- no benchmark logic is duplicated,
- no literal `/n` newline mistakes,
- no unnecessary blank lines,
- code lines are kept within the repository's 100-character convention where practical.
