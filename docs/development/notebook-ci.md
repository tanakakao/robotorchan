# Notebook CI

## Purpose

Notebook CI validates executable documentation without making every pull request pay the cost of
the heaviest examples.

The execution policy has two complementary workflows.

## Pull-request tier

The `notebooks` job in `.github/workflows/ci.yml` runs the default notebook manifest on Python
3.11 CPU for every pull request and push to `main`.

This tier is the merge gate for the ordinary executable examples. It uses:

```text
python examples/run_notebooks.py --timeout 300
```

The default set contains the canonical BO / AL cross-layer workflows and should remain deterministic
and reasonably bounded.

## Full-suite tier

`.github/workflows/notebook-full.yml` executes the complete notebook inventory:

```text
python examples/run_notebooks.py \
    --include-slow \
    --include-excluded \
    --timeout 600
```

It runs:

- manually through `workflow_dispatch`;
- weekly on the scheduled workflow.

The schedule uses UTC. Its purpose is regression detection rather than a pull-request merge gate.

The full environment installs the example, Fully Bayesian, and NGBoost extras so specialist
notebooks can execute in the same clean environment.

## Manifest semantics

`examples/run_notebooks.py` has three explicit groups:

- `DEFAULT_NOTEBOOKS`: normal PR CI;
- `SLOW_NOTEBOOKS`: omitted from normal PR CI because of runtime;
- `EXCLUDED_NOTEBOOKS`: omitted from normal PR CI for specialist / benchmark execution policy.

The names describe CI policy, not support status. Both slow and excluded notebooks are included by
the full-suite workflow.

The manifest validator requires every committed notebook to belong to exactly one group.

## Failure policy

A default-notebook failure blocks ordinary CI.

A full-suite failure should be investigated as an executable-documentation regression. A notebook
may move between tiers when runtime or optional-dependency characteristics change, but it should
not become undeclared to avoid a failure.

## Design rationale

This split keeps fast feedback on every pull request while ensuring that heavy and specialist
notebooks are not permanently unexecuted.

Notebook CI complements unit and integration tests. It validates the user-visible sequence of
imports, fitting, posterior use, acquisition construction, candidate generation, and state updates;
it does not replace focused tests for individual contracts.
