# Model design and implementation guidelines

## Purpose

These rules apply when adding or changing robotorchan model wrappers. Current
`main` and current upstream BoTorch semantics are authoritative. Historical PRs
are design evidence only.

## API migration

Do not add compatibility APIs, compatibility functions, old-name aliases,
deprecated wrappers, or monkey patches. When a public contract changes, migrate
the implementation, exports, tests, examples, notebooks, and documentation to
the new contract in the same phase.

## Mixed-model semantics

Mixed models accept raw mixed-space inputs and `cat_dims`. Structural columns
such as task, fidelity, context, and hierarchy selectors are not ordinary
categorical design variables and must be excluded or rejected on overlap.
Prefer native categorical covariance when it preserves the model mathematics.
Use model-owned categorical encoding only when native covariance would change
the upstream model's inference, prior, or structural semantics.

## Mandatory pre-PR validation

Before opening a PR, perform a repository-wide stale-contract check for every
public symbol or semantic decision changed by the phase. In particular:

1. Search all tests for assertions that a newly introduced model or API must
   not exist. Feasibility-gate tests from earlier phases must be migrated or
   removed once their gate is deliberately closed.
2. Search all tests, examples, notebooks, docs, and imports for superseded
   constructor arguments, module paths, aliases, and negative exposure tests.
3. Classify tests by dependency requirements. Tests that instantiate optional
   models must run only in the CI job that installs those optional dependencies.
   Fully Bayesian tests requiring JAX / NumPyro belong in
   `fully-bayesian-extra`, not the core matrix.
4. Run or reproduce the repository's exact lint and format gates before the PR:
   `ruff check .` and `ruff format --check .`. Do not rely on visual formatting.
5. Check public `__all__` ordering and the public model contract tests whenever
   exports change.
   Treat the export list and `tests/models/test_public_model_contracts.py` as
   one atomic public-API change: every newly exported model must be added to
   `PUBLIC_MODEL_NAMES` in the same commit/phase, and removed models must be
   removed there at the same time.
6. Verify that every changed semantic gate has a positive replacement test.
   Do not leave contradictory old tests in place.
7. For every new or edited Python file, run Ruff's import organization and
   formatter over the final file state. In particular, do not hand-tune blank
   lines after import blocks: `ruff check .` and `ruff format --check .` are the
   authoritative contract.

This stale-contract sweep is mandatory because robotorchan develops in phases:
an earlier phase may intentionally add a negative contract test that a later
phase is expected to invalidate.

## Python line-length and formatting discipline

The repository line-length contract is **100 characters maximum** (`E501`).
Treat this as a construction rule, not as a CI cleanup task. Every new or edited
Python line must be kept at or below 100 characters while writing the change.

Before the final commit of every phase that touches Python:

1. Sweep every changed Python file for lines longer than 100 characters.
   Long `assert`, constructor, function-call, type-annotation, and comprehension
   lines are common failure points and must be wrapped proactively.
2. Do not assume that Ruff's formatter will repair every `E501` violation.
   Ruff formatting and Ruff linting are separate gates; a formatter-compatible
   line can still fail `ruff check .` for `E501`.
3. After line-length cleanup, validate the **entire final changed-file set**,
   not only the line reported by the previous CI run. A repeated CI cycle caused
   by fixing one of several equivalent `E501` violations is considered an
   incomplete preflight.
4. Run the equivalent of both `ruff check .` and `ruff format --check .` on the
   final branch state. When remote connector editing cannot execute Ruff,
   manually inspect all changed Python lines for the 100-character limit before
   relying on CI.
5. If CI reports an `E501`, inspect the surrounding changed file for additional
   lines with the same pattern and repair them together.

## Python 3.11 preflight

Python 3.11 is the repository's lint/format gate, not merely another test
version. Any phase that changes Python files must treat the 3.11 preflight as a
separate mandatory check before PR creation or update: inspect the final branch
state after all edits, then run the equivalent of `ruff check .` followed by
`ruff format --check .`. Export changes additionally require isort-style
`__all__` ordering. A green 3.12/3.13 test result does not imply that the 3.11
job will pass, because those jobs do not execute these lint/format steps.

When a connector or remote editing workflow cannot execute Ruff locally, inspect
Python 3.11 CI first after the initial push and apply Ruff's exact suggested
formatting before treating the phase as complete. Formatting-only failures are
not version-specific model failures and must not be debugged independently in
3.12/3.13.

## CI interpretation

Do not report a phase as successful while required jobs are queued or running.
For failures, inspect the actual job log and fix the root cause. Required jobs
depend on the touched model family: core Python 3.11/3.12/3.13 always applies;
package applies to public/export changes; notebooks applies to public examples
or namespace changes; `fully-bayesian-extra` is mandatory when fully Bayesian
models or their tests change.
