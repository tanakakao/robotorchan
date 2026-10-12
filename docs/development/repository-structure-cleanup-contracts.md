# Repository structure cleanup contracts

These contracts govern package boundaries and safe structural changes.
A directory listing alone is not evidence of complete dependency analysis
or successful runtime validation.

## Package boundaries

- `robotorchan.models`: surrogate models and model capabilities. Keep
  `models.heterogeneous.HeterogeneousModel` distinct from
  `models.structured.heterogeneous.HeterogeneousMTGP`; they serve different
  purposes.
- `robotorchan.acquisition`: acquisition functions, semantic composition,
  sampling, and optimizer compatibility checks.
- `robotorchan.optim`: candidate optimization backends, search strategies,
  domains, and candidate constraints.
- `robotorchan.semantics`: objective and constraint meaning, separate from
  model prediction and acquisition optimization.
- `robotorchan.benchmarks`: importable benchmark definitions, execution,
  metrics, persistence, and reports.
- Root `benchmarks/`: executable experimental scripts, not an importable
  package.
- `tests/`: regression tests organized by subsystem.
- `examples/notebooks/`: notebook examples with an explicit execution
  manifest in `examples/run_notebooks.py`.

## Cleanup rules

1. Preserve public import paths, registry behavior, and BoTorch-compatible
   low-level interfaces unless a separately reviewed breaking change is
   explicitly approved.
2. Never delete a module merely because static imports do not reference it:
   check dynamic registries, string imports, notebooks, scripts, tests, and
   documentation.
3. Do not merge persistence modules solely on similar names. Compare artifact
   schemas, validation rules, and round-trip compatibility first.
4. Treat `src/robotorchan/benchmarks/` subpackage proposals as optional;
   first prefer internal refactoring that does not move public modules.
5. When moving notebooks into `examples/notebooks/`, update documentation,
   tests, and the execution manifest together. Keep expensive benchmark
   notebooks in the manual execution group unless a separate CI budget review
   approves their inclusion.
6. Split cleanup into small pull requests with focused regression checks.
   Preserve the existing Python 3.11–3.13 CI, packaging checks, and
   notebook execution contracts.

## Validation required before closing the cleanup cycle

- Check Ruff lint and formatting, package build, and the relevant pytest
  suites on the proposed changes.
- Verify exported imports and registries still resolve.
- Verify benchmark persistence and report round trips if those files change.
- Record unresolved dynamic references and any unexecuted checks explicitly;
  do not claim full repository validation based on a tree listing.
