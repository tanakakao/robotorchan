# Mathematical verification: Phases 7–9

## Purpose and evidence rules

The mapping inventories in `models.md`,
`acquisition.md` and `optimization.md` are
static evidence. Mathematical verification
requires independently specified invariants,
controlled numerical examples and test assertions.
A passing CI job, registry entry, or matching
tensor shape is not sufficient.

| Phase | Mathematical object | Verification ownership |
| --- | --- | --- |
| 7 | Model posterior, likelihood, covariance and input/output transforms | Model tests and independent analytic or reference checks |
| 8 | Acquisition values, Monte Carlo estimators, objectives and constraints | Acquisition tests and independent reference values |
| 9 | Candidate optimization, feasibility, search state and coordinate mappings | Optimizer/strategy tests and reference optima or invariants |

For every test, record the model/method ID,
formula or reference, assumptions, input dtype
and device, random seed, numerical tolerance,
observed result and the precise property asserted.
Use **Verified** only for that property, not
for a whole model family.

## Phase 7 — model verification matrix

| Model family | Independent mathematical assertion | Suggested reference and failure signal |
| --- | --- | --- |
| Exact single-output GP | For noiseless fixed hyperparameters, posterior mean and covariance agree with direct kernel conditioning `m*=K_*X(K_XX+Sigma)^-1(y-m_X)+m_*`, `C*=K_**-K_*X(K_XX+Sigma)^-1K_X*`. | Small float64 dataset, explicit Cholesky solve, diagonal covariance nonnegative within tolerance; test actual posterior values, not just shape. |
| Mixed categorical GP | Permuting category IDs consistently preserves predictions when the kernel represents equality rather than ordinal distance. | Compare fitted kernel matrices under category relabeling; detect accidental one-hot or numeric-distance assumptions. |
| Multi-task / Kronecker | Task-indexed covariance blocks match the stated task and input kernels; flattened ordering agrees with posterior output. | Small 2-task hand-constructed covariance and task-label permutation. Do not infer Kronecker exactness for variational approximations. |
| Multi-fidelity | Kernel cross-fidelity covariance and fidelity feature indexing match the source-defined model, with fixed training data. | Compare explicit block covariance for two fidelity levels. |
| Reduced input / latent models | Public-space prediction equals the documented transform followed by model evaluation, for a frozen fitted transform. | Deterministic transform and reconstruction; check train/test transform consistency and categorical-coordinate preservation. |
| Heteroskedastic / robust models | Predictive noise, latent epistemic uncertainty and observation variance are not conflated. | Controlled two-noise-level example and likelihood-specific reference; do not require homoskedastic analytic formulas of non-Gaussian likelihoods. |
| Binary GP classification | Latent posterior is distinct from predictive class probability; `predict_proba` stays in [0,1], and Bernoulli likelihood integrates the link over latent uncertainty. | Fixed latent Gaussian distribution with numerical quadrature or sufficiently accurate Monte Carlo; test 0/1 label semantics and calibration separately. |
| Non-GP surrogates | Predictive mean and uncertainty match each adapter's declared estimator, not GP posterior identities. | Frozen estimator with reference predictions; mark unsupported uncertainty explicitly. |

### Initial Phase 7 execution gate

1. Inspect existing tests for **actual value-level**
   posterior/covariance assertions versus shape
   and smoke tests. Link exact assertions before
   claiming coverage.
2. Implement an independent float64 exact-GP
   conditioning oracle on a small synthetic
   dataset and compare against the exposed
   posterior using fixed kernel hyperparameters.
   Keep model training out of this identity test.
3. Add one categorical-kernel relabeling invariant
   and one classification latent-vs-probability
   check only after the relevant source contract
   is verified.
4. Run focused tests and repository CI, record
   tolerances and failures, then expand to
   multi-task, fidelity, reduced and robust
   families without changing the fixed phases.

**Current status:** Phase 7 started; no
independent numerical result is claimed by
this document. The tests and execution
evidence above are pending.
