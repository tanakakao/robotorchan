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

**Current status:** Phase 7 has independent mathematical reference tests for
selected model properties. The coverage is property-specific; see the
execution reconciliation and outstanding gaps below.


### Kronecker block-design covariance reference (Phase 7)

`tests/models/standard/test_phase7_kronecker_covariance_reference.py`
checks the covariance of `KroneckerMultiTaskGP` with a fixed
three-input, two-task float64 example on CPU. For input-major flattening,
`K[i*m+a, j*m+b] = K_input[i,j] * K_task[a,b]`.
The independent scalar-index reference checks every entry, symmetry,
positive semidefiniteness (minimum eigenvalue >= -1e-10), and task
permutation. Entry comparisons use atol=rtol=1e-10; no random seed
or parameter fitting is involved. This covers the *latent prior
covariance structure*, not posterior conditioning, learned
hyperparameters, observation noise, or other Kronecker variants.
The corresponding PR passed CI and documentation-link checks and was merged.


### Data-fidelity downsampling kernel reference (Phase 7)

`tests/models/standard/test_phase7_multifidelity_kernel_reference.py`
checks the native data-fidelity kernel in
`MixedSingleTaskMultiFidelityGP` with fixed float64 inputs and no
hyperparameter fitting. For `K_f(s,t)=offset+c(1-s)^p(1-t)^p`, the
high-fidelity boundary `s=1` leaves the constant offset, and the
residual covariance after subtracting that offset is rank one.
The test checks this boundary, symmetry, a residual rank-one identity, and positive
semidefiniteness (atol=rtol=1e-10). This does not verify the complete
multi-fidelity posterior, the linear-truncated kernel, or learned
hyperparameters. The corresponding PR passed CI and documentation-link checks and was merged.


### PCA reduced-space projection reference (Phase 7)

`tests/models/high_dimensional/reduced/test_phase7_pca_projection_reference.py`
checks the `PCAInputReducer` used by reduced GP models on a fixed
six-sample, three-feature float64 dataset. An independently computed
centered covariance eigenspectrum is compared with the reducer's
explained variance and projected sample covariance. The test also
checks basis orthonormality, projection reconstruction, and residual
orthogonality (atol=rtol=1e-10 where specified). PCA basis signs
are intentionally not fixed. This validates the input reduction
geometry, not the full reduced GP posterior or PLS/neural reducers.
The corresponding PR passed CI and documentation-link checks and was merged.


### Replicate observation-noise statistical reference (Phase 7)

`tests/models/robust/test_phase7_replicate_noise_reference.py`
checks the unbiased within-group sample variance
`s² = sum((y_i - mean(y))²) / (n - 1)` and the observation-noise
variance of each group mean `s²/n`. The fixed float64 dataset
contains interleaved groups with unequal replicate counts. The test
independently calculates group statistics and checks the aggregated
raw targets, replicate counts, replicate variance, and raw observation
variance. It does not establish the accuracy of a learned
heteroskedastic noise process or a robust GP posterior.
The corresponding PR passed CI and documentation-link checks and was merged.


### Binary GP probit predictive reference (Phase 7)

`tests/models/classification/test_phase7_binary_probit_reference.py`
independently evaluates the Gaussian-probit integral
`E[Phi(f)] = Phi(mu / sqrt(1 + sigma²))` for a latent Gaussian
posterior `f ~ N(mu, sigma²)`. It compares the closed-form
positive-class probability against `predict_proba` and verifies
the complementary negative-class probability and normalization
on fixed float64 candidates. This checks the probit predictive
probability mapping, not ELBO convergence, calibration quality,
or the quality of the latent GP fit. The corresponding PR passed CI and documentation-link checks and was merged.


### Non-GP Gaussian posterior reparameterization reference (Phase 7)

`tests/models/non_gp/test_phase7_gaussian_posterior_reference.py`
verifies the BoTorch-compatible `GaussianDistributionPosterior`
used for distribution-aware non-GP surrogates. Fixed float64
base samples are mapped using the independent affine Gaussian
identity `y = mu + sqrt(variance) * epsilon`. The test covers
batch and sample dimensions, a zero-variance deterministic
component, exact mean/variance accessors, and gradient propagation
through the mean. This is a posterior adapter contract check,
not a claim about the predictive accuracy of external estimators.
The corresponding PR passed CI and documentation-link checks and was merged.


## Phase 7 execution reconciliation and remaining scope

The following merged pull requests had successful CI and documentation-link
workflows on their final feature-branch heads (Python 3.11/3.12/3.13 CI).
This records **test execution**, not proof of every model-family property.

| Mathematical reference | Merged PR | Assertion scope |
| --- | --- | --- |
| Exact GP posterior conditioning | Earlier Phase 7 reference | Fixed-parameter posterior values |
| Mixed categorical relabeling | Earlier Phase 7 reference | Categorical covariance invariance |
| Multi-task covariance | #1416 | Task-indexed kernel blocks |
| Kronecker covariance | #1417 | Input/task product structure |
| Multi-fidelity downsampling | #1418 | Offset boundary and residual rank one |
| PCA reduction | #1419 | Orthogonal projection and eigenvalues |
| Replicate noise | #1420 | Unbiased variance and variance of mean |
| Binary GP probit | #1421 | Analytic class probability |
| Non-GP Gaussian adapter | #1422 | Gaussian reparameterization and gradients |

**Phase 7 completion assessment: not yet complete.** The original matrix
requires more than one property per broad model family. The current tests
do not yet establish the following representative properties:

- Reduced GP: posterior equivalence between public input coordinates and
  the frozen latent transform, including categorical coordinate retention.
- Heteroskedastic GP: explicit separation of latent uncertainty and
  observation noise in a controlled two-noise-level example.
- Non-GP estimators: agreement between a frozen external estimator's
  predictions and the adapter's exposed mean/uncertainty, including
  explicitly unsupported uncertainty.

The next work should address these gaps within **Phase 7**, without
expanding the fixed 12-phase plan. Passing CI for this reconciliation
alone must not be used to mark Phase 7 complete.


### Reduced GP public/latent posterior coordinate reference (Phase 7)

`tests/models/high_dimensional/reduced/test_phase7_posterior_coordinate_reference.py`
compares the PCA-reduced GP's posterior mean and variance when candidates
are supplied in the original input coordinates and in the frozen PCA
coordinates `T(x)`. The identity is
`posterior_original(x) = posterior_reduced(T(x))` for the **same**
trained GP and fixed reducer. The float64 test covers q-batch inputs
and checks that posterior evaluation does not mutate PCA components
or centering. This establishes coordinate-path consistency, not
the quality of the dimensionality reduction or categorical retention.
Execution evidence is pending CI.


### Heteroskedastic latent-versus-observation noise reference (Phase 7)

`tests/models/robust/test_phase7_heteroskedastic_uncertainty_reference.py`
uses a deterministic two-level log-variance surrogate (0.01 and 0.09
on the variance scale) to check `predicted_noise(x) =
exp(noise_posterior(x).mean)` and to distinguish that path from the
response GP's latent posterior variance. The response GP is not
refitted, and the controlled noise surrogate is a test double rather
than a trained model. This does **not** establish calibrated
heteroskedastic uncertainty or a correct total observation
posterior at arbitrary candidate inputs. Execution evidence is
pending CI.


### Non-GP Random Forest estimator-level reference (Phase 7)

`tests/models/non_gp/test_phase7_forest_estimator_reference.py`
compares the public ensemble posterior with independently collected
predictions from every fitted scikit-learn tree, using fixed float64
q-batch candidates and `random_state=17`. It checks each sample,
the ensemble population mean and variance (denominator `s`, not
`s-1`), and explicit rejection of unsupported observation noise.
This verifies Random Forest adapter semantics, not generalization
accuracy, epistemic calibration, or other non-GP estimator families.
Execution evidence is pending CI.
