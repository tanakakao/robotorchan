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
PR #1424 passed CI and documentation-link checks and was merged.


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
confirmed by PR #1425 CI and documentation-link checks after fixes; merged.


### Non-GP Random Forest estimator-level reference (Phase 7)

`tests/models/non_gp/test_phase7_forest_estimator_reference.py`
compares the public ensemble posterior with independently collected
predictions from every fitted scikit-learn tree, using fixed float64
q-batch candidates and `random_state=17`. It checks each sample,
the ensemble mean and BoTorch sample variance (denominator `s-1`), and explicit rejection of unsupported observation noise.
This verifies Random Forest adapter semantics, not generalization
accuracy, epistemic calibration, or other non-GP estimator families.
PR #1426 is merged; its final-head CI status was pending at the earlier audit.

## Phase 7 exit audit (property-specific)

The three follow-up references are tracked independently of the earlier
nine-reference reconciliation:

| Reference | PR | Execution evidence | Verified property | Explicit boundary |
| --- | --- | --- | --- | --- |
| PCA reduced posterior coordinate equivalence | #1424 | CI and docs checks passed; merged | Original/latent coordinate posterior mean and variance with frozen PCA | No categorical retention, output reducer, or non-PCA reducers |
| Heteroskedastic latent/noise separation | #1425 | CI and docs checks passed; merged | Two controlled log-noise levels and separate latent variance | No fitted noise-GP calibration or full observation posterior identity |
| Random Forest external estimator reference | #1426 | Merged; final-head CI pending at audit time | Individual sklearn tree predictions and ensemble sample moments | No other non-GP families or estimator uncertainty calibration |

**Exit decision: Phase 7 remains open pending #1426 final-head CI and
resolution or explicit acceptance of the matrix boundaries.**
In particular, categorical-coordinate preservation in reduced mixed
models and the heteroskedastic observation-posterior composition
remain unverified. Neither a green workflow nor a merged PR by itself
upgrades these properties to mathematical verification.

The fixed Phase 8 acquisition and Phase 9 optimizer verification scopes
are unchanged. This audit does not add phases or move model-specific
unverified properties into acquisition/optimizer verification.

### Mixed reduced GP categorical-coordinate reference (Phase 7)

`tests/models/high_dimensional/reduced/test_phase7_mixed_reduced_categorical_reference.py`
uses fixed float64 continuous/category inputs with the category column
interleaved among continuous features. It independently checks that
the reduced representation and training inputs retain the exact category
IDs, that the category is moved to the declared reduced categorical
index, and that original-coordinate and explicitly reduced-coordinate
posterior means/variances agree (atol=rtol=1e-10). This covers the
MixedPCAGP frozen PCA path, not all mixed reducers or categorical
generalization quality. PR #1428 passed CI and documentation-link checks and was merged.

### Heteroskedastic observation-variance composition boundary (Phase 7)

`tests/models/robust/test_phase7_heteroskedastic_uncertainty_reference.py`
also checks the *manually composed* conditional identity
`Var[y(x)|D] = Var[f(x)|D] + v(x)` for independent zero-mean
observation error with fixed conditional variance `v(x)`.
It uses the public latent posterior and separately exposed
`predicted_noise(x)` with controlled values 0.01 and 0.09
on float64 candidates. This is **not** a verification of
`posterior(observation_noise=True)`: the heteroskedastic wrapper
does not override that public method to inject candidate-specific
predicted noise. The log-noise GP posterior uncertainty is also
not integrated in this plug-in variance identity. Accordingly,
a complete heteroskedastic observation-posterior implementation
remains **Unverified/Unsupported** rather than being silently
marked Verified. PR #1429 passed CI and documentation-link checks and was merged.

## Phase 7 final exit reconciliation

**Decision: complete for the agreed representative, property-specific
mathematical verification scope.** The earlier exit audit above records
historical gaps; subsequent merged, green references #1428 and #1429
close the categorical-coordinate check and the *manual* conditional
observation-variance identity. The latter deliberately does not
establish candidate-specific `posterior(observation_noise=True)`
semantics; this remains **Unsupported/Unverified**, not a hidden
verification success. Tests do not establish broad numerical
correctness across every registered model, fitted calibration,
or all dimensionality reducers.

The exit evidence is the fixed float64 mathematical references and
their CI-backed execution, not mere source inventory or test presence.
Phase 8 can proceed to acquisition mathematical verification under
the unchanged twelve-phase plan. Phase 9 remains optimizer verification.

## Phase 8 — regression active-learning formula reference

`tests/acquisition/test_phase8_regression_al_formula_reference.py`
checks four local q=1 regression AL scores against separately written
posterior-moment formulas on fixed float64 candidates (CPU, no fitting):
`PosteriorVariance = Var[f]`,
`PosteriorStd = sqrt(max(Var[f],0))`,
`Straddle = beta * std - abs(mean - target)`, and
`BoundaryVariance = max(Var[f],0) * exp(-0.5*((mean-target)/max(std,eps))**2)`.
The reference uses the model posterior as an input; it verifies
acquisition arithmetic and batch-axis semantics, **not** independent
GP posterior conditioning (Phase 7), optimizer behavior (Phase 9),
or all acquisition families. Absolute/relative tolerance is 1e-12.
Execution evidence is pending CI.
