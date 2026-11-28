# Temporal extension: architecture and implementation contract

Baseline: 3b0ed048a6dcf5801f91586c034ed3e273c31ec9. This is a Phase 1 design record, not a claim of runtime compatibility. The 19-phase plan and additional cross-cutting requirements remain in force.

## Architecture
Use existing models/standard, structured, robust, expressive, classification, acquisition/active_learning, optim/strategies, optim/trust_region, study and workflow. Introduce temporal/ for tensor time-axis, lag, splitting, forecast and update utilities, and models/temporal/ only for genuinely new model/inference implementations. Separate files by responsibility as implementations grow; no empty packages, duplicate optimizer or registry, speculative abstractions, monkey patches or new compatibility aliases. Keep low-level temporal data utilities independent of models, acquisition and workflow; model-specific forecast/update adapters may depend on model interfaces but models must not import those adapters. Workflow and study consume lower layers, never the reverse. Domain-specific alarms, RUL, controllers and materials workflows remain downstream. No new mandatory dependency without demonstrated need; CPU and optional-dependency-free imports must work.

## Evidence from baseline source
- models/standard/multitask.py: MultiTaskGP accepts covar_module; KroneckerMultiTaskGP accepts data_covar_module. Mixed variants construct their own covariance.
- models/structured/latent_kronecker.py: separate covar_module_X/T, posterior(X,T), fit; do not assume arbitrary asynchronous observations or generic acquisition compatibility.
- models/classification/{binary,multiclass,ordinal}/standard/single_task.py: custom covar_module and probability APIs; multiclass has explicit fit and latent posterior. Reuse these, distinguish latent and probability distributions; mixed variants construct kernels.
- models/robust/replicate_noise.py: replicate aggregation and empirical variance, not evidence of future heteroskedastic prediction.
- optim/contextual.py: ContextualOptimizer.optimize and extract_decisions.
- models/factory.py: constructor discovery and registry resolution.
- study/decoupled.py: decoupled proposer and budget handling.
- workflow/problem.py: ProblemSpec and task/observation/objective semantics.
These are static source findings, not executed tests.

## Contracts
Tensor-first; time dimension normalized to nonnegative index; consistent units and origin fitted only on training history. Preserve original timestamps where needed. Separate observation, availability, decision, evaluation start/end and target time; distinguish controllable duration. No future observations or fitted preprocessing in historical decisions. Handle missingness explicitly, do not silently interpolate across split boundaries. Preserve dtype/device, batch and q semantics. Use existing fit/posterior APIs and public BoTorch conditioning where supported; distinguish conditioning from hyperparameter refit. New posterior implementations must document mean, variance, joint sampling, base samples and differentiability support; no blanket acquisition compatibility. Forecasts distinguish latent versus observation noise and known versus unknown future covariates. Joint trajectory samples must preserve dependence. Additive GP components require cross-component posterior covariance; for independent priors and Gaussian observations cross covariance is -K_i(*,X) K_y^{-1} K_j(X,*). Use solves, not explicit inverse. Trend is not necessarily physical degradation.

## Implementation roadmap and gates
2: time conversion, irregular sampling, masks, lags, series-aware splits; leakage tests.
3: reuse native kernels and existing GP; only add missing mathematically valid kernels; fit/posterior/forecast/candidate tests.
4: additive component posterior, joint sampling and reconstruction; identifiable and confounded examples.
5: change point/window, nonstationarity, regime and generic change detection; PSD and online leakage tests.
6: NAR/NARX, distributed lag and delay; causal feature alignment.
7: linear state-space GP, filtering/smoothing, irregular discretization, exact Matérn cases and approximation labels; GPSSM separate nonlinear inference, no false Gaussian posterior guarantee.
8: multitask, LMC/ICM, spatiotemporal and Kronecker; asynchronous/missing outputs tested separately.
9: robust likelihood, heteroskedastic, replicate, uncertain input, censored and correlated noise; distinguish noise from latent variance.
10: sparse, SKI, Toeplitz, Kronecker, local and state-space scaling; benchmark accuracy and resource use.
11: refit, warm start, conditioning, windows, discounting, online variational and state-space update; validate objective and retained state.
12: direct, recursive and joint forecast, trajectory uncertainty, scenarios; compare mean plug-in versus distribution propagation.
13: binary, multiclass, ordinal, multitask, deep, high-dimensional and non-GP compatibility; test class probability normalization and temporal calibration.
14: fixed-time existing acquisition first; dynamic UCB/EI/TS, discount/window/restart, dynamic TR, forecast and lookahead; dynamic regret and time constraints.
15: temporal AL variance/entropy/BALD, integrated variance and information gain, forecast/change-oriented, cost and batch; distinguish archive queries from future measurements.
16: single/multiobjective, constrained, safe, robust, contextual, fidelity, cost, decoupled, transfer, causal, combinatorial, mixed, q and classification AL; safety guarantees conditional on drift assumptions.
17: existing factory, registry, capability, ProblemSpec, recommendation, study, DataFrame, storage/resume; preserve low-level single-candidate path.
18: rolling-origin baselines, proper scores, calibration, dynamic regret, cost, memory and runtime; equal budgets and multiple seeds.
19: Japanese theory/guide/notebooks, user-facing examples and final regression verification. Forecast-only notebooks need not propose candidates.

For each phase record existing verified / newly implemented / conditional / mathematically inappropriate / technically unsupported. Static candidate reuse is not verified support. Test direct dependencies and relevant cross-cutting combinations, not unrelated repository-wide audits. Add tests and docs with implementation; no audit-only completion for implementation phases. Keep PRs coherent by feature, avoid artificial micro-phases. No API or package skeleton added in Phase 1.

## Upstream API and literature decisions

The project metadata requires Python >=3.11 and botorch>=0.18.1,<0.19; pandas is already optional via dataframe. BoTorch 0.18 release notes require PyTorch >=2.4. Keep dependency metadata unchanged. Stable/latest documentation is evidence of public interfaces, not a substitute for tests against the supported version range.

- [BoTorch model API](https://botorch.readthedocs.io/en/stable/models.html): reuse SingleTaskGP, ModelListGP, MultiTaskGP and public condition_on_observations where supported. Conditioning returns a new model; validate robotorchan raw training data metadata and fixed-noise transforms before online reuse. MixedSingleTaskGP exposes cont_kernel_factory; verify actual behavior on supported versions rather than assuming all mixed wrappers accept custom kernels.
- [BoTorch releases](https://github.com/meta-pytorch/botorch/releases): OrthogonalAdditiveGP and LatentKroneckerGP already exist in the 0.18 series. Evaluate OrthogonalAdditiveGP for Phase 4 before creating a new model; its orthogonality assumptions are not interchangeable with arbitrary additive temporal decomposition.
- [GPyTorch kernel API](https://docs.gpytorch.ai/en/latest/kernels.html): use native PeriodicKernel, ScaleKernel and kernel products/sums, spectral mixture and grid interpolation. Preserve units of period_length and active_dims. Grid methods require suitable stationary kernels and structured inducing grids; irregular observations are not themselves a regular grid.

Primary design references checked at publisher abstract or paper level:
- [Wilson and Adams (2013)](https://proceedings.mlr.press/v28/wilson13.html), Gaussian Process Kernels for Pattern Discovery and Extrapolation: stationary spectral mixtures, Phase 3.
- [Hartikainen and Särkkä (2010)](https://doi.org/10.1109/MLSP.2010.5589113): temporal state-space GP, Phase 7. Half-integer Matérn admits finite exact representation; squared exponential requires approximation.
- [Särkkä and Hartikainen (2012)](https://proceedings.mlr.press/v22/sarkka12.html): spatiotemporal state-space inference with separability conditions, Phase 8.
- [Frigola et al. (2014)](https://proceedings.neurips.cc/paper_files/paper/2014/file/875b60cf697a9e09e12c0f07b982e431-Paper.pdf): nonlinear GPSSM with variational and sequential Monte Carlo inference, Phase 7; distinct from linear state-space GP.
- [Bui et al. (2017)](https://proceedings.neurips.cc/paper/2017/hash/f31b20466ae89669f9741e047487eb37-Abstract.html): streaming sparse posterior and hyperparameter updates, Phase 11.
- [Bogunovic et al. (2016)](https://proceedings.mlr.press/v51/bogunovic16.html): Markov time-varying GP-UCB, reset and gradual forgetting, Phase 14; regret claims require stated drift assumptions.

Use original implementations based on mathematical descriptions or existing installed public APIs, not copied third-party source. No new external code or datasets introduced; check license of any subsequently proposed code/data before adoption. Detailed derivations and exact-version runtime checks are acceptance gates of the respective implementation phases.

## Phase 1 disposition

Design baseline and upstream selection are established. No runtime support is claimed until implementation tests pass. Phase 2 can proceed after this contract PR passes CI and is merged. Phase 2 should deliver a coherent tensor-first time-axis, lag, missing-data and temporal split implementation with tests, rather than further planning-only PRs.
