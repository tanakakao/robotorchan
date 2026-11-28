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

## Literature and upstream verification still required
Starting references: Rasmussen & Williams (2006), Wilson & Adams (2013), Hartikainen & Särkkä (2010), Frigola et al. (2014), Bui et al. (2017), Bogunovic et al. (2016), Balandat et al. (2020). Verify originals, applicable assumptions, upstream public API/version and licenses before marking Phase 1 complete. No runtime or CI evidence is asserted by this record.
