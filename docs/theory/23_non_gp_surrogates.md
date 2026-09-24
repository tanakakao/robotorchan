# Non-GP surrogates for Bayesian optimization

## 1. Posterior interface without a Gaussian process

Bayesian optimization requires a predictive distribution or a sampleable predictive representation;
it does not require every surrogate to be a Gaussian Process. For an empirical ensemble with
predictions \(f_s(X)\), robotorchan represents

\[
\{f_1(X),\ldots,f_S(X)\}
\]

as an `EnsemblePosterior` with shape `... x S x q x m`. Here \(S\) is the ensemble dimension,
\(q\) the candidate batch size, and \(m\) the number of outputs. Monte Carlo acquisition functions
can sample from this discrete empirical distribution.

The empirical mean and variance summarize ensemble predictions, but their interpretation depends on
how the ensemble was generated. They are not automatically a Bayesian posterior mean and variance.

## 2. Random Forest and Extra Trees

For Random Forest and Extra Trees, a fitted tree is used as one empirical prediction member. Random
Forest obtains diversity from bootstrap/data sampling and feature selection; Extra Trees adds stronger
split randomization. The resulting tree disagreement can be useful as an epistemic heuristic, but it
does not by itself identify observation noise or guarantee calibrated credible intervals.

## 3. Why boosting stages are not posterior samples

A gradient boosting predictor has additive form

\[
F_M(x)=F_0(x)+\sum_{j=1}^{M}\eta h_j(x).
\]

The stage predictors \(h_j\) are trained sequentially to correct residual structure. They are not
exchangeable draws from a predictive distribution. Treating their spread as posterior uncertainty
would therefore give the ensemble dimension the wrong statistical meaning.

robotorchan instead draws bootstrap datasets \(D_s\), fits a complete boosting model \(F^{(s)}_M\)
on each dataset, and uses

\[
\{F^{(1)}_M(X),\ldots,F^{(S)}_M(X)\}
\]

as empirical predictive samples.

## 4. Multi-output samples

For \(m>1\), each bootstrap member produces a vector prediction. Outputs belonging to the same
bootstrap member retain a shared resampling index, while output-specific regressors can still be
fitted independently. This preserves member alignment but should not be described as a fully
Bayesian cross-output covariance model.

## 5. Acquisition functions

Because the posterior is sampleable, MC acquisitions such as qEI and qEHVI can consume it through
the BoTorch model/posterior interface. Analytic Gaussian acquisitions are not generally justified.

sklearn tree and boosting prediction is piecewise/non-differentiable with respect to candidate input
and crosses a CPU/numpy boundary. Acquisition maximization therefore uses gradient-free candidate
search rather than `optimize_acqf`'s gradient-based path.

## 6. Constraints

Constraint handling belongs at the acquisition/objective layer. If output \(g(x)\) represents a
constraint, feasibility can be defined by a callable such as \(g(x)\leq0\) while the surrogate
continues to model outputs. This separation permits the same fitted surrogate to support different
decision policies.

## 7. Distributional surrogate: NGBoost

`NGBoostSurrogate` differs from empirical tree ensembles. It predicts a conditional probability
distribution rather than treating boosting stages as posterior members. The initial robotorchan
adapter uses a Gaussian NGBoost predictive law and exposes its mean, variance, and samples through
the BoTorch posterior interface.

This predictive distribution represents total conditional predictive uncertainty. It must not be
called a Gaussian Process posterior, and it does not by itself provide the epistemic / aleatoric
decomposition required by BALD. MC BO and moment-based regression Active Learning can use the
sampleable distribution when their assumptions are satisfied; analytic GP acquisition compatibility
must not be inferred merely from Gaussian marginal predictions.

## 8. Calibration and benchmarking

An empirical ensemble standard deviation is useful only after checking its behavior on the target
problem. Useful diagnostics include predictive RMSE, interval coverage after defining an interval
construction rule, BO simple regret, and computational cost. A deterministic predictive benchmark can provide regression diagnostics, but it must not claim that
raw ensemble spread is calibrated without a dedicated calibration study.
