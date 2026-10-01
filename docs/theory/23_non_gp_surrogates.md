# Non-GP Surrogates for Bayesian Optimization

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
Forest obtains diversity from bootstrap/data sampling and feature selection;
Extra Trees adds stronger split randomization. The resulting tree disagreement can be useful as an epistemic heuristic, but it
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
construction rule, BO simple regret, and computational cost. A deterministic predictive benchmark can provide regression diagnostics,
but it must not claim that raw ensemble spread is calibrated without a dedicated calibration study.


## 9. 実装上の fitting contract

robotorchan の non-GP surrogate は BoTorch `Model` interface を提供しますが、GP marginal
likelihood を持ちません。constructor で raw training data を保持し、`fit()` を明示的に呼びます。

したがって GP の `make_mll()` / `fit_gpytorch_mll()` contract を適用しません。

## 10. HistGradientBoostingSurrogate

`HistGradientBoostingSurrogate` も complete bootstrap models を predictive members とします。
boosting stage の spread を posterior とするのではなく、独立 bootstrap resamples 上で fit した
complete `HistGradientBoostingRegressor` の predictions を使います。

## 11. Multi-output bootstrap contract

`BootstrapEnsembleSurrogate` は `train_Y` の `n x m` output を扱えます。multi-output の場合も
同一 ensemble member 内では同じ bootstrap row indices を共有します。

ただし output-specific regressors が独立に fit されるため、member alignment があることを
MultiTask GP / Kronecker GP の fully modeled cross-output covariance と同一視しません。

`output_indices` は valid range、non-empty、duplicateなしを要求します。

## 12. Mixed tree inputs の意味

Random Forest / Extra Trees は `cat_dims` の category values を検証します。

ただし sklearn tree backend が raw numeric category codes を native categorical variable として
扱うわけではありません。threshold split は数値コード上で行われます。

したがって

```text
cat_dims validation
!= native categorical split semantics
```

です。

## 13. Observation-noise semantics

現在の empirical tree / boosting surrogate は
`posterior(..., observation_noise=True)` による追加 noise modeling を提供しません。

ensemble spread を observation noise とみなして自動加算することもありません。

これは GP の latent-function posterior と likelihood noise の分離とは異なります。

## 14. IndexSampler と MC acquisition

`validate_non_gp_acquisition` は empirical non-GP model に `MCAcquisitionFunction` を要求します。

explicit sampler が設定される場合は `IndexSampler` を要求します。
これは continuous Gaussian base samples ではなく ensemble member indices を sampling するためです。

qEI / qEHVI など sample-based acquisition と接続できても、すべての acquisition が自動的に
互換になるわけではありません。

joint Gaussian、fantasize、multi-output、MultiFidelity などの追加要件は別途満たす必要があります。

## 15. Capability-based compatibility

robotorchan は model / acquisition registry の metadata から static compatibility を判定します。

non-GP model では特に、

- `non_gp`
- `ensemble_posterior`
- `supports_multi_output`
- `supports_posterior_samples`
- `posterior_sampling_type`
- `supports_fantasize`

が重要です。

BoTorch `Model` を継承していることだけから acquisition compatibility を推論しません。

## 16. TreeEnsembleSearchStrategy

sklearn tree / boosting prediction は candidate input に関して piecewise / non-differentiable で、
CPU / NumPy boundary も通ります。

robotorchan は `TreeEnsembleSearchStrategy` により gradient-free random candidate search を
提供します。

この strategy は、

- continuous bounds
- integer dimensions
- categorical allowed values
- non-GP MC acquisition validation

を扱います。

integer dimensions は feasible integer range、categorical dimensions は明示された allowed values
から sampling します。

これは GP の gradient-based `optimize_acqf` / `optimize_acqf_mixed` route とは別です。

## 17. Output constraints と candidate constraints

output constraint は objective / acquisition layer で扱います。

一方、bounds、integer dimensions、categorical allowed values のような candidate geometry は
search strategy が扱います。

surrogate、decision policy、candidate feasibility の責務を分離することが重要です。

## 18. NGBoost の現在の実装境界

現在の `NGBoostSurrogate` は、

- single-output
- Gaussian `Normal` predictive law
- explicit `fit()`
- sampleable `GaussianDistributionPosterior`
- additional observation noise unsupported

という contract です。

NGBoost library 一般が複数 distribution を扱えることと、robotorchan adapter が現在公開している
capability は区別します。

## 19. NGBoost uncertainty

現在の adapter は

```text
Y | X=x
 ~ Normal(
     mu(x),
     sigma(x)^2
   )
```

という conditional predictive distribution を返します。

この variance は GP の epistemic posterior variance と同じものではありません。

また単一 predictive distribution から epistemic / aleatoric uncertainty を一意に分解できないため、
BALD のような decomposition を必要とする acquisition へ predictive variance だけを根拠に
適用してはいけません。

## 20. Active Learning

non-GP surrogate でも predictive variance、interval width、sample disagreement を利用する
regression Active Learning は可能です。

ただし uncertainty score の statistical meaning は model family ごとに異なります。

empirical tree spread や NGBoost variance を単に BALD と呼ばないことが重要です。

## 21. GP family との境界

| Family | Predictive representation | Fitting | Input gradient |
| --- | --- | --- | --- |
| Exact GP | Gaussian posterior | MLL | Yes |
| Fully Bayesian GP | integrated GP posterior | MCMC | Yes |
| DeepGP | stochastic variational posterior | variational | Yes |
| Tree ensemble | empirical members | estimator fit | No |
| Bootstrap boosting | empirical complete models | repeated fit | No |
| NGBoost | conditional distribution | distributional boosting | No |

Infinite-width BNN GP や Spectral Mixture GP は特殊な kernel を使っても GP family です。

Random Forest、Extra Trees、boosting、NGBoost は non-GP surrogate です。

high-dimensional / expressive という用途ラベルではなく predictive model と inference semantics で
分類します。

## 22. 実装との対応

現在の理論章で保持すべき実装契約は次です。

- non-GP model は BoTorch interface を持つが GP ではない
- empirical ensemble は `EnsemblePosterior` を使う
- Random Forest / Extra Trees は現在 single-output
- `cat_dims` validation は native categorical tree semantics を意味しない
- boosting stages 自体を posterior members にしない
- Gradient / HistGradient Boosting は complete bootstrap models を members にする
- bootstrap boosting は multi-output を扱える
- shared bootstrap index は full Bayesian cross-output covariance を意味しない
- empirical MC acquisition の explicit sampler は `IndexSampler`
- analytic Gaussian acquisition を empirical ensemble に適用しない
- TreeEnsembleSearchStrategy は gradient-free search を行う
- integer / categorical candidate sampling は search strategy が扱う
- capability registry で acquisition requirements を確認する
- 現NGBoost adapterは single-output Gaussian predictive law
- NGBoost variance を GP posterior variance と同一視しない
- BALD compatibility を predictive variance だけから推論しない

## 参考文献

1. Breiman, L. (2001).
   Random forests. *Machine Learning*, 45, 5-32.
2. Geurts, P., Ernst, D., and Wehenkel, L. (2006).
   Extremely randomized trees. *Machine Learning*, 63, 3-42.
3. Friedman, J. H. (2001).
   Greedy function approximation: A gradient boosting machine.
   *The Annals of Statistics*, 29(5), 1189-1232.
4. Duan, T., Avati, A., Ding, D. Y., Basu, S., Ng, A. Y., and Schuler, A. (2020).
   NGBoost: Natural gradient boosting for probabilistic prediction.
   *International Conference on Machine Learning*.
5. Wilson, J., Hutter, F., and Deisenroth, M. (2018).
   Maximizing acquisition functions for Bayesian optimization.
   *Advances in Neural Information Processing Systems*.

## 関連章

- [Acquisition Functions](04_acquisition_function.md)
- [Advanced High-dimensional Models](21_advanced_high_dimensional_models.md)
- [Expressive GP](22_expressive_gp.md)
- [High-dimensional Search](20_high_dimensional_search.md)
