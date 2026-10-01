# Non-GP Surrogates for Bayesian Optimization

## 直感

Bayesian optimization に必要なのは「Gaussian Processそのもの」ではありません。

candidate `X` に対して、

- predictive uncertainty を表現できる
- posterior-like samples を生成できる
- acquisition function が必要とする統計量を提供できる

surrogate であれば、non-GP model も BO に接続できます。

ただし

```text
sampleable predictive distribution
!= Gaussian Process posterior
```

です。

robotorchan はこの違いを model / posterior / acquisition / candidate optimization の各層で
明示します。

## 1. BoTorch Model interface と statistical model

BoTorch の `Model` interface を実装することと、model が Bayesian GP であることは別です。

robotorchan の non-GP surrogate は

```text
raw train_X, train_Y
 -> external estimator
 -> posterior(X)
 -> BoTorch Posterior
 -> acquisition
```

という adapter structure を使います。

これにより acquisition 側は共通 interface を利用できます。

一方、posterior object の statistical meaning は surrogate family ごとに異なります。

## 2. Empirical EnsemblePosterior

ensemble member `s` の prediction を

```text
f_s(X)
```

とすると、robotorchan は

```text
{f_1(X), ..., f_S(X)}
```

を BoTorch `EnsemblePosterior` として表します。

内部 values shape は

```text
... x S x q x m
```

です。

- `S`: ensemble members
- `q`: candidate batch
- `m`: outputs

この representation から IndexSampler により empirical predictive samples を選択できます。

ensemble mean / variance は便利な要約ですが、それだけで Bayesian posterior mean / variance と
解釈してはいけません。

## 3. Random Forest

`RandomForestSurrogate` は fitted forest の各 tree prediction を ensemble member として使います。

```text
tree_1(X)
tree_2(X)
...
tree_S(X)
 -> EnsemblePosterior
```

tree 間の差は bootstrap sampling や feature selection などから生じます。

これは epistemic uncertainty の heuristic として利用できますが、

- observation noise model
- calibrated credible interval
- Bayesian parameter posterior

を自動的に意味しません。

現在の `RandomForestSurrogate` は **single-output only** です。

## 4. Extra Trees

`ExtraTreesSurrogate` も各 tree prediction を empirical ensemble member とします。

Random Forest より split randomization が強く、ensemble diversity の生成機構が異なります。

しかし tree disagreement の統計的解釈に関する注意は同じです。

現在の `ExtraTreesSurrogate` も **single-output only** です。

## 5. Categorical inputs in tree surrogates

Random Forest / Extra Trees は `cat_dims` を受け取れます。

ただし categorical value は現在、finite な integer-valued numeric labels として validation されます。

これは

```text
category code
 -> sklearn numeric input
```

という contract です。

GP の native categorical kernel のように「category label間の数値距離を無視する専用 covariance」を
構築するものではありません。

したがって categorical handling の statistical semantics は Mixed GP と同一ではありません。

## 6. なぜ boosting stage を posterior sample にしないか

gradient boosting predictor は概念的に

```text
F_M(x)
  = F_0(x)
    + sum_j eta h_j(x)
```

です。

stage `h_j` は独立な model draws ではなく、前段の residual を逐次補正するために学習されます。

そのため

```text
{h_1(X), ..., h_M(X)}
```

の spread を posterior uncertainty とみなすのは statistical meaning が不適切です。

## 7. Bootstrap boosting posterior

robotorchan の boosting surrogate は dataset bootstrap を使います。

```text
D_1 -> fit complete model F_1
D_2 -> fit complete model F_2
...
D_S -> fit complete model F_S
```

そして

```text
{F_1(X), ..., F_S(X)}
```

を empirical predictive samples とします。

この方式なら ensemble axis は「complete fitted models の resampling variation」という意味を
持ちます。

## 8. GradientBoostingSurrogate

`GradientBoostingSurrogate` は `BootstrapEnsembleSurrogate` を継承します。

主な contract は、

- `n_members >= 2`
- memberごとに独立bootstrap resample
- memberごとにcomplete GradientBoostingRegressorをfit
- posteriorはEnsemblePosterior

です。

single-outputだけでなく multi-output `train_Y` も受け取れます。

multi-output時は member内部で `MultiOutputRegressor` を利用します。

## 9. HistGradientBoostingSurrogate

`HistGradientBoostingSurrogate` も同じ bootstrap infrastructure を使います。

違いは各 member の base estimator が `HistGradientBoostingRegressor` である点です。

posterior semantics は

```text
bootstrap complete-model ensemble
```

であり、boosting stage ensemble ではありません。

こちらも multi-output を扱えます。

## 10. Multi-output bootstrap semantics

`m > 1` の場合、同じ bootstrap memberでは全outputsに同じ resampled row indicesを使います。

```text
bootstrap member s
 -> same sampled observations
 -> output 1 regressor
 -> output 2 regressor
 -> ...
```

これにより ensemble member alignment は保持されます。

しかし output-specific regressors は独立にfitされるため、これは一般のmulti-output GPのように
明示的 cross-output covariance kernel を学習する model ではありません。

shared bootstrap variation により sample-level dependence が現れる可能性と、
explicit probabilistic output covariance model は区別します。

## 11. observation_noise

tree / bootstrap ensemble posterior は observation noise model を追加する interface を
提供していません。

そのため

```text
posterior(X, observation_noise=True)
```

のような要求は現在サポートしません。

ensemble spread と observation noise variance を同一視しないための明示的な制約です。

## 12. Distributional surrogate: NGBoost

`NGBoostSurrogate` は empirical tree ensemble とは異なります。

NGBoost は conditional probability distribution の parameters を boosting により予測します。

現在の robotorchan adapter は Gaussian predictive law を使います。

```text
Y | X=x
 ~ Normal(
     mu(x),
     sigma(x)^2
   )
```

`posterior(X)` は mean / variance / samples を持つ
`GaussianDistributionPosterior` を返します。

## 13. NGBoost posterior は GP posterior ではない

各 candidate point の predictive marginal が Gaussian でも、

```text
Gaussian marginal
!= joint Gaussian process posterior
```

です。

現在の `GaussianDistributionPosterior` は

```text
Y_i
 = mu_i
   + sigma_i epsilon_i
```

として independent standard-normal base samples を使います。

つまり候補点 `X_1, ..., X_q` 間の full predictive covariance matrix を持つ
`MultivariateNormal` posterior ではありません。

この違いは joint candidate behavior を必要とする acquisition を解釈するときに重要です。

## 14. NGBoost uncertainty の意味

NGBoost が返す predictive variance は configured conditional predictive distribution の
uncertainty です。

これを自動的に

```text
epistemic variance
+
aleatoric variance
```

へ分解できるわけではありません。

したがって BALD のように epistemic / conditional uncertainty の分離を必要とする acquisition を、
Gaussian predictive variance があるという理由だけで適用してはいけません。

## 15. NGBoost の現在の制約

現在の `NGBoostSurrogate` は、

- Gaussian `Normal` distribution
- single-output
- explicit `fit()`
- CPU / numpy-backed external estimator
- `observation_noise != False` 非対応

です。

distribution family や multi-output structure を将来拡張する場合は posterior semantics も
同時に見直す必要があります。

## 16. Acquisition compatibility

non-GP surrogate が sampleable でも、すべての acquisition と互換ではありません。

robotorchan の `validate_non_gp_acquisition` は non-GP empirical ensemble に対して
BoTorch `MCAcquisitionFunction` を要求します。

さらに ensemble posterior の sampler は `IndexSampler` を要求します。

これは discrete empirical posterior を Gaussian base-sample sampler で扱わないためです。

## 17. Analytic Gaussian acquisition

analytic EI などは Gaussian posterior の analytic moments / distribution assumptions を利用します。

tree ensemble posterior は discrete empirical distribution なので、その前提を満たしません。

NGBoostも marginal Gaussian predictionを返しますが、GPのような joint Gaussian covarianceを
持たないため、「Gaussianという名前だけ」で joint-Gaussian acquisition compatibility を
推論してはいけません。

robotorchan の capability / compatibility layer で model requirement を確認します。

## 18. MC acquisition

sampleable posterior に対しては MC acquisition が主要な接続点です。

例えば empirical ensembleでは、

```text
posterior sample
 -> objective
 -> improvement / utility
 -> Monte Carlo average
```

という評価ができます。

qEI / qEHVI などの具体的な適用可否は、

- posterior sampling semantics
- multi-output support
- ensemble support
- acquisition側のposterior requirement

を合わせて判断します。

「MC acquisitionだから常に互換」とは限りません。

## 19. Candidate optimization

sklearn-backed tree / boosting model の prediction は、

- piecewise / non-differentiable
- CPU / numpy boundary
- candidate input gradient を保持しない

という特徴があります。

したがって gradient-based `optimize_acqf` path を前提にしません。

robotorchan は `TreeEnsembleSearchStrategy` を提供し、random-search based candidate
optimization を行います。

## 20. TreeEnsembleSearchStrategy

`TreeEnsembleSearchStrategy` は generic random search を再利用しつつ、non-GP tree model向けの
compatibility validation を追加します。

現在、

- continuous dimensions
- integer dimensions
- categorical dimensions

を candidate sampling に反映できます。

integer dimensions は feasible integer range からsampleし、categorical dimensions は指定された
allowed values からsampleします。

同じ dimension を integer と categorical の両方には指定できません。

## 21. Search strategy と model categorical semantics

model側の categorical treatment と search側の categorical feasibility は別です。

```text
model:
  category labelsをどう予測に使うか

search:
  candidateがallowed category valuesを守るか
```

`TreeEnsembleSearchStrategy` の categorical sampling は candidate feasibility を保証しますが、
surrogate内部に categorical kernel を導入するものではありません。

## 22. Constraints

output constraint

```text
g(x) <= 0
```

のような feasibility rule は acquisition / objective layer に置けます。

surrogate は outputs の predictive representation を提供し、decision policy は acquisition 側で
定義します。

一方 candidate-space constraint は search strategy が candidate generation 時に満たす必要が
あります。

したがって

```text
outcome constraint
!= candidate/input-space constraint
```

です。

## 23. Fantasization / asynchronous BO

GP model では `fantasize()` によって pending observations を条件付けした fantasy model を作る
経路があります。

external sklearn / NGBoost adapter はそのような GP conditioning algebra を自動的には持ちません。

non-GP modelを asynchronous acquisition に使う場合は、

- model capability metadata
- acquisitionの `requires_fantasize`
- pending-point handling

を明示的に確認する必要があります。

sampleable posteriorだけで fantasization support があるとは限りません。

## 24. Active Learning

non-GP posteriorはActive Learningにも利用できますが、criterionの前提が重要です。

moment-based regression AL なら predictive variance を利用できる場合があります。

一方、

- BALD
- mutual information
- latent-function uncertainty decomposition

のようなcriterionは、posterior varianceの存在だけでは成立しません。

特にNGBoostのtotal conditional predictive uncertaintyを epistemic uncertainty と読み替えないことが
重要です。

## 25. Calibration

empirical ensemble standard deviation や NGBoost scale を uncertainty として使うなら、
target problem 上で calibration を検証します。

有用な diagnostics には、

- RMSE / MAE
- negative log predictive density
- interval coverage
- interval width
- calibration curve
- BO simple regret
- acquisition optimization cost

があります。

raw ensemble spread が calibrated uncertainty であると、benchmarkなしに主張してはいけません。

## 26. GP surrogate との比較

| 項目 | Exact GP | Tree ensemble | Bootstrap boosting | NGBoost |
| --- | --- | --- | --- | --- |
| Predictive representation | joint Gaussian | empirical trees | empirical models | Gaussian marginals |
| Candidate covariance | explicit | sample-induced | sample-induced | independent |
| MLL fitting | Yes | No | No | No |
| Input gradients | Yes | No | No | No |
| Observation-noise query | model dependent | No | No | No |
| Multi-output | model dependent | RF/ET: No | Yes | No |
| Typical acquisition | analytic / MC | MC | MC | MC / moments |
| Typical search | gradient-based | gradient-free | gradient-free | gradient-free |

この表は statistical contract の比較であり、性能順位ではありません。

## 27. 実装との対応

現在の理論章で保持すべき実装契約は次です。

- non-GP surrogate も BoTorch `Model.posterior()` interface を提供する
- RandomForest / ExtraTrees は各treeを empirical posterior member とする
- RandomForest / ExtraTrees は現在 single-output only
- tree `cat_dims` は integer-valued numeric labels を validation する
- GradientBoosting / HistGradientBoosting は complete-model bootstrap ensemble
- boosting stages 自体を posterior samples とみなさない
- bootstrap boosting は multi-output を扱える
- multi-output bootstrap は member index をoutputs間で共有する
- empirical ensemble posterior は `EnsemblePosterior`
- NGBoost は Gaussian conditional predictive distribution を返す
- NGBoost は現在 single-output
- NGBoost posterior は full joint Gaussian covariance を持たない
- empirical ensemble と NGBoost は observation-noise query を追加しない
- non-GP acquisition compatibility は posterior requirement で判定する
- empirical ensemble MC sampling は `IndexSampler` を使う
- tree/boosting candidate search は gradient-free path を使う
- TreeEnsembleSearchStrategy は integer / categorical candidate sampling を扱う
- outcome constraint と candidate-space constraint を分離する
- posterior sampling support と fantasization support を区別する
- predictive variance と epistemic / aleatoric decomposition を混同しない

## 参考文献

1. Breiman, L. (2001).
   Random forests.
   *Machine Learning*, 45, 5-32.
2. Geurts, P., Ernst, D., and Wehenkel, L. (2006).
   Extremely randomized trees.
   *Machine Learning*, 63, 3-42.
3. Friedman, J. H. (2001).
   Greedy function approximation: A gradient boosting machine.
   *Annals of Statistics*, 29(5), 1189-1232.
4. Duan, T., Avati, A., Ding, D. Y., Basu, S., Ng, A. Y., and Schuler, A. (2020).
   NGBoost: Natural gradient boosting for probabilistic prediction.
   *International Conference on Machine Learning*.
5. Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2017).
   Simple and scalable predictive uncertainty estimation using deep ensembles.
   *Advances in Neural Information Processing Systems*.
6. Hutter, F., Hoos, H. H., and Leyton-Brown, K. (2011).
   Sequential model-based optimization for general algorithm configuration.
   *Learning and Intelligent Optimization*.

## 関連章

- [Bayesian Optimization](01_bayesian_optimization.md)
- [Acquisition Functions](04_acquisition_function.md)
- [Mixed Variables](05_mixed_variables.md)
- [High-dimensional Search](20_high_dimensional_search.md)
- [Expressive GP](22_expressive_gp.md)
