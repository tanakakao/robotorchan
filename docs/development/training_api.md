# Model training API note

robotorchanではExact GP wrapperの `make_mll()` と、Joint neural modelの `training_loss()` を役割で分けます。

- `supports_fit=True` のExact GP: `model.fit()`（内部ではBoTorchのMLL最適化）
- その他のExact GP / frozen reducer model: `fit_gpytorch_mll(model.make_mll())`
- Joint neural model: optimizerで `model.training_loss()` を最小化

この区別により `make_mll()` の共通Exact GP契約を保ったまま、GP MLL以外のreconstruction / KL regularizationを欠落させず学習できます。

詳細は [Joint neural training](../models/joint_neural_training.md) を参照してください。

## `fit()` と学習方式の関係

`fit()` は、学習方式が定義済みのモデルで、コンストラクタに渡した学習データから学習するための便利な入口です。
例えば `RandomForestSurrogate.fit()` は内部のscikit-learn推定器を学習します。
`SingleTaskGP`、`MixedSingleTaskGP`、標準のMultiTask / Kronecker / MultiFidelity GPと
そのMixed派生モデル、`ModelListGP` では `fit()` も利用できます。全モデル共通の
`fit()` は保証していません。

将来ほかのモデルに `fit()` を追加する場合も、以下の区別を維持してください。

- Exact GP: `make_mll()` とBoTorchの `fit_gpytorch_mll` を利用できる状態を保つ
- Pairwise GP: Laplace近似MLLを維持し、Gaussian Exact GPのMLLに置き換えない
- Variational GP: `fit(epochs=..., lr=..., batch_size=...)` でELBOをAdam最適化する。Exact GP用MLLに置換しない
- Joint neural / robust GP: モデル固有の `training_loss()` や学習手順を維持する
- Fully Bayesian GP: MCMC等の固有手順を考慮し、一般の勾配法と同一視しない
- Non-GP: 既存の `fit()` と学習済み状態のチェックを維持する
- Heterogeneous model: 構成モデルごとの学習方式を尊重し、共通の事後分布を仮定しない

`torch.nn.Module.train()` / `eval()` は学習・評価モードの切り替えであり、
推定器をフィットする `fit()` とは異なります。
共通化ではBoTorch標準の `posterior()`、MLL、外部Optimizer、Samplerを
引き続き直接利用できることが前提です。
対応しない学習方式は無理に汎用化せず、明確なCapabilityとエラーで示します。

## 学習API対応表

| モデル系統 | 学習方法 | 確認対象 |
| --- | --- | --- |
| 標準Exact GP・Mixed・MultiTask・MultiFidelity | `fit(optimizer_kwargs=...)` | Exact GP契約 |
| ModelListGP | `fit(optimizer_kwargs=...)` | SumMarginalLogLikelihood |
| PairwiseGP | `fit()` | Pairwise Laplace MLL |
| Variational GP・Mixed | `fit(epochs=..., lr=..., batch_size=...)` | 変分ELBO |
| Fully Bayesian SAAS | `fit(warmup_steps=..., num_samples=..., ...)` | NUTS |
| Joint neural GP | `fit(epochs=..., lr=...)` | モデル固有のtraining_loss |
| DeepGP・Binary DeepGP | `fit(epochs=..., lr=..., ...)` | 変分ELBO |
| Binary variational GP・派生分類GP | `fit(epochs=..., lr=..., batch_size=...)` | Bernoulli ELBO |
| Scikit-learn・Bootstrap分類 | `fit()` | 非GP分類器の学習 |
| GP・Heterogeneous分類アンサンブル | メンバーを個別に学習 | 集約層にfitなし |

この表は統一シグネチャを意味しません。モデルの学習目的・バックエンドに応じて
引数と最適化手法は異なります。個々のモデルで `supports_fit` の有無や
`make_mll()` の対応を確認し、非対応の学習方式を推測して呼び出さないでください。
分類アンサンブルは学習済みメンバーの確率予測を集約します。

## 標準Exact GPの便利な学習API

対応する標準Exact GPモデルは以下のように学習できます。

```python
model = SingleTaskGP(train_X, train_Y)
model.fit()
posterior = model.posterior(test_X)
```

学習の設定が必要な場合は、BoTorchのOptimizerに渡す引数を指定します。

```python
model.fit(optimizer_kwargs={"options": {"maxiter": 100}})
```

`fit()` は `None` を返し、BoTorchの `fit_gpytorch_mll(model.make_mll())`
を内部で使用します。従来の直接呼び出しも利用できます。
この便利なAPIは標準のSingleTask、MultiTask、Kronecker MultiTask、MultiFidelity GPと
対応するMixed派生モデルで利用できます。`ModelListGP` でも `fit()` を利用でき、
`SumMarginalLogLikelihood` をBoTorchの `fit_gpytorch_mll` に渡します。
従来の `fit_gpytorch_mll(model.make_mll())` も引き続き使用できます。
モデル固有の特殊損失、Variational GP、MCMC、Joint Neuralの学習方式は、
Exact GP向けのMLL最適化には置き換えません。各系統のfit契約を使用します。

## Pairwise preference GP

`PairwiseGP` は `model.fit()` に対応しています。内部では
`PairwiseLaplaceMarginalLogLikelihood` を `fit_gpytorch_mll` で最適化します。
比較データの勝敗関係を学習するモデルであり、通常の回帰用Exact GPとは
異なるMLLを維持します。従来の `fit_gpytorch_mll(model.make_mll())` も使用できます。

## Variational GP

`SingleTaskVariationalGP` と `MixedSingleTaskVariationalGP` は
`model.fit(epochs=60, lr=0.03, batch_size=64)` を利用できます。
コンストラクタで `train_Y` を指定する必要があります。
`batch_size` を省略すると全件学習になります。ELBOの `num_data` は
minibatchのサイズではなく、保持された学習データの全件数です。

## Fully Bayesian SAAS GP

`SaasFullyBayesianSingleTaskGP` と `SaasFullyBayesianMultiTaskGP` は
`fit(warmup_steps=..., num_samples=..., thinning=..., disable_progbar=...)`
でBoTorchのNUTS学習を実行します。Mixed派生モデルも同じ契約です。
`supports_fit=True` ですが `supports_mll=False` のままであり、
Exact GPのMLL最適化とは異なります。

## Joint neural GP

`JointEncoderGP` と `HybridAutoEncoderGP` は `fit(epochs=..., lr=...)`
でモデル固有の `training_loss()` をAdam最適化します。Joint VAEとMixed派生モデルも
継承した `fit()` を使用できます。再構成損失やKL項を含む場合も、
`training_loss()` の定義を変更せずに学習します。

## Deep Gaussian Process

`SingleTaskDeepGP` とそのMultiTask・Mixed派生モデルは
`fit(epochs=..., lr=..., num_likelihood_samples=...)` に対応します。
`training_loss()` の変分ELBOをAdamで最適化し、Exact GPのMLL最適化には
置き換えません。学習後は評価モードに切り替わります。

## Binary DeepGP classification

`BinarySingleTaskDeepGPClassifier` はDeepGPの `fit(epochs=..., lr=...,
num_likelihood_samples=...)` を継承します。学習損失は分類器が定義する
Bernoulli変分ELBOであり、回帰用Gaussian likelihoodには変更しません。
学習後は `predict_proba()` でクラス確率を取得できます。

## Binary variational GP classification

`BinarySingleTaskGPClassifier` とMixed・MultiTask派生モデルは、
`SingleTaskVariationalGP.fit(epochs=..., lr=..., batch_size=...)` を継承します。
`make_mll()` がBernoulli ELBOを返すため、分類用likelihoodのまま学習します。
Notebook 39では `fit()` から候補提案までの流れを確認できます。

## Extended binary classification models

高次元MAP-SAAS、ラベルノイズ、および候補入力不確実性を扱うBinary GPは、
標準Binary Variational GPの `fit()` を継承します。代表的な派生モデルについて、
ミニバッチ学習と学習後の確率予測を契約テストで確認します。

GP分類アンサンブルは学習済みメンバーを集約するモデルであり、
アンサンブル自体に `fit()` があるとはみなしません。各メンバーの学習と
アンサンブルの構築は区別してください。非GP分類器の学習APIも別契約です。

## Non-GP binary classification

Scikit-learn系Binary分類器とBootstrap分類アンサンブルは、
`fit()`（引数なし）で保存済みの訓練データを学習します。
GPの `fit(epochs=..., lr=..., batch_size=...)` とは異なる契約です。
`is_fitted` で学習済み状態を確認し、学習後に `predict_proba()` を使用します。
Scikit-learnはoptional dependencyのままとし、未導入環境では該当テストをskipします。

## Classification ensembles

`GPBinaryClassificationEnsemble` と
`HeterogeneousBinaryClassificationEnsemble` は、学習済みのメンバーモデルを
受け取り、確率予測を集約するコンポジション層です。アンサンブル本体には
`fit()` を追加しません。各メンバーをそれぞれの学習APIで学習してから
アンサンブルを構築してください。重み付き異種アンサンブルでは、
`predict_proba()` が各メンバーの確率の重み付き平均を返します。
