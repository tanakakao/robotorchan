# Model training API note

robotorchanではExact GP wrapperの `make_mll()` と、Joint neural modelの `training_loss()` を役割で分けます。

- 通常のExact GP / frozen reducer model: `fit_gpytorch_mll(model.make_mll())`
- Joint neural model: optimizerで `model.training_loss()` を最小化

この区別により `make_mll()` の共通Exact GP契約を保ったまま、GP MLL以外のreconstruction / KL regularizationを欠落させず学習できます。

詳細は [Joint neural training](../models/joint_neural_training.md) を参照してください。

## `fit()` と学習方式の関係

`fit()` は、学習方式が定義済みのモデルで、コンストラクタに渡した学習データから学習するための便利な入口です。
例えば `RandomForestSurrogate.fit()` は内部のscikit-learn推定器を学習します。
`SingleTaskGP` と `MixedSingleTaskGP` では `fit()` も利用できます。全モデル共通の
`fit()` は保証していません。

将来ほかのモデルに `fit()` を追加する場合も、以下の区別を維持してください。

- Exact GP: `make_mll()` とBoTorchの `fit_gpytorch_mll` を利用できる状態を保つ
- Variational GP: 適切なELBOとデータ数・最適化設定を扱い、Exact GP用MLLに置換しない
- Joint neural / robust GP: モデル固有の `training_loss()` や学習手順を維持する
- Fully Bayesian GP: MCMC等の固有手順を考慮し、一般の勾配法と同一視しない
- Non-GP: 既存の `fit()` と学習済み状態のチェックを維持する
- Heterogeneous model: 構成モデルごとの学習方式を尊重し、共通の事後分布を仮定しない

`torch.nn.Module.train()` / `eval()` は学習・評価モードの切り替えであり、
推定器をフィットする `fit()` とは異なります。
共通化ではBoTorch標準の `posterior()`、MLL、外部Optimizer、Samplerを
引き続き直接利用できることが前提です。
対応しない学習方式は無理に汎用化せず、明確なCapabilityとエラーで示します。

## 標準Exact GPの便利な学習API

標準 `SingleTaskGP` と `MixedSingleTaskGP`は以下のように学習できます。

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
この便利なAPIは、現時点で標準の2モデルだけが対応します。
特殊損失、Variational GP、MCMC、Joint Neuralの学習手順は置き換えません。
