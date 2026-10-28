# Unified model-to-acquisition workflow

このページは、robotorchan のモデル学習 API と BoTorch 標準の
posterior / sampler / objective / acquisition / optimizer を接続する利用ガイドです。
実行可能な最小例は
[`examples/unified_api_workflow.py`](../../examples/unified_api_workflow.py)
にあります。

## 1. 実行する

リポジトリのルートで以下を実行します。

```bash
pip install -e .
python examples/unified_api_workflow.py
```

外部データやネットワークを使用せず、小さな合成データで
2出力GPの学習、予測、獲得関数評価、`q=2`の候補生成を行います。
結果は候補の乱数生成だけでなく、モデル学習の初期状態にも依存するため、
表示される候補座標そのものの一致は保証しません。

## 2. モデル学習と raw data

通常のExact GPでは `model.make_mll()` と
`botorch.fit.fit_gpytorch_mll(model.make_mll())` を直接利用できます。
明示的に opt-in したモデルでは `model.fit(optimizer_kwargs=...)` も利用できます。

```python
model = SingleTaskGP(train_X, train_Y)
if model.supports_fit:
    model.fit(optimizer_kwargs={"options": {"maxiter": 100}})
```

`raw_train_X`、`raw_train_Y`、`raw_train_Yvar` は
初期の入力データに関する契約です。
`supports_fit` は実装済みモデルだけが `True` を宣言します。
特殊なVariational / Fully Bayesian / Deepモデルに `fit()` が
存在することを一律に仮定しないでください。
反復測定ノイズの例は
[Replicate-noise GP](../models/replicate_noise_gp.md) を参照してください。

## 3. Posterior → Sampling → Objective → Acquisition

```python
sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=110)
objective = LinearMCObjective(weights=torch.tensor([0.7, 0.3], dtype=torch.double))
acquisition = qSimpleRegret(model=model, sampler=sampler, objective=objective)
```

上記の `qSimpleRegret` はBoTorchのクラスです。
robotorchanは標準の獲得関数を独自クラスとして再公開しません。
`qSimpleRegret` のサンプル上の目的値は `sample × batch × q`、
獲得関数の値は `batch` です。
複数の出力を1目的へ変換するときは `LinearMCObjective` などの
適切なBoTorch Objectiveを明示的に渡します。

改善量を使う通常のBOには `qLogExpectedImprovement`、
観測ノイズを考慮する場合は `qLogNoisyExpectedImprovement` を検討します。
`qProbabilityOfImprovement` は確率改善型で、
BoTorchのOutcome Constraintを指定できます。
`qSimpleRegret` は制約付き獲得関数としては登録していません。

## 4. Optimizerと制約の責務

```python
candidates, score = optimize_acqf(
    acquisition,
    bounds,
    q=2,
    optimizer="sobol",
    raw_samples=32,
    seed=110,
)
```

この `optimize_acqf` は `robotorchan.optim` の任意のdispatch APIです。
標準の `botorch.optim.optimize_acqf` を直接利用する方法も維持します。
`optimizer="botorch"` がデフォルトで、`"sobol"` は
軽量な候補生成・確認用の選択肢です。

実験結果で定義される **Outcome Constraint** は
獲得関数・目的関数側で扱います。一方、
設計変数Xの既知制約はOptimizerの `CandidateConstraints` を使います。
カテゴリ変数を含むBoTorch mixed最適化は、
カテゴリ割当てを明示する `optimize_acqf_mixed` の利用も可能です。

## 5. 回帰と分類が混在する問題

`HeterogeneousModel` は回帰と分類を同一のoutput index / nameで管理しますが、
**共通の `posterior()` は作成しません**。
回帰のlatent posteriorと分類ラベル確率を混同しないためです。
`ProblemSemantics` によって回帰目的と分類の実行可能確率を指定し、
対応するAcquisition Compositionを明示的に利用します。

具体的な `MixedSingleTaskGP + MixedBinarySingleTaskGPClassifier`
接続は
[`test_phase9_mixed_heterogeneous.py`](../../tests/integration/test_phase9_mixed_heterogeneous.py)
で検証しています。q-batchの分類実行可能確率の積は
独立性近似であり、分類モデルの共同実行可能確率そのものではありません。

## 6. 対応範囲と検証

- 公開API: [Acquisition architecture](../../docs-internal/architecture/acquisition-architecture.md)
- モデル学習: [Training API](../development/training_api.md)
- 獲得関数の対応範囲: [Acquisition integration](../../docs-internal/contracts/optimization/acquisition-integration.md)
- Optimizer: [Public optimizer API](public-optimizer-api.md)

モデル×獲得関数×Optimizerの全組み合わせを動作保証するものではありません。
利用する組み合わせについてはCapabilityと実行時テストの両方を確認してください。
