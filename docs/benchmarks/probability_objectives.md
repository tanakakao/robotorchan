# 確率を目的とする問題

Pass 確率を実行可能性制約ではなく最適化目的として扱います。

- `pass_probability_objective`：Pass 確率を最大化。
- `yield_probability_tradeoff`：決定的 Yield と Pass 確率を同時に最大化。

真値確率は次です。

```text
p(Pass | x) = 0.1 + 0.8 * exp(-8 * ((x0 - 0.7)^2 + (x1 - 0.3)^2))
```


単一目的の最適点は `x=(0.7,0.3)`、確率は `p=0.9` です。多目的の Yield は `1-(x0-0.2)^2-(x1-0.7)^2`、reference point は `(0,0)` です。

`evaluate_truth(X)` は解析的確率を返します。分類訓練の Bernoulli 0/1 label は、明示的な PyTorch RNG を使う `sample_pass_labels(X, generator)` で別に抽出します。

runner は解析的確率を評価し、確率的な label を評価するわけではありません。label を較正済みの確率観測と混同せず、分類器の API が返す予測確率を獲得目的へ接続してください。

`simple_regret_curve`・`hypervolume_curve` は真値目的に適用します。問題定義だけでは分類器の較正や qEHVI 合成を認定しません。

## 学習確率を使う検証

有限プール上の `ProbabilityObjective` 検証は [過去の検証記録](../../.github/project-records/benchmarks/probability-objective-e2e.md)に保存しています。解析的確率問題とは別のデータ・評価契約です。実行方法は `pytest -q tests/benchmarks/test_heterogeneous_probability_objective_e2e.py` です。有限プールでの選択は native qEI・qNEI・qEHVI の検証や真の改善・較正の保証ではありません。
