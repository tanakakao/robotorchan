# Multi-Fidelity / Cost-aware：理論から実装へ

## 1. この章で解決する問い

低fidelityの安さだけを最大化せず、target fidelityを改善する情報とcostを同時に評価する。fidelity featureは単なるcategoryではない。

数学的な定義・仮定・導出は[理論の対応章](../../../../theory/acquisition/11_multifidelity_cost_aware.md)が正本であり、この章は**数式の計算対象 → 実装の所有権 → 公開API → 探索上の制約**をつなぐ日本語実装ガイドである。

## 2. posteriorからcandidateへの処理の流れ

```text
train observations → surrogate.fit / posterior(X)
                                     ↓
                          objective / utility
                                     ↓
                  acquisition function α(X)
                                     ↓
                    candidate optimizer
                                     ↓
             selected X → costly evaluation
```

候補点で観測する未知の応答値と、観測前に計算する獲得関数値は異なる。獲得関数が最大化するutilityを、posterior meanの予測結果と読み替えない。analytic計算、MC sampling、fantasization、classification probability等、用いる数値表現は手法とモデルによって異なる。

## 3. 実装の対応と委譲

`qMultiFidelityKnowledgeGradient`とcost-aware utilityは主としてBoTorch-native。robotorchanの`SingleTaskMultiFidelityGP`、mixed/reduced variantsはposterior基盤を提供し、acquisitionとの接続を検査する。

参照する代表的な公開APIは ``botorch.acquisition.knowledge_gradient.qMultiFidelityKnowledgeGradient``。ここでは**名前・責務の参照**であり、すべてのモデルやoptimizerとの組合せを実行検証済みとするものではない。

| 層 | 何を担当するか | 検証上の要点 |
| --- | --- | --- |
| surrogate | posterior、必要に応じてclass probability | posterior type、出力軸、学習方式 |
| acquisition | utilityの期待値・近似または探索score | 定義した数式、方向、q、sample形状 |
| semantic bridge | objective/constraintの意味論変換 | objectiveとclassification feasibilityの分離 |
| optimizer | 入力境界、fixed features、candidate constraintsを満たす候補検索 | one-shot/gradient/mixedの対応、最適化失敗の検出 |

## 4. 採用時の確認と注意事項

target-fidelity projection、cost model、fantasy/one-shot route、fidelity次元固定、費用尺度を揃える。

モデル側のposteriorが正しい形を返すことと、獲得関数の数学的同値性を示すことは別である。またacquisition functionの単体テストと、候補探索が可行な点を返すことは別々に検証する。

## 5. 数学的検証と実装の追跡

[既存Acquisition Mapping](../../acquisition.md)はsource inspectedの実装契約、entry・テスト証拠を保持する。[Verification Mapping](../../verification.md)は独立数値参照や既存のassertionの追跡先とする。両者を消去・置き換えず、原文区間のmigration ledgerをverifiedとする際に照合する。CI成功だけで全数値契約が検証されたことにはならない。

[獲得関数の基礎ガイド](../04_acquisition_function.md)、[獲得関数利用ドキュメント](../../../../optimization/README.md)、[理論Acquisition索引](../../../../theory/acquisition/README.md)にも戻れる。個別の数式、適用例や比較条件は対応する理論章と使用ガイドで読む。

## 6. Fidelity依存コストを評価する最小例

情報量や改善量をコストと比較するときは、fidelity列とコスト関数の関係を明示する。
以下は`qMultiFidelityKnowledgeGradient`そのものを評価する前段階として、
BoTorchの`AffineFidelityCostModel`が候補ごとの正のコストを返すことを確かめる。

```python
import torch
from botorch.models.cost import AffineFidelityCostModel

cost_model = AffineFidelityCostModel(fidelity_weights={1: 1.0}, fixed_cost=0.1)
candidates = torch.tensor([[[0.3, 0.5]], [[0.3, 1.0]]], dtype=torch.double)
costs = cost_model(candidates)
assert torch.isfinite(costs).all()
assert torch.all(costs > 0)
assert costs.reshape(-1)[0] < costs.reshape(-1)[1]
```

この単純な線形コストモデルでは同じ設計点でも高いfidelityの費用が大きい。
これは`qMultiFidelityKnowledgeGradient`が改善情報とコストを正しく統合できる証拠ではない。
その確認には`project`、`num_fantasies`、one-shotの拡張q次元と
`InverseCostWeightedUtility`を含めた獲得関数テストが別途必要となる。
