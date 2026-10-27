# Level-set / Straddle：理論から実装へ

## 1. この章で解決する問い

目的は最適値でなくf(x)=tの境界・閾値上下の分類。posterior平均が境界近くても不確実性が小さければ追加測定価値は限られる。

数学的な定義・仮定・導出は[理論の対応章](../../../../theory/acquisition/10_level_set.md)が正本であり、この章は**数式の計算対象 → 実装の所有権 → 公開API → 探索上の制約**をつなぐ日本語実装ガイドである。

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

`Straddle`のスコアは`beta*sigma-abs(mu-t)`、`RandomizedStraddle`は確率的beta、`BoundaryVariance`は別の境界近接heuristic。すべてlocal Implemented。

参照する代表的な公開APIは ``robotorchan.acquisition.Straddle``。ここでは**名前・責務の参照**であり、すべてのモデルやoptimizerとの組合せを実行検証済みとするものではない。

| 層 | 何を担当するか | 検証上の要点 |
| --- | --- | --- |
| surrogate | posterior、必要に応じてclass probability | posterior type、出力軸、学習方式 |
| acquisition | utilityの期待値・近似または探索score | 定義した数式、方向、q、sample形状 |
| semantic bridge | objective/constraintの意味論変換 | objectiveとclassification feasibilityの分離 |
| optimizer | 入力境界、fixed features、candidate constraintsを満たす候補検索 | one-shot/gradient/mixedの対応、最適化失敗の検出 |

## 4. 採用時の確認と注意事項

q=1と出力選択の制限、Straddle betaとUCBのsqrt(beta)慣行の違い、RandomizedStraddleのresample境界を確認。

モデル側のposteriorが正しい形を返すことと、獲得関数の数学的同値性を示すことは別である。またacquisition functionの単体テストと、候補探索が可行な点を返すことは別々に検証する。

## 5. 数学的検証と実装の追跡

[既存Acquisition Mapping](../../acquisition.md)はsource inspectedの実装契約、entry・テスト証拠を保持する。[Verification Mapping](../../verification.md)は独立数値参照や既存のassertionの追跡先とする。両者を消去・置き換えず、原文区間のmigration ledgerをverifiedとする際に照合する。CI成功だけで全数値契約が検証されたことにはならない。

[獲得関数の基礎ガイド](../04_acquisition_function.md)、[獲得関数利用ドキュメント](../../../../optimization/README.md)、[理論Acquisition索引](../../../../theory/acquisition/README.md)にも戻れる。個別の数式、適用例や比較条件は対応する理論章と使用ガイドで読む。

## 6. Straddleで閾値付近の候補を評価する

`Straddle`は予測平均と閾値の距離を、不確実性による探索ボーナスから引く。
`beta`は標準偏差に直接掛かるので、UCBの`sqrt(beta)`と取り違えない。

```python
import torch
from robotorchan.acquisition import Straddle
from robotorchan.models import SingleTaskGP

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(4 * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
candidates = torch.tensor([[[0.15]], [[0.45]], [[0.85]]], dtype=torch.double)
target = 0.35
beta = 1.7
scores = Straddle(model, target=target, beta=beta)(candidates)
assert scores.shape == (3,)
assert torch.isfinite(scores).all()
```

このスコアは境界探索のheuristicであり、閾値の上下を表す分類確率ではない。
また`q=1`の候補別評価であり、複数候補のjoint utilityを求めたものでもない。
