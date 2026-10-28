# Level-set / Straddle：理論から実装へ

## 1. この章で解決する問い

目的は最適値でなくf(x)=tの境界・閾値上下の分類。posterior平均が境界近くても不確実性が小さければ追加測定価値は限られる。

数学的な定義・仮定・導出は[理論の対応章](../../theory/acquisition/10_level_set.md)が正本であり、この章は**数式の計算対象 → 実装の所有権 → 公開API → 探索上の制約**をつなぐ日本語実装ガイドである。

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

[既存Acquisition Mapping](../../../.github/project-records/theory-guide/mapping/acquisition.md)はsource inspectedの実装契約、entry・テスト証拠を保持する。[Verification Mapping](../../../.github/project-records/theory-guide/mapping/verification.md)は独立数値参照や既存のassertionの追跡先とする。両者を消去・置き換えず、原文区間のmigration ledgerをverifiedとする際に照合する。CI成功だけで全数値契約が検証されたことにはならない。

[獲得関数の基礎ガイド](../04_acquisition_function.md)、[獲得関数利用ドキュメント](../../optimization/README.md)、[理論Acquisition索引](../../theory/acquisition/README.md)にも戻れる。個別の数式、適用例や比較条件は対応する理論章と使用ガイドで読む。

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

## 実装補足：理論本文から分離した対応情報

### 10.24 BoTorch / robotorchanとの対応

robotorchan は level-set / boundary learning 用に Straddle、RandomizedStraddle、BoundaryVariance を提供します。

これらは現在、scalarized q=1 regression acquisition として明示的な適用範囲を持ち、ensemble posterior など未定義の reduction
semantics は拒否します。

Theory は将来の拡張可能性を説明できますが、現在の利用可能範囲については次を参照してください。

- [Level-set learning guide](../../optimization/level_set_learning.md)
- [BoundaryVariance guide](../../optimization/boundary-variance.md)
- [Acquisition integration status](../../../docs-internal/contracts/optimization/acquisition-integration.md)

## 理論章から分離した実装上の条件

#### 10.6 Straddle：実装上の補足

robotorchan で扱う形は

#### 10.11 Randomized Straddle：実装上の補足

robotorchan の実装が参照する randomized straddle 系の考え方では、正の random variable から探索係数を生成し、

#### 10.13 Randomized Straddleのprovenance：実装上の補足

Randomized Straddle は robotorchan 固有の名称を付けた heuristic ではなく、randomized straddle に関する既存研究に基づく
acquisition family として扱います。

#### 10.13 Randomized Straddleのprovenance：実装上の補足

Theory では手法の原理を説明し、現在の具体的な実装挙動は [Level-set guide](../../optimization/level_set_learning.md) と
[Acquisition integration status](../../../docs-internal/contracts/optimization/acquisition-integration.md) を正とします。

#### 10.14 BoundaryVariance：実装上の補足

BoundaryVariance は robotorchan-specific heuristic です。

#### 10.17 Literature methodとlibrary-specific heuristicを分ける：実装上の補足

robotorchan の theory docs では provenance を明確にします。

#### 10.17 Literature methodとlibrary-specific heuristicを分ける：実装上の表現

```text
Straddle
    established level-set / boundary-learning criterion

Randomized Straddle
    literature-based randomized acquisition family

BoundaryVariance
    robotorchan-specific heuristic
```

#### 10.21 Batch Level-set Learning：実装上の補足

robotorchan-specific q=1 acquisitions が存在することは、任意 q の joint semantics が自動的に定義されることを意味しません。

#### 10.25 この章で覚えておくこと：実装上の補足

- function learningとlevel-set learningは同じuncertaintyを使えても目的が違う
- level setとexcursion setを区別する
- Straddleは `beta * std - abs(mean - target)` のq=1 criterion
- fixed betaは実装上のexploration parameterで、理論confidence scheduleと自動的に同一ではない
- RandomizedStraddleはselection round内で1つのrandom coefficientを固定する
- BoundaryVarianceはrobotorchan-specific heuristicで、beta parameterを持たない
- multi-outputやstructured outputでは、境界を定義するscalar quantityが必要
- pointwise score上位q点はjoint batch level-set acquisitionではない
- level-set estimation、constrained BO、classification ALはboundaryを共有してもdecision purposeが違う

#### 10.26 まとめ：実装上の表現

```text
Straddle
    uncertainty - target distance

Randomized Straddle
    randomized exploration coefficient を持つ literature-based family

BoundaryVariance
    variance × standardized target proximity
    robotorchan-specific heuristic
```

#### 10.26 まとめ：実装上の補足

`BoundaryVariance` は robotorchan 固有であり、文献上の名前付き獲得手法としての引用先はありません。

## 数値 API と利用上の補足

したがって API 上の対称性だけを理由に $`\beta`$ を追加すると、定義されていない parameter を持つ別の heuristic に変わってしまいます。

Theory と public API の双方で、この違いを維持することが重要です。

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

したがって BoundaryVariance は、

BoundaryVariance の定義には Straddle のような $`\beta`$ exploration parameter はありません。

PosteriorVariance は

```text
PosteriorVariance
    function learning

Straddle / BoundaryVariance
    boundary learning
```
