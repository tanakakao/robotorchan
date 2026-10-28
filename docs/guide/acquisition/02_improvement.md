# Improvement / LogEI / PI：理論から実装へ

## 1. この章で解決する問い

PIは改善する確率、EIは正の改善量の期待値。log-space formは目的自体を変えるのではなく数値的安定性を改善する。

数学的な定義・仮定・導出は[理論の対応章](../../theory/acquisition/02_improvement.md)が正本であり、この章は**数式の計算対象 → 実装の所有権 → 公開API → 探索上の制約**をつなぐ日本語実装ガイドである。

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

標準EI・LogEI・LogPIはBoTorch-native（Delegated）。`make_qei_acquisition`はsemantic objective bridge（Extended）で、独立したEIアルゴリズムではない。

参照する代表的な公開APIは ``botorch.acquisition.analytic.LogExpectedImprovement``。ここでは**名前・責務の参照**であり、すべてのモデルやoptimizerとの組合せを実行検証済みとするものではない。

| 層 | 何を担当するか | 検証上の要点 |
| --- | --- | --- |
| surrogate | posterior、必要に応じてclass probability | posterior type、出力軸、学習方式 |
| acquisition | utilityの期待値・近似または探索score | 定義した数式、方向、q、sample形状 |
| semantic bridge | objective/constraintの意味論変換 | objectiveとclassification feasibilityの分離 |
| optimizer | 入力境界、fixed features、candidate constraintsを満たす候補検索 | one-shot/gradient/mixedの対応、最適化失敗の検出 |

## 4. 採用時の確認と注意事項

`best_f`の向き、標準化済みposteriorとの尺度、zero posterior variance、legacy EIとLogEIの数値安定性を照合する。

モデル側のposteriorが正しい形を返すことと、獲得関数の数学的同値性を示すことは別である。またacquisition functionの単体テストと、候補探索が可行な点を返すことは別々に検証する。

## 5. 数学的検証と実装の追跡

[既存Acquisition Mapping](../../../.github/project-records/theory-guide/mapping/acquisition.md)はsource inspectedの実装契約、entry・テスト証拠を保持する。[Verification Mapping](../../../.github/project-records/theory-guide/mapping/verification.md)は独立数値参照や既存のassertionの追跡先とする。両者を消去・置き換えず、原文区間のmigration ledgerをverifiedとする際に照合する。CI成功だけで全数値契約が検証されたことにはならない。

[獲得関数の基礎ガイド](../04_acquisition_function.md)、[獲得関数利用ドキュメント](../../optimization/README.md)、[理論Acquisition索引](../../theory/acquisition/README.md)にも戻れる。個別の数式、適用例や比較条件は対応する理論章と使用ガイドで読む。

## 6. LogEIを評価する最小例

次の例では、観測済み最大値を`best_f`に設定し、
`LogExpectedImprovement`のスコアを候補点で計算する。
学習と獲得関数の評価は異なる処理なので、ここでは候補最適化を実行しない。

```python
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from robotorchan.models import SingleTaskGP

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(2 * torch.pi * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
candidates = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double)
best_f = train_Y.max()
acqf = LogExpectedImprovement(model=model, best_f=best_f)
scores = acqf(candidates)
assert scores.shape == (2,)
assert torch.isfinite(scores).all()
```

`candidates`は`batch_shape × q × d`で、この例では`q=1`。
値はlog-EIなので、EIに戻すには`scores.exp()`を使う。
観測ノイズと`best_f`の整合、予測分散が小さい場合の安定性、
真の最適化改善幅は別途評価する。

## 理論章から分離した実装上の条件

#### References：実装上の補足

- Ament, S. et al. (2023), *Unexpected Improvements to Expected Improvement for Bayesian
Optimization*. arXiv:2310.20708. BoTorch's log-EI family follows this numerically robust
formulation.
- BoTorch documentation, *Acquisition Functions*, for the current analytic / Monte Carlo
acquisition interfaces and log-space EI variants.
