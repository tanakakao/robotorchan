# Information-theoretic：理論から実装へ

## 1. この章で解決する問い

観測の価値は最適点、最適値、Pareto解など指定した未知量についてのmutual information。対象を明記しない情報利得は不十分。

数学的な定義・仮定・導出は[理論の対応章](../../../../theory/acquisition/05_information_theoretic.md)が正本であり、この章は**数式の計算対象 → 実装の所有権 → 公開API → 探索上の制約**をつなぐ日本語実装ガイドである。

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

MES/PES/JESなど情報理論系の実体は原則BoTorch側を出典として追跡する。`ExpectedPredictiveInformationGain`はローカルのGaussian regression ALであり、分類BALDやMESと同一視しない。

参照する代表的な公開APIは ``robotorchan.acquisition.ExpectedPredictiveInformationGain``。ここでは**名前・責務の参照**であり、すべてのモデルやoptimizerとの組合せを実行検証済みとするものではない。

| 層 | 何を担当するか | 検証上の要点 |
| --- | --- | --- |
| surrogate | posterior、必要に応じてclass probability | posterior type、出力軸、学習方式 |
| acquisition | utilityの期待値・近似または探索score | 定義した数式、方向、q、sample形状 |
| semantic bridge | objective/constraintの意味論変換 | objectiveとclassification feasibilityの分離 |
| optimizer | 入力境界、fixed features、candidate constraintsを満たす候補検索 | one-shot/gradient/mixedの対応、最適化失敗の検出 |

## 4. 採用時の確認と注意事項

ローカルEPIGはq=1、single-outputのjoint Gaussian posteriorを前提とする。情報対象、近似法、sample数を記録する。

モデル側のposteriorが正しい形を返すことと、獲得関数の数学的同値性を示すことは別である。またacquisition functionの単体テストと、候補探索が可行な点を返すことは別々に検証する。

## 5. 数学的検証と実装の追跡

[既存Acquisition Mapping](../../acquisition.md)はsource inspectedの実装契約、entry・テスト証拠を保持する。[Verification Mapping](../../verification.md)は独立数値参照や既存のassertionの追跡先とする。両者を消去・置き換えず、原文区間のmigration ledgerをverifiedとする際に照合する。CI成功だけで全数値契約が検証されたことにはならない。

[獲得関数の基礎ガイド](../04_acquisition_function.md)、[獲得関数利用ドキュメント](../../../../optimization/README.md)、[理論Acquisition索引](../../../../theory/acquisition/README.md)にも戻れる。個別の数式、適用例や比較条件は対応する理論章と使用ガイドで読む。


## 6. MES・GIBBON・PES・JESの情報対象と委譲境界

旧Acquisition Mappingの固定区間346–362では、情報を獲得する**対象**によって手法を区別している。観測 $Y_X$ と対象 $T$ の条件付き相互情報量は

$$
I(Y_X;T\mid\mathcal D)=H(T\mid\mathcal D)
-\mathbb E_{Y_X\mid\mathcal D}[H(T\mid\mathcal D,Y_X)]
$$

である。MES/GIBBONの対象は最適**値** $f^\star$、PESは最適**位置** $x^\star$、JESは両者 $(x^\star,f^\star)$ である。この概念式は実際のBoTorch推定器が厳密に計算していることを意味しない。

| 手法 | APIと所有権 | 制約・未対応境界 |
| --- | --- | --- |
| MES | BoTorch qMaxValueEntropyを直接利用 | 有限candidate_setを使った最大値分布の近似。ローカルwrapper/registry entryなし |
| GIBBON | BoTorch qLowerBoundMaxValueEntropyを直接利用 | 最大値情報量の下界近似。MESと同一値ではない。ローカルwrapper/registry entryなし |
| PES・JES | 最適位置、位置と値の共同情報量 | 調査したローカルregistry/testには名前付きconstructorがなく、特定の実行経路は未検証 |
| ExpectedPredictiveInformationGain | robotorchan独自のGaussian regression AL | target observationsに関する情報量、q=1・単一出力joint Gaussianが前提。最大値情報量ではない |
| BALD | robotorchan分類AL | label/parameter情報量でありMES・EPIGと同義ではない |

MES/GIBBONを評価する最小例は、既存の tests/acquisition/test_information_theoretic.py の入力設定に対応する。

~~~python
import torch
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
    qMaxValueEntropy,
)
from robotorchan.models import SingleTaskGP

train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
train_y = torch.sin(train_x * 5.0)
model = SingleTaskGP(train_x, train_y)
candidate_set = torch.linspace(0.0, 1.0, 32, dtype=torch.double).unsqueeze(-1)
x = torch.tensor([[[0.4]]], dtype=torch.double)
mes = qMaxValueEntropy(model=model, candidate_set=candidate_set, num_mv_samples=4)
gibbon = qLowerBoundMaxValueEntropy(
    model=model,
    candidate_set=candidate_set,
    num_mv_samples=4,
)
mes_value = mes(x)
gibbon_value = gibbon(x)
~~~

ここでcandidate_setは[32,1]、評価入力xは[t-batch=1,q=1,d=1]、num_mv_samplesは4。上記テストがassertするのは両方の評価値が有限ということだけであり、独立情報量参照、推定収束、探索での最適候補の発見を検証しない。src/robotorchan/acquisition/registry.pyにはKG entryがあるがMES/GIBBON/PES/JESの名前付きentryはない。BoTorch直接利用とrobotorchanによるregistry保証を混同しない。関連するone-shot計算は[Lookaheadガイド](06_lookahead.md)を参照する。

**移行判定：partial / review required。** 既存ソースとassertionの静的照合であり、旧固定区間の全文意味論的検収や新規の数値実験は未実施。
