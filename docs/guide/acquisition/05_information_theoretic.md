# Information-theoretic：理論から実装へ

## この章で解決する問い

観測の価値は最適点、最適値、Pareto解など指定した未知量についてのmutual information。対象を明記しない情報利得は不十分。

数学的な定義・仮定・導出は[理論の対応章](../../theory/acquisition/05_information_theoretic.md)が正本であり、
この章は**数式の計算対象 → クラス構造 → 内部処理 → 近似と制限**をつなぐ日本語実装ガイドである。

## 実装の対応と委譲

MES/PES/JESなど情報理論系の実体は原則BoTorch側を出典として追跡する。
`ExpectedPredictiveInformationGain`はローカルのGaussian regression ALであり、分類BALDやMESと同一視しない。

## 採用時の確認と注意事項

ローカルEPIGはq=1、single-outputのjoint Gaussian posteriorを前提とする。情報対象、近似法、sample数を記録する。

## MES・GIBBON・PES・JESの情報対象と委譲境界

$$
I(Y_X;T\mid\mathcal D)=H(T\mid\mathcal D)
-\mathbb E_{Y_X\mid\mathcal D}[H(T\mid\mathcal D,Y_X)]
$$

である。MES/GIBBONの対象は最適**値** $f^\star$、PESは最適**位置** $x^\star$、JESは両者 $(x^\star,f^\star)$ である。
この概念式は実際のBoTorch推定器が厳密に計算していることを意味しない。

- **MES**
  APIと所有権：BoTorch qMaxValueEntropyを直接利用
  制約・未対応境界：有限candidate_setを使った最大値分布の近似。ローカルwrapper/registry entryなし

- **GIBBON**
  APIと所有権：BoTorch qLowerBoundMaxValueEntropyを直接利用
  制約・未対応境界：最大値情報量の下界近似。MESと同一値ではない。ローカルwrapper/registry entryなし

- **PES・JES**
  APIと所有権：最適位置、位置と値の共同情報量
  制約・未対応境界：ローカルregistryに名前付きentryを持たない。上流の推定器と接続条件を区別する

- **ExpectedPredictiveInformationGain**
  APIと所有権：robotorchan独自のGaussian regression AL
  制約・未対応境界：target observationsに関する情報量、q=1・単一出力joint Gaussianが前提。最大値情報量ではない

- **BALD**
  APIと所有権：robotorchan分類AL
  制約・未対応境界：label/parameter情報量でありMES・EPIGと同義ではない

ここでcandidate_setは[32,1]、評価入力xは[t-batch=1,q=1,d=1]、num_mv_samplesは4。
上記テストがassertするのは両方の評価値が有限ということだけであり、独立情報量参照、推定収束、探索での最適候補の発見を検証しない。
src/robotorchan/acquisition/registry.pyにはKG entryがあるがMES/GIBBON/PES/JESの名前付きentryはない。
BoTorch直接利用とrobotorchanによるregistry保証を混同しない。関連するone-shot計算は
[Lookaheadガイド](06_lookahead.md)を参照する。

### BoTorch / robotorchan との対応

BoTorchはMESとGIBBONを含むinformation-theoretic acquisitionを提供しています。
robotorchanはBoTorch-nativeで利用可能な標準手法を再実装しません。

現在のintegration guideはMES / GIBBONをnative pathとして扱います。一方、
robotorchanのcapability registryはBoTorch acquisition全件のcatalogではなく、
MES / GIBBONのentryがないことを「利用不可」とは解釈しません。

またinformation-theoretic acquisitionのposterior requirementは手法ごとに確認します。
「information-theoreticだからPOSTERIOR_SAMPLES」と一括して決めず、optimum-value sampling、conditional entropy、
joint Gaussian structure等、実際の計算経路が要求するcontractを見ます。

利用上の位置付けは
[Information-theoretic optimization guide](../../optimization/information_theoretic.md)、
現在の統合状況は
[統合契約](../../../docs-internal/contracts/optimization/acquisition-integration.md)
を参照してください。

## 補足となる実装上の設計

#### この章で覚えておくこと：実装上の補足

- information-theoretic acquisitionでは最初にinformation targetを確認する
- entropyが大きい点を選ぶこととmutual informationを最大化することは同じではない
- optimizer locationとoptimum valueは別のunknown quantity
- PESはoptimizer location、MESはoptimum valueを代表的なinformation targetとする
- MES criterionとoptimum-value sampling approximationを分けて考える
- GIBBONはMESを単純にq点へ並べたものではなくbatch redundancyを考慮する
- observation noise由来のentropyが高くてもoptimum informationが大きいとは限らない
- MESとKGはinformation gainとdecision valueという異なる価値基準を持つ
- registryにentryがないことをBoTorch-native acquisitionの利用不可とは解釈しない

#### まとめ：実装上の補足

- Wang, Z. and Jegelka, S. (2017), *Max-value Entropy Search for Efficient Bayesian Optimization*.
  ICML.
- Moss, H. B. et al. (2021), *GIBBON: General-purpose Information-Based Bayesian Optimisation*.
  JMLR.
- BoTorch documentation, *Acquisition Functions*, for current native MES and lower-bound max-value
  entropy interfaces.

## 情報対象ごとの実装経路

BoTorchの`qMaxValueEntropy`は`MaxValueBase`と`MCSamplerMixin`を継承します。
`_sample_max_values()`が有限候補集合を使って最適値のsampleを準備し、
`_compute_information_gain()`がその条件の下で観測entropyの差を近似します。`qLowerBoundMaxValueEntropy`は同じ基底を使いますが、
情報量の下界を計算します。この違いは[BoTorchのmax_value_entropy_search.py][m]に実装されています。

一方、robotorchanの[epig.py](../../../src/robotorchan/acquisition/active_learning/epig.py)の
`ExpectedPredictiveInformationGain`は`AcquisitionFunction`を継承する独自実装です。
`forward()`は候補と各targetを2点組にしてjoint posteriorの共分散を取り出し、観測noiseを加えた2つの分散から相関の二乗 $\rho^2$ を計算します。
Gaussianの相互情報量 $-\tfrac12\log(1-\rho^2)$ を`log1p`で評価し、target weightsで平均します。
分母と相関には浮動小数点に応じたclampが入ります。

ここで周辺化する対象はtarget predictionであり、最適値ではありません。有限target集合・joint Gaussian・q=1という近似上の条件も、
同じ`forward()`内のshape・出力数・ensemble拒否に対応します。

[m]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/acquisition/max_value_entropy_search.py
