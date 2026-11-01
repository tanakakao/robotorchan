# Multi-Fidelity / Cost-aware：理論から実装へ

## この章で解決する問い

低fidelityの安さだけを最大化せず、target fidelityを改善する情報とcostを同時に評価する。fidelity featureは単なるcategoryではない。

数学的な定義・仮定・導出は[理論の対応章](../../theory/acquisition/11_multifidelity_cost_aware.md)が正本であり、
この章は**数式の計算対象 →クラス構造 → 内部処理 → 近似と制限**をつなぐ日本語実装ガイドである。

## 実装の対応と委譲

`qMultiFidelityKnowledgeGradient`とcost-aware utilityは主としてBoTorch-native。
robotorchanの`SingleTaskMultiFidelityGP`、mixed/reduced variantsはposterior基盤を提供し、acquisitionとの接続を検査する。

## 採用時の確認と注意事項

target-fidelity projection、cost model、fantasy/one-shot route、fidelity次元固定、費用尺度を揃える。

### BoTorch / robotorchanとの対応

BoTorch は Multi-Fidelity Knowledge Gradient、cost-aware utility、fidelity cost model、
target-fidelity projection、および discrete fidelity を扱う acquisition optimization primitivesを提供しています。

robotorchan はこれらの標準機能を再実装せず、BoTorch native path を基本とします。

利用上の位置付けは
[Multi-Fidelity / Cost-aware optimization guide](../../optimization/multifidelity-cost-aware.md)、
現在の統合状況は
[統合契約](../../../docs-internal/contracts/optimization/acquisition-integration.md)
を参照してください。

## 補足となる実装上の設計

#### Multi-Fidelity MESは現在future extension：実装上の補足

MF-KGがterminal decision valueを基準にするのに対し、MF-MES系はoptimumに関するinformation gainを基準にします。
ただしMF-MESは現在のrobotorchan runtime-validated contractには含まれず、future extensionです。
理論上の存在をintegration supportの主張にしません。

#### まとめ：実装上の補足

- Wu, J., Toscano-Palmerin, S., Frazier, P. I., and Wilson, A. G. (2020), *Practical
Multi-fidelity Bayesian Optimization for Hyperparameter Tuning*. UAI.
- BoTorch documentation and tutorials for `qMultiFidelityKnowledgeGradient`, target-fidelity
projection, `AffineFidelityCostModel`, and `InverseCostWeightedUtility`.

## target projectionと費用によるutility変換

BoTorchの`qMultiFidelityKnowledgeGradient`は`qKnowledgeGradient`を継承します。
`forward()`は実候補とfantasy decisionを分離し、`project`を使う終端価値と、cost-aware utilityを組み合わせます。
投影先は最終的な意思決定を評価するfidelityであり、実候補の評価費用と役割が異なります。

[knowledge_gradient.py][kg]のこの経路と、
[cost_aware.py][cost]の`InverseCostWeightedUtility.forward()`が、
Theoryの「得られる価値と費用を合わせる」計算へ対応します。費用モデルによる値の変換を、robotorchanのsurrogate wrapperは再実装しません。
ローカルの[registry](../../../src/robotorchan/acquisition/registry.py)と
[compatibility](../../../src/robotorchan/acquisition/compatibility.py)は、
fidelity構造とposterior・fantasize能力の条件を照合します。

[kg]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/acquisition/knowledge_gradient.py
[cost]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/acquisition/cost_aware.py

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/03_multi_fidelity_gp.ipynb)を参照してください。
