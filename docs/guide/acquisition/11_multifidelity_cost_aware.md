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


## CostBudgetとMF-KGのclosed-loopの接続

[費用管理の実装](../../../src/robotorchan/workflow/cost_budget.py)の
`CostBudget`は、予算、支出済み費用、未精算の予約を保持します。
`can_afford()`で予測費用を確認し、`reserve()`で評価IDに紐付け、
実験完了後に`settle()`で実費へ置き換えます。
これは**実行時の会計・予約機構**であり、
`InverseCostWeightedUtility`の価値計算を置き換えません。

[MF-KG逐次例](../../../examples/mfkg_cost_budget_loop.py)は、
`SingleTaskMultiFidelityGP.fit()`で目的関数モデルを更新し、
BoTorchの`PosteriorMean`と`optimize_acqf`でtarget fidelity上の
`current_value`を求めます。次に
`qMultiFidelityKnowledgeGradient`のone-shot候補最適化へ進み、
候補の予測費用を予約して実測費用を精算します。
実験の実行自体はrobotorchanのsurrogate・acquisitionの外側です。

費用が未知の場合の[学習型コスト例](../../../examples/learned_multifidelity_cost.py)は、
別の`SingleTaskGP`で`log(cost)`を学習し、
正の値へ変換した費用をBoTorchのcost utilityに渡します。
`exp(E[log(cost)])`と`E[cost]`は一般に一致しないため、
学習型費用の意味を区別します。

## TuRBOとの組み合わせと状態更新

[TuRBO統合例](../../../examples/turbo_multifidelity_cost_budget.py)では、
`TuRBOStrategy.optimize_multifidelity()`にMF-KGを渡します。
既存のTuRBOが設計変数側のTrust Regionを扱い、
fidelity次元の探索範囲は大域のままです。

`TuRBOStrategy.update_state()`の`state_values`には
target fidelityへ射影した候補の**評価前posterior mean**を渡します。
これは観測されたtarget fidelity値ではありません。
TuRBOの改善判定が推定値に依存する点は、この接続の重要な制限です。

## 検証経路と未対応の境界

[費用比較ベンチマーク](../../../benchmarks/multifidelity_cost_budget.py)は
MF-KGとrandomを同じ評価費用予算で比較し、
[Notebook](../../../examples/notebooks/37_multifidelity_cost_aware.ipynb)で可視化します。
`best_target_oracle`は合成関数から求める診断値で、
学習や実験予算に含めていません。

MF-MES専用の実行検証、予算超過候補の再最適化、
非同期の予約管理、joint q-batchの予算制約、
TuRBOとMF-KGの統計的な優越性は保証していません。
現時点の対応・制限は
[Multi-Fidelity / Cost-aware optimization guide](../../optimization/multifidelity-cost-aware.md)
に集約します。
