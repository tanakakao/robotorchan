# Multi-objective / EHVI：理論から実装へ

## 1. この章で解決する問い

複数outputがあるだけでmulti-objective optimizationではない。Pareto dominance、reference point、hypervolume improvementをutilityとして定義する。

数学的な定義・仮定・導出は[理論の対応章](../../../../theory/acquisition/07_multiobjective.md)が正本であり、この章は**数式の計算対象 → 実装の所有権 → 公開API → 探索上の制約**をつなぐ日本語実装ガイドである。

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

`qLogExpectedHypervolumeImprovement`・`qLogNoisyExpectedHypervolumeImprovement`はBoTorch-native。`make_qehvi_acquisition`・`make_qnehvi_acquisition`はobjective/constraint bridgeを追加する。

参照する代表的な公開APIは ``botorch.acquisition.multi_objective.logei.qLogExpectedHypervolumeImprovement``。ここでは**名前・責務の参照**であり、すべてのモデルやoptimizerとの組合せを実行検証済みとするものではない。

| 層 | 何を担当するか | 検証上の要点 |
| --- | --- | --- |
| surrogate | posterior、必要に応じてclass probability | posterior type、出力軸、学習方式 |
| acquisition | utilityの期待値・近似または探索score | 定義した数式、方向、q、sample形状 |
| semantic bridge | objective/constraintの意味論変換 | objectiveとclassification feasibilityの分離 |
| optimizer | 入力境界、fixed features、candidate constraintsを満たす候補検索 | one-shot/gradient/mixedの対応、最適化失敗の検出 |

## 4. 採用時の確認と注意事項

reference pointと目的尺度の一致、partitioning、noise/baseline、feasible hypervolumeを区別する。

モデル側のposteriorが正しい形を返すことと、獲得関数の数学的同値性を示すことは別である。またacquisition functionの単体テストと、候補探索が可行な点を返すことは別々に検証する。

## 5. 数学的検証と実装の追跡

[既存Acquisition Mapping](../../acquisition.md)はsource inspectedの実装契約、entry・テスト証拠を保持する。[Verification Mapping](../../verification.md)は独立数値参照や既存のassertionの追跡先とする。両者を消去・置き換えず、原文区間のmigration ledgerをverifiedとする際に照合する。CI成功だけで全数値契約が検証されたことにはならない。

[獲得関数の基礎ガイド](../04_acquisition_function.md)、[獲得関数利用ドキュメント](../../../../optimization/README.md)、[理論Acquisition索引](../../../../theory/acquisition/README.md)にも戻れる。個別の数式、適用例や比較条件は対応する理論章と使用ガイドで読む。

## 6. 複数回帰目的と分類可行性を接続する

Pareto最適化ではm個の目的値ベクトルに対するhypervolumeを考える。すべて最大化方向にそろえた参照点rと非劣集合Pについて、増分hypervolumeの期待値は概念的に次の通り。

$$
\alpha_{\mathrm{EHVI}}(X)=
\mathbb E[\operatorname{HV}(P\cup Y(X);r)-\operatorname{HV}(P;r)].
$$

qEHVIは観測がnoise-freeであるという単純な意味ではなく、与えられたpartitioningとposteriorからMCでhypervolume improvementを見積もる。qNEHVIはnoiseを含むbaselineの不確実性を扱い、X_baselineという入力点集合を受け取る。qNEHVIとqEHVIに同一のconstructor引数やpartitioning入力を強制しない。

| 目的・制約 | 公開builder | 前提と制約 |
| --- | --- | --- |
| 複数回帰目的・制約なし | make_qehvi_acquisition | 共有するBoTorch regression posterior、directed ref_point、対応するBoxDecomposition |
| 複数回帰目的・noisy・制約なし | make_qnehvi_acquisition | 共有するregression posterior、nonempty X_baseline[n,d]、directed ref_point |
| 複数回帰目的・連続／分類制約 | make_constrained_qehvi_acquisition | 1つ以上の制約。連続制約はobjectiveと同じposteriorに属し、分類PoFは外側で重み付け |
| 制約付きqNEHVI | 当該composition.pyに対応するmake_constrained_qnehvi_acquisitionはない | qEHVIのbuilderをqNEHVI用とみなさない。別経路のBoTorch利用可否を個別に検討 |

### 6.1 ModelListとHeterogeneousのモデル所有権

src/robotorchan/acquisition/composition.pyのmake_botorch_multiobjective_bridgeは、HeterogeneousModel内の複数回帰目的が**同じBoTorch model entry**を所有することを要求する。ModelListGPは独立GPのjoint regression modelを形成できるが、HeterogeneousModelの別entryに並べた独立回帰器から自動でshared posteriorを合成するわけではない。

連続制約は回帰posterior sampleと同じMC sampleの値を参照する。一方、classificationのpredict_probaから得るPoFはposterior sampleの目的値ベクトルへ単純連結せず、FeasibilityWeightedAcquisitionで外側から掛ける。例えばq候補とJ個の分類制約でproduct reductionを使うと、

$$
\alpha_{\mathrm{weighted}}(X)=
\alpha_{\mathrm{native}}(X)\prod_{i=1}^{q}\prod_{j=1}^{J}P(C_j(x_i)=1\mid\mathcal D).
$$

この積はcandidate間・分類器間の条件付き独立を暗黙に近似し、joint feasibilityを厳密積分した値とは一般に一致しない。classification probabilityの校正、feasible hypervolumeの統計的正確さ、候補最適化の成功は別途検証する。

### 6.2 directed reference pointと拒否条件

make_qehvi_acquisitionとmake_constrained_qehvi_acquisitionはref_pointが目的数と一致し有限値であり、partitioning.ref_pointと**完全一致**することを要求する。これに対しqNEHVI builderはX_baselineを要求し、非空の[n,d]以外を拒否する。参照点や目的値の符号が不整合ならhypervolumeの数学的意味が変わる。constrained qEHVIではetaが正値であること、分類出力重複がないことも検査する。

**根拠：** src/robotorchan/acquisition/composition.py のmake_qehvi_acquisition、make_qnehvi_acquisition、make_constrained_qehvi_acquisitionと、models/heterogeneous.py。上記は静的ソース照合に基づく解説であり、joint feasibility、数値hypervolume誤差、end-to-end最適化を今回再実行したものではない。
