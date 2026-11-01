# Multi-objective / EHVI：理論から実装へ

## この章で解決する問い

複数outputがあるだけでmulti-objective optimizationではない。Pareto dominance、reference point、
hypervolume improvementをutilityとして定義する。

数学的な定義・仮定・導出は[理論の対応章](../../theory/acquisition/07_multiobjective.md)が正本であり、
この章は**数式の計算対象 → クラス構造 → 内部処理 → 近似と制限**をつなぐ日本語実装ガイドである。

## 実装の対応と委譲

`qLogExpectedHypervolumeImprovement`・`qLogNoisyExpectedHypervolumeImprovement`はBoTorch-native。
`make_qehvi_acquisition`・`make_qnehvi_acquisition`はobjective/constraint bridgeを追加する。

## 採用時の確認と注意事項

reference pointと目的尺度の一致、partitioning、noise/baseline、feasible hypervolumeを区別する。

## 複数回帰目的と分類可行性を接続する

Pareto最適化ではm個の目的値ベクトルに対するhypervolumeを考える。すべて最大化方向にそろえた参照点rと非劣集合Pについて、増分hypervolumeの期待値は概念的に次の通り。

$$
\alpha_{\mathrm{EHVI}}(X)=
\mathbb E[\operatorname{HV}(P\cup Y(X);r)-\operatorname{HV}(P;r)].
$$

qEHVIは観測がnoise-freeであるという単純な意味ではなく、与えられたpartitioningとposteriorからMCでhypervolume improvementを見積もる。
qNEHVIはnoiseを含むbaselineの不確実性を扱い、X_baselineという入力点集合を受け取る。
qNEHVIとqEHVIに同一のconstructor引数やpartitioning入力を強制しない。

- **複数回帰目的・制約なし**
  公開builder：make_qehvi_acquisition
  前提と制約：共有するBoTorch regression posterior、directed ref_point、対応するBoxDecomposition

- **複数回帰目的・noisy・制約なし**
  公開builder：make_qnehvi_acquisition
  前提と制約：共有するregression posterior、nonempty X_baseline[n,d]、directed ref_point

- **複数回帰目的・連続／分類制約**
  公開builder：make_constrained_qehvi_acquisition
  前提と制約：1つ以上の制約。連続制約はobjectiveと同じposteriorに属し、分類PoFは外側で重み付け

- **制約付きqNEHVI**
  公開builder：当該composition.pyに対応するmake_constrained_qnehvi_acquisitionはない
  前提と制約：qEHVIのbuilderをqNEHVI用とみなさない。別経路のBoTorch利用可否を個別に検討

### ModelListとHeterogeneousのモデル所有権

src/robotorchan/acquisition/composition.pyのmake_botorch_multiobjective_bridgeは、
HeterogeneousModel内の複数回帰目的が**同じBoTorch model entry**を所有することを要求する。
ModelListGPは独立GPのjoint regression modelを形成できるが、
HeterogeneousModelの別entryに並べた独立回帰器から自動でshared posteriorを合成するわけではない。

連続制約は回帰posterior sampleと同じMC sampleの値を参照する。一方、
classificationのpredict_probaから得るPoFはposterior sampleの目的値ベクトルへ単純連結せず、
FeasibilityWeightedAcquisitionで外側から掛ける。例えばq候補とJ個の分類制約でproduct reductionを使うと、

$$
\alpha_{\mathrm{weighted}}(X)=
\alpha_{\mathrm{native}}(X)\prod_{i=1}^{q}\prod_{j=1}^{J}P(C_j(x_i)=1\mid\mathcal D).
$$

この積はcandidate間・分類器間の条件付き独立を暗黙に近似し、joint feasibilityを厳密積分した値とは一般に一致しない。classification probabilityの校正、
feasible hypervolumeの統計的正確さ、候補最適化の成功は別途検証する。

### directed reference pointと拒否条件

make_qehvi_acquisitionとmake_constrained_qehvi_acquisitionはref_pointが目的数と一致し有限値であり、
partitioning.ref_pointと**完全一致**することを要求する。これに対しqNEHVI builderはX_baselineを要求し、非空の[n,d]以外を拒否する。
参照点や目的値の符号が不整合ならhypervolumeの数学的意味が変わる。constrained qEHVIではetaが正値であること、分類出力重複がないことも検査する。

## 補足となる実装上の設計

#### まとめ：実装上の補足

- Daulton, S., Balandat, M., and Bakshy, E. (2020), *Differentiable Expected Hypervolume
Improvement for Parallel Multi-Objective Bayesian Optimization*. NeurIPS.
- Daulton, S., Balandat, M., and Bakshy, E. (2021), *Parallel Bayesian Optimization of Multiple
Noisy Objectives with Expected Hypervolume Improvement*. NeurIPS.
- Ament, S. et al. (2023), *Unexpected Improvements to Expected Improvement for Bayesian
Optimization*, for log-space improvement formulations used by modern BoTorch APIs.

## hypervolumeのセル分割とsample集約

BoTorchの`qLogExpectedHypervolumeImprovement`は`MultiObjectiveMCAcquisitionFunction`と
`SubsetIndexCachingMixin`を継承します。`forward()`はposterior sampleをobjectiveへ通し、
`_compute_log_qehvi()`が非支配領域のセル境界と候補部分集合の重なりを使ってlog-spaceの改善量を計算します。Theoryのhypervolumeを領域の和で評価する構成が、
このセル単位の計算に対応します。

`qLogNoisyExpectedHypervolumeImprovement`は`NoisyExpectedHypervolumeMixin`を加え、
baselineの不確実性を含む領域を扱います。これらは[BoTorchのmulti_objective/logei.py][hv]にあり、
robotorchanの[composition.py](../../../src/robotorchan/acquisition/composition.py)は出力の所有権、方向変換、
参照点を上流クラスへ接続します。

[hv]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/acquisition/multi_objective/logei.py

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/27_multiobjective_acquisition.ipynb)を参照してください
。
