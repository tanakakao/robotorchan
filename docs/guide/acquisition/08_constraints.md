# Constrained Acquisition：制約の定義域と合成の責務

## 制約を二種類に分ける

[理論の制約付き獲得関数](../../theory/acquisition/08_constraints.md)を正本とする。目的と制約を観測して初めて可行性が分かる**outcome
constraint**と、実験前にXだけで可行か判断できる**candidate constraint**は別の層で処理する。
例えばモデル予測の制約残差`c(Y) <= 0`を実行可能確率やMC目的に反映するのはacquisition側、
設計変数の`g(X) >= 0`を満たすよう候補を探索するのはoptimizer側である。符号規約はAPIによって確認する必要がある。

## RegressionとClassificationの同時利用

`src/robotorchan/acquisition/composition.py`は`HeterogeneousModel`の出力と`ProblemSemantics`を、
`resolve_acquisition_composition`を介して`ObjectiveBinding`および`FeasibilityBinding`に対応付ける。
`AcquisitionCompositionPlan`は所有権と出力索引を保持し、この段階で統一されたposteriorやsamplerを生成しない。
従って回帰latent Gaussian値と分類class probabilityを**一つのGaussian posteriorに単純連結しない**。

- **`RegressionObjective` / `ProbabilityObjective`**
  役割と所有権：意味論的に目的変数を定義
  契約：maximize/minimizeと出力の意味を明示する

- **`ClassificationProbabilityOfFeasibility`**
  役割と所有権：classification probabilityを可行性として利用
  契約：latent `posterior`値ではなくclass probabilityを利用

- **`AcquisitionCompositionPlan`**
  役割と所有権：semantic binding（robotorchan側でImplemented）
  契約：`objectives`と`feasibility`の対応を記録。acquisitionそのものではない

- **`ConstrainedMCObjective`等**
  役割と所有権：BoTorch-native objective（Delegated）
  契約：sample shape / constraintの符号を揃える

- **`CandidateConstraints`等**
  役割と所有権：optimizer側の探索集合
  契約：outcome feasibilityとは独立にXの範囲を制限

`SampleShapeContract`は`sample_shape + batch_shape + [q]`をscalar sample-wise valueの期待形状とする。
この形状に分類可行確率をbroadcastする際、出力軸とq軸を混同してはならない。

## BoTorchへの委譲とapproximation

robotorchanの`composition.py`はBoTorchの`qExpectedImprovement`、`qNoisyExpectedImprovement`、
`qExpectedHypervolumeImprovement`等を部品として利用する。
各`make_*constrained*` helperは**acquisitionの新規アルゴリズムではなく**semantic adapter / builderである。
独立なclassification feasibilityを各点で乗じる場合、同一batch内の可行性の相関を正しく積分したjoint probabilityとは一般に一致しない。
独立性近似を無条件の厳密公式と説明しない。

## 制約付きqEI・qNEI・qEHVIの数式とconstructor対応

この章では、出力の意味と BoTorch の制約付き獲得関数を接続する処理を説明します。
以下のbuilderはすべてsrc/robotorchan/acquisition/composition.py に実装される。robotorchan独自の責務は意味論的な出力対応、向き・閾値の変換、
BoTorch部品の合成であり、qEI/qNEI/qEHVIの推定アルゴリズム自体はBoTorchへ委譲する。

- **make_continuous_constrained_qei_acquisition**
  BoTorchへの接続とローカル処理：qExpectedImprovement に ConstrainedMCObjective を渡す
  契約：回帰目的1個と共通posterior上の連続制約1個以上。best_f、sampler、infeasible_costを指定可能

- **make_mixed_constrained_qei_acquisition**
  BoTorchへの接続とローカル処理：上記native qEIを FeasibilityWeightedAcquisition で包む
  契約：回帰目的1個、連続制約と分類制約の両方が必須。重複分類出力は拒否

- **make_continuous_constrained_qnei_acquisition**
  BoTorchへの接続とローカル処理：qNoisyExpectedImprovement に ConstrainedMCObjective を渡す
  契約：連続制約のみ。X_baselineは非空の[n,d]。prune_baseline、cache_rootに対応

- **make_multiple_learned_constrained_qnei_acquisition**
  BoTorchへの接続とローカル処理：連続残差はnative qNEIのobjectiveへ、分類PoFは外側へ
  契約：少なくとも1制約。分類のみでも可。baseline分類可行性をjointに補正しない

- **make_constrained_qehvi_acquisition**
  BoTorchへの接続とローカル処理：qExpectedHypervolumeImprovement のnative constraintsと、任意の外部分類PoFを合成
  契約：複数回帰目的、1制約以上。向きを揃えたref_pointとpartitioningが一致し、eta > 0

- **make_deterministic_pof_acquisition**
  BoTorchへの接続とローカル処理：既存acquisitionと分類PoFを FeasibilityWeightedAcquisition で乗算
  契約：EI/HVを作成する独立アルゴリズムやjoint classifier posteriorではない

### 同じposteriorの連続制約と、分類確率の外部乗算

ContinuousConstraintの上限制約 y_j <= t は、make_continuous_constraint_bridge によって残差 y_j - t <= 0 に変換される。
下限制約では符号を反転する。posterior sampleが sample_shape + batch_shape + [q,m] なら、
目的と制約残差はそれぞれ sample_shape + batch_shape + [q] に対応する。

連続制約付き獲得値は概念的に

$$
\widehat A_{\mathrm{cont}}(X)
=\frac{1}{S}\sum_{s=1}^{S}
U(f^{(s)}(X),c^{(s)}(X);\mathrm{best\_f},\mathrm{infeasible\_cost})
$$

であり、同一BoTorch posteriorからの目的・制約サンプル間の依存性を保持する。この式は処理の模式図であって、
native qEI/qNEIのq方向集約やBoTorch ConstrainedMCObjectiveのinfeasible sample処理を厳密に再定義するものではない。
infeasible_costはこのnative目的に渡す代替値で、分類PoFとの外積とは異なる。best_fはObjectiveDirection.applyによって最大化方向に変換される。

一方、ClassificationProbabilityOfFeasibilityが返す分類可行性を p_j(x_i) とすると、
IndependentFeasibilityAggregator は分類出力間で積を計算し、
FeasibilityWeightedAcquisitionの既定 q_reduction="product" では

$$
A_{\mathrm{weighted}}(X)=A_{\mathrm{native}}(X)
\prod_{i=1}^{q}\prod_{j=1}^{J_{\mathrm{cls}}}p_j(x_i)
$$

となる。q_reduction="minimum" は分類PoFを出力方向に積にした後、q候補方向に最小値を取る。
この因子化はcandidate間・classifier間の依存性を明示的に積分しないため、joint可行確率の一般的な厳密式ではない。
FeasibilityWeightedAcquisitionは入力を batch_shape + [q,d]、PoFを batch_shape + [q ] と照合し、
t-batchごとに1スカラーを返す。

制約付きqNEIの X_baseline はnoiseを考慮した比較集合であり、外側の分類PoFはcandidate側の確率のみを掛ける。
制約付きqEHVIはcontinuous residualをBoTorchのnative constraintsへ渡し、ref_pointが目的数と一致する有限値で、
partitioning.ref_pointと完全一致することを検査する。制約付きqEHVIがあることから、未定義のlocal constrained qNEHVI builderの存在を推論しない。

## optimizerとの互換性

`src/robotorchan/acquisition/optimizer_compatibility.py`の
`check_acquisition_optimizer_compatibility(name, optimizer, q=...)`はregistryの`max_q`とone-shotの意味論を検査
する。通常の`CandidateOptimizer.MIXED`はone-shot augmented batchの行ごとのカテゴリ割当てを保存できないため、
登録されたone-shot acquisitionを拒否する。一方`MIXED_ONE_SHOT`はone-shot acquisition以外を拒否する。

これは**事前の静的互換性判定**であり、実際に候補点を探索して可行性を証明するものではない。candidate nonlinear constraintの全backend対応を認証するものでもない。

## ProbabilityObjectiveとClassificationConstraintを取り違えない

ProbabilityObjectiveは「分類確率を最適化する」という意味論的目的、
ClassificationProbabilityOfFeasibilityは「分類確率で可行性を評価する」という意味論的制約である。例えばpass確率を上げることと、
強度の獲得値をpass確率で重み付けすることは異なる探索目的になる。AcquisitionCompositionPlanは出力のownerを対応付けるだけであり、
その時点で共有posteriorや獲得関数を生成しない。

$$
\alpha_{\mathrm{weighted}}(X)=\alpha_{\mathrm{reg}}(X)
\prod_{i=1}^{q} P(\mathrm{pass}=1\mid x_i,\mathcal D)
$$

はFeasibilityWeightedAcquisitionのproduct reductionを使う場合の外部重み付けの模式式であり、
ProbabilityObjectiveを使ってpass確率自体を最大化する獲得関数と同一ではない。分類器間・候補間の相関を積分したjoint PoFでもない。

- **ProbabilityOfFeasibility**
  判定する値：周辺化済みP(feasible|X,D)
  主な契約：deterministicなper-candidate値。MC sample軸を持たない

- **SampleProbabilityOfFeasibility**
  判定する値：sampled class probability
  主な契約：supports_probability_samplesを持つモデルに限定

- **SampleResidualFeasibility**
  判定する値：sampled outcomeの残差
  主な契約：残差 <= 0で可行。BoTorch constrained MCと接続

- **ProbabilityResidualFeasibility**
  判定する値：閾値からの予測確率残差
  主な契約：決定論的な確率制約。サンプル制約やPoFと別

BoTorchのConstrainedMCObjectiveへ渡すsample-wise constraint、分類PoFを外側から乗じるwrapper、
候補Xに直接課すCandidateConstraintsの3層を混同しない。
make_sample_classification_feasibility_bridgeはepistemic class-probability sampleの能力確認を行い、
非対応ならTypeErrorを出す。サンプルの存在はpredict_probaの校正やjoint feasibility保証とは別である。

## 補足となる実装上の設計

#### Unknown output / black-box constraints：実装上の補足

BoTorch 的な構成を理解するには、次を分離することが重要です。

- 観測した応答のsampleに課す制約：残差が0以下なら可行とする。
- 分類モデルからの可行確率：latent値ではなく確率を使い、決定論的なPoFと確率sampleを区別する。
- 候補座標への既知制約：`CandidateConstraints`として候補optimizerへ渡す。

各層の対応を確認し、ある層の制約を別の層へ暗黙に変換しない。

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/37_outcome_constrained_bo.ipynb)を参照してください。
