# Constrained Acquisition：制約の定義域と合成の責務

## 1. 制約を二種類に分ける

[理論の制約付き獲得関数](../../../../theory/acquisition/08_constraints.md)を正本とする。目的と制約を観測して初めて可行性が分かる**outcome constraint**と、実験前にXだけで可行か判断できる**candidate constraint**は別の層で処理する。例えばモデル予測の制約残差`c(Y) <= 0`を実行可能確率やMC目的に反映するのはacquisition側、設計変数の`g(X) >= 0`を満たすよう候補を探索するのはoptimizer側である。符号規約はAPIによって確認する必要がある。

## 2. RegressionとClassificationの同時利用

`src/robotorchan/acquisition/composition.py`は`HeterogeneousModel`の出力と`ProblemSemantics`を、`resolve_acquisition_composition`を介して`ObjectiveBinding`および`FeasibilityBinding`に対応付ける。`AcquisitionCompositionPlan`は所有権と出力索引を保持し、この段階で統一されたposteriorやsamplerを生成しない。従って回帰latent Gaussian値と分類class probabilityを**一つのGaussian posteriorに単純連結しない**。

| 要素 | 役割と所有権 | 契約 |
| --- | --- | --- |
| `RegressionObjective` / `ProbabilityObjective` | 意味論的に目的変数を定義 | maximize/minimizeと出力の意味を明示する |
| `ClassificationProbabilityOfFeasibility` | classification probabilityを可行性として利用 | latent `posterior`値ではなくclass probabilityを利用 |
| `AcquisitionCompositionPlan` | semantic binding（robotorchan側でImplemented） | `objectives`と`feasibility`の対応を記録。acquisitionそのものではない |
| `ConstrainedMCObjective`等 | BoTorch-native objective（Delegated） | sample shape / constraintの符号を揃える |
| `CandidateConstraints`等 | optimizer側の探索集合 | outcome feasibilityとは独立にXの範囲を制限 |

`SampleShapeContract`は`sample_shape + batch_shape + [q]`をscalar sample-wise valueの期待形状とする。この形状に分類可行確率をbroadcastする際、出力軸とq軸を混同してはならない。

## 3. BoTorchへの委譲とapproximation

robotorchanの`composition.py`はBoTorchの`qExpectedImprovement`、`qNoisyExpectedImprovement`、`qExpectedHypervolumeImprovement`等を部品として利用する。各`make_*constrained*` helperは**acquisitionの新規アルゴリズムではなく**semantic adapter / builderである。独立なclassification feasibilityを各点で乗じる場合、同一batch内の可行性の相関を正しく積分したjoint probabilityとは一般に一致しない。独立性近似を無条件の厳密公式と説明しない。


## 4. 制約付きqEI・qNEI・qEHVIの数式とconstructor対応

旧Acquisition Mappingの固定区間288–308（および直後のInterpretation）に記載された技術契約を、Phase 3 routingに基づきこの章の本文へ反映する。以下のbuilderはすべて src/robotorchan/acquisition/composition.py に実装される。robotorchan独自の責務は意味論的な出力対応、向き・閾値の変換、BoTorch部品の合成であり、qEI/qNEI/qEHVIの推定アルゴリズム自体はBoTorchへ委譲する。

| builder | BoTorchへの接続とローカル処理 | 契約 |
| --- | --- | --- |
| make_continuous_constrained_qei_acquisition | qExpectedImprovement に ConstrainedMCObjective を渡す | 回帰目的1個と共通posterior上の連続制約1個以上。best_f、sampler、infeasible_costを指定可能 |
| make_mixed_constrained_qei_acquisition | 上記native qEIを FeasibilityWeightedAcquisition で包む | 回帰目的1個、連続制約と分類制約の両方が必須。重複分類出力は拒否 |
| make_continuous_constrained_qnei_acquisition | qNoisyExpectedImprovement に ConstrainedMCObjective を渡す | 連続制約のみ。X_baselineは非空の[n,d]。prune_baseline、cache_rootに対応 |
| make_multiple_learned_constrained_qnei_acquisition | 連続残差はnative qNEIのobjectiveへ、分類PoFは外側へ | 少なくとも1制約。分類のみでも可。baseline分類可行性をjointに補正しない |
| make_constrained_qehvi_acquisition | qExpectedHypervolumeImprovement のnative constraintsと、任意の外部分類PoFを合成 | 複数回帰目的、1制約以上。向きを揃えたref_pointとpartitioningが一致し、eta > 0 |
| make_deterministic_pof_acquisition | 既存acquisitionと分類PoFを FeasibilityWeightedAcquisition で乗算 | EI/HVを作成する独立アルゴリズムやjoint classifier posteriorではない |

### 4.1 同じposteriorの連続制約と、分類確率の外部乗算

ContinuousConstraintの上限制約 y_j <= t は、make_continuous_constraint_bridge によって残差 y_j - t <= 0 に変換される。下限制約では符号を反転する。posterior sampleが sample_shape + batch_shape + [q,m] なら、目的と制約残差はそれぞれ sample_shape + batch_shape + [q] に対応する。

連続制約付き獲得値は概念的に

$
\widehat A_{\mathrm{cont}}(X)
=\frac{1}{S}\sum_{s=1}^{S}
U(f^{(s)}(X),c^{(s)}(X);\mathrm{best\_f},\mathrm{infeasible\_cost})
$

であり、同一BoTorch posteriorからの目的・制約サンプル間の依存性を保持する。この式は処理の模式図であって、native qEI/qNEIのq方向集約やBoTorch ConstrainedMCObjectiveのinfeasible sample処理を厳密に再定義するものではない。infeasible_costはこのnative目的に渡す代替値で、分類PoFとの外積とは異なる。best_fはObjectiveDirection.applyによって最大化方向に変換される。

一方、ClassificationProbabilityOfFeasibilityが返す分類可行性を p_j(x_i) とすると、IndependentFeasibilityAggregator は分類出力間で積を計算し、FeasibilityWeightedAcquisitionの既定 q_reduction="product" では

$
A_{\mathrm{weighted}}(X)
=A_{\mathrm{native}}(X)
\prod_{i=1}^{q}\prod_{j=1}^{J_{\mathrm{cls}}}p_j(x_i)
$

となる。q_reduction="minimum" は分類PoFを出力方向に積にした後、q候補方向に最小値を取る。この因子化はcandidate間・classifier間の依存性を明示的に積分しないため、joint可行確率の一般的な厳密式ではない。FeasibilityWeightedAcquisitionは入力を batch_shape + [q,d]、PoFを batch_shape + [q] と照合し、t-batchごとに1スカラーを返す。

制約付きqNEIの X_baseline はnoiseを考慮した比較集合であり、外側の分類PoFはcandidate側の確率のみを掛ける。制約付きqEHVIはcontinuous residualをBoTorchのnative constraintsへ渡し、ref_pointが目的数と一致する有限値で、partitioning.ref_pointと完全一致することを検査する。制約付きqEHVIがあることから、未定義のlocal constrained qNEHVI builderの存在を推論しない。

### 4.2 最小コード例：連続制約付きqEI

下記は出力0を最大化し、出力1を0.8以下にする例。train_yは[n,2]で、候補Xは[q,d]。これはacquisition値の評価であり、候補探索やfeasible incumbentの自動認証ではない。

~~~python
import torch
from robotorchan.acquisition.composition import make_continuous_constrained_qei_acquisition
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard import SingleTaskGP
from robotorchan.semantics import (
    ContinuousConstraint, ProblemSemantics, RegressionObjective,
)

train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
train_y = torch.cat((torch.sin(train_x), torch.cos(train_x)), dim=-1)
model = HeterogeneousModel(SingleTaskGP(train_x, train_y))
semantics = ProblemSemantics(
    objectives=(RegressionObjective(0),),
    constraints=(ContinuousConstraint(1, threshold=0.8),),
)
acq = make_continuous_constrained_qei_acquisition(
    model, semantics, best_f=0.5, infeasible_cost=0.0,
)
value = acq(torch.tensor([[0.25]], dtype=torch.double))
# 既存契約テストではshape [1]を確認する。
~~~

### 4.3 拒否条件と検証水準

- 連続qEI/qNEIでは目的と連続制約が異なるmodel entryにあると拒否する。混合qEI、複数制約qNEI、制約付きqEHVIでもcontinuous residualは目的と同じBoTorch posteriorでなければならない。
- 混合qEIは連続・分類の両種類を要求する。分類出力の重複指定は拒否する。分類だけの制約には混合qEI builderを流用しない。
- FeasibilityWeightedAcquisitionは非空のX_pendingとjoint fantasizeを拒否する。q_reductionは product または minimum のみ。
- make_sample_classification_feasibility_bridge はepistemic class-probability samplesをサポートするclassifier用の別経路であり、deterministic PoFが暗黙に利用するものではない。
- 根拠：src/robotorchan/acquisition/composition.py、classification_constraints.py、tests/acquisition/test_composition.py、test_phase5_native_integration.py。既存assertionをソースで照合した段階であり、今回独立数値実験を再実行したわけではない。分類PoFの校正、joint近似の統計的妥当性、noisy improvement/HV推定の一般数値精度、安全保証は未検証とする。

## 5. optimizerとの互換性

`src/robotorchan/acquisition/optimizer_compatibility.py`の`check_acquisition_optimizer_compatibility(name, optimizer, q=...)`はregistryの`max_q`とone-shotの意味論を検査する。通常の`CandidateOptimizer.MIXED`はone-shot augmented batchの行ごとのカテゴリ割当てを保存できないため、登録されたone-shot acquisitionを拒否する。一方`MIXED_ONE_SHOT`はone-shot acquisition以外を拒否する。

これは**事前の静的互換性判定**であり、実際に候補点を探索して可行性を証明するものではない。candidate nonlinear constraintの全backend対応を認証するものでもない。

## 6. 検証と未検証の境界

[旧Acquisition Mapping](../../acquisition.md)はBoTorch delegation・分類可行性の合成・one-shot制限を保全する。[旧Optimization Mapping](../../optimization.md)はcandidate feasibility側の責務を記す。[Verification Mapping](../../verification.md)と[検証索引](../verification/acquisition.md)で個別propertyの独立数値参照を確認する。

CIでリンクや契約テストが成功しても、任意の制約付き関数に対する探索最適性、joint feasibilityの厳密性、全てのoptimizerとの接続成功は証明されない。旧Mappingは移行元として保存する。
