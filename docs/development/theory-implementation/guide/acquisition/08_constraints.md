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

## 4. optimizerとの互換性

`src/robotorchan/acquisition/optimizer_compatibility.py`の`check_acquisition_optimizer_compatibility(name, optimizer, q=...)`はregistryの`max_q`とone-shotの意味論を検査する。通常の`CandidateOptimizer.MIXED`はone-shot augmented batchの行ごとのカテゴリ割当てを保存できないため、登録されたone-shot acquisitionを拒否する。一方`MIXED_ONE_SHOT`はone-shot acquisition以外を拒否する。

これは**事前の静的互換性判定**であり、実際に候補点を探索して可行性を証明するものではない。candidate nonlinear constraintの全backend対応を認証するものでもない。

## 5. 検証と未検証の境界

[旧Acquisition Mapping](../../acquisition.md)はBoTorch delegation・分類可行性の合成・one-shot制限を保全する。[旧Optimization Mapping](../../optimization.md)はcandidate feasibility側の責務を記す。[Verification Mapping](../../verification.md)と[検証索引](../verification/acquisition.md)で個別propertyの独立数値参照を確認する。

CIでリンクや契約テストが成功しても、任意の制約付き関数に対する探索最適性、joint feasibilityの厳密性、全てのoptimizerとの接続成功は証明されない。旧Mappingは移行元として保存する。
