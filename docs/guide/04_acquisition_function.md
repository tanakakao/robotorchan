# Acquisition Function：予測分布を効用へ写す実装

## この章で学ぶこと

[獲得関数のTheory](../theory/04_acquisition_function.md)の4.3・4.5・4.9節を、
予測分布、目的変換、標本、効用集約の処理へ対応付けます。
期待効用の導出と個別手法の式はTheoryへ集約し、
本章では標準獲得関数への委譲と、robotorchanが追加する接続処理を説明します。

獲得関数の評価と、その最大点を探索する処理は別です。
獲得関数は候補のスコアを返し、候補探索はそのスコアを繰り返し評価します。
どちらも、高価な目的関数を実験で観測する処理とは異なります。

## 1. 構成要素とクラスの関係

| 構成要素 | 主な実装 |
| --- | --- |
| 獲得関数の基底 | BoTorchの`AcquisitionFunction` |
| 単一点の閉形式評価 | `AnalyticAcquisitionFunction`と個別の`forward()` |
| 予測標本と目的値の生成 | `MCAcquisitionFunction._get_samples_and_objectives()` |
| 標本・q次元の集約 | `SampleReducingMCAcquisitionFunction.forward()` |
| 出力と目的の対応解決 | robotorchanの`resolve_acquisition_composition()` |
| 目的値の向き・出力選択 | `RegressionObjective.to_botorch()` |
| 独自の不確実性スコア | `PosteriorVariance`・`PosteriorStd`など |

標準獲得関数の本体は[BoTorchのacquisition][b-acquisition]にあります。
robotorchanは標準クラスを一括で複製・再exportする層ではありません。
[composition.py][composition]は、目的・制約を上流のインターフェースへつなぐ層です。
[registry.py][registry]は、目的の種類や必要な予測情報などのmetadataを保持します。
registryの検索自体が獲得関数の計算や候補探索を実行するわけではありません。

## 2. Analytic獲得関数の内部

BoTorchの`analytic.py`にある`AnalyticAcquisitionFunction._mean_and_sigma()`は、
`model.posterior()`から平均と分散を取得します。
標準偏差を使う場合は、分散に下限を設けて平方根を取り、qの次元を取り除きます。
`LogExpectedImprovement`や`UpperConfidenceBound`は、この共通処理を利用します。
これらの代表的な`forward()`は、decoratorでq=1の入力を要求します。

`LogExpectedImprovement.forward()`は、平均、標準偏差、`best_f`、目的の向きから
標準化した改善量を作り、`_log_ei_helper()`と標準偏差の対数を組み合わせます。
通常のEIを一度計算してから対数を取る処理ではなく、
小さな改善量を対数領域で扱う設計です。

`UpperConfidenceBound.forward()`は、方向をそろえた平均と、
`beta.sqrt()`を係数とする標準偏差を加えます。
Theoryの不確実性の重みと、実装引数`beta`の平方根を対応付けて読む必要があります。
これらの式の計算はBoTorchへのDelegatedです。

## 3. MC獲得関数の内部

MC経路では、平均・分散だけでなく候補間の依存を含む同時事後分布を利用します。
`MCAcquisitionFunction._get_samples_and_objectives()`は、
`posterior_transform`を指定して分布を取得し、`get_posterior_samples()`で標本を生成し、
`objective(samples, X)`で標本を目的値へ変換します。

標本の形は概念的に`sample_shape × batch_shape × q × m`です。
目的変換後には、最後の出力次元mを効用の定義に従って処理します。
標本次元、モデルや入力のbatch次元、同時候補数qは異なる役割を持ちます。
点ごとの周辺分布から独立に標本を作ることでは、同じ同時分布を再現できません。

`SampleReducingMCAcquisitionFunction._non_reduced_forward()`は、
各標本の目的値を個別の`_sample_forward()`へ渡し、必要なら制約重みを適用します。
`forward()`はq次元の集約に続いて標本次元を集約します。

例えば`logei.py`の`qLogExpectedImprovement`は、
`LogImprovementMCAcquisitionFunction`を介してこの構造を継承します。
`_sample_forward()`は`_log_improvement()`へ進み、改善量を対数領域で扱います。
この系統では平滑化や対数領域の集約も使うため、
全てのMCクラスを単純な「各標本の効用の算術平均」という実装に置き換えて説明しません。

基底標本の生成はsamplerの責務です。
通常の乱数とQMC標本では生成規則が異なりますが、
獲得関数のどの標本次元を集約するかという役割は保たれます。
詳しくは[獲得関数の基礎Guide](acquisition/01_foundations.md)へ進めます。

## 4. robotorchanの目的変換と出力の所有関係

異種モデルをまとめた場合、全体での出力番号と、個々のモデル内部での出力番号は異なります。
`resolve_acquisition_composition()`は`ProblemSemantics`を検査し、
`HeterogeneousModel.output_owner()`から所有モデルと内部出力番号を解決します。
結果は`ObjectiveBinding`・`FeasibilityBinding`を持つ
`AcquisitionCompositionPlan`です。
この段階では、モデル間の共通posteriorやsamplerを作りません。

[semantics/objectives.py][objectives]の`RegressionObjective.to_botorch()`は、
標本の`local_output_index`を選び、`ObjectiveDirection.apply()`で方向をそろえる
`GenericMCObjective`を構成します。
[direction.py][direction]では、最大化ならそのまま、最小化なら符号を反転します。
単なる出力列の選択と、目的の向きを含む効用への変換を区別します。

`make_botorch_objective_bridge()`は、bindingが現在のモデルと一致し、
所有モデルがBoTorchの`Model`であることを確認して、モデルとobjectiveを返します。
`make_qei_acquisition()`は、単一の無制約回帰目的であることを確認し、
`best_f`にも同じ方向変換を適用して`qExpectedImprovement`を構成します。
このhelperが返すのはqEIであり、LogEIへ暗黙に置き換える処理はありません。

このようにrobotorchanは接続と検査をImplemented/Extendedとして担い、
EIの標本処理や集約は上流へ委譲します。
複数の回帰目的を扱うbridgeは、一つのモデルが所有する同時事後分布を要求します。
別々のモデルの周辺分布を、自動的に共通の同時事後分布とみなすことはしません。

## 5. 制約と独自スコアの処理

連続応答の制約と、候補入力そのものの制約は別の層で処理します。
`make_continuous_constrained_qei_acquisition()`は、目的と制約が
同じBoTorchモデルに属することを確認し、標本上の残差を
`ConstrainedMCObjective`へ接続します。
候補の境界や点間制約は、[候補制約Guide](optimization/05_constraints.md)の最適化側が担当します。

分類確率を扱う場合は、予測確率の評価と、確率自体の標本生成を区別します。
`ProbabilityObjectiveBridge`は、確率標本を生成する能力がない分類器に対して
標本生成を拒否します。分類確率をGaussian潜在関数の値と同一視しません。
詳細な可行性の合成は[制約Guide](acquisition/08_constraints.md)に分けます。

一方、[active_learning/variance.py][variance]の`PosteriorVariance`は、
BoTorchの`AcquisitionFunction`を継承し、分散のスコア計算を独自に実装します。
`forward()`はq=1と出力構造を検査し、必要な出力を選んで`posterior.variance`を返します。
`PosteriorStd`は、その値を非負へ切り詰めて平方根を取ります。
これらの経路はensemble posteriorを拒否し、複数出力では出力選択を要求します。

[Straddle][straddle]も独自の`forward()`で、標準偏差への重みから
平均と目標水準の距離を引きます。
標準BOのEI、分類のBALD、回帰のStraddleを、全て同じGaussian analytic実装とみなしません。
それぞれの計算は[Active Learning Guide](acquisition/09_active_learning.md)と
[水準集合Guide](acquisition/10_level_set.md)へ進めます。

## 6. 最適化への接続と参照先

[optimizer_compatibility.py][compatibility]の検査は、registryのq上限や
one-shotの有無を見て、指定した最適化方式との不整合を返します。
通常のmixed最適化では、拡張batchの行ごとに異なるカテゴリ割当を保持できないため、
one-shotを別に扱います。このmetadata検査は、数値探索そのものではありません。

標準獲得関数を評価する上流クラス、意味を接続するcomposition、
独自スコアの`forward()`、候補を探索するoptimizerを分けることで、
Theoryの期待効用がどこで計算されるかを追えます。

使用と可視化は[標準GP Notebook][notebook]、
出力所有関係やbridgeの境界条件は[compositionのテスト][composition-test]、
標準BOの数式との対応は[解析参照テスト][analytic-test]を参照できます。
上流コードへの参照はBoTorch 0.18.1を対象とします。

[b-acquisition]: https://github.com/pytorch/botorch/tree/v0.18.1/botorch/acquisition
[composition]: ../../src/robotorchan/acquisition/composition.py
[registry]: ../../src/robotorchan/acquisition/registry.py
[objectives]: ../../src/robotorchan/semantics/objectives.py
[direction]: ../../src/robotorchan/semantics/direction.py
[variance]: ../../src/robotorchan/acquisition/active_learning/variance.py
[straddle]: ../../src/robotorchan/acquisition/active_learning/straddle.py
[compatibility]: ../../src/robotorchan/acquisition/optimizer_compatibility.py
[notebook]: ../../examples/notebooks/01_single_task_gp.ipynb
[composition-test]: ../../tests/acquisition/test_composition.py
[analytic-test]: ../../tests/acquisition/test_phase8_analytic_bo_gaussian_reference.py
