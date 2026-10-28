# Bayesian Optimization：意思決定を担う実装の境界

## この章で学ぶこと

[BOのTheory](../theory/01_bayesian_optimization.md)の1.3・1.5節では、
観測から予測分布を更新し、期待効用を最大化して次の実験を選ぶ流れを説明しました。
本章では、その流れをモデル、獲得関数、候補最適化の実装へ対応付けます。
各部品がどの情報を受け取り、どこまで処理するかを読むための入口です。

高価な実験対象と、モデルから計算する獲得関数は別の関数です。
候補探索の中で獲得関数を何度評価しても、実験の観測値がその回数だけ増えるわけではありません。
この違いが、学習・予測・数値探索を別の部品として接続する理由です。

## 1. Theoryの構成要素と実装

| Theory上の構成要素 | 主な実装の追跡先 |
| --- | --- |
| 観測データと代理モデル | `SingleTaskGP`などのモデルクラス |
| 観測に条件付けた分布 | 継承したBoTorchの`posterior()`とGPyTorchの条件付け |
| パラメータの推定 | `make_mll()`による目的の構成とBoTorchのfitting |
| 予測から期待効用への変換 | BoTorchの獲得関数、またはrobotorchanの専用獲得関数 |
| 効用を大きくする候補探索 | `optim.dispatch.optimize_acqf()`と選択したbackend |
| 実験・観測の追加 | これらの部品を接続する呼び出し側の処理 |

標準GPの入口は[models/standard/single_task.py][single-task]、
学習とデータ保持の共通部は[models/base.py][base]です。
`SingleTaskGP`は、BoTorchの確率モデルへ共通mixinを加えています。
Gaussianの条件付けを新しく実装するのではなく、モデルに元データの保持と学習入口を追加します。
詳しい継承関係と線形代数の経路は[GP Guide](02_gaussian_process.md)に集約しています。

## 2. モデルの構成・学習・条件付けを分ける

コンストラクタは観測と確率モデルを結び付けますが、パラメータ推定は別の処理です。
`ExactGPModelMixin.make_mll()`は、モデルとlikelihoodを持つ
`ExactMarginalLogLikelihood`を構成します。
`FittableExactGPModelMixin.fit()`は、その目的をBoTorchの
`fit_gpytorch_mll()`へ渡します。

一方、`posterior()`は現在のモデル状態に基づく予測を返します。
モデルを包むクラス、学習目的を包むクラス、予測分布を包むクラスは同一ではありません。
また、元データを保持するraw bufferは来歴の保存先であり、
追加観測に条件付けた後の最新の推論状態とは区別します。

これらは、Theoryの「モデルを更新する」をコード上で分解したものです。
ハイパーパラメータの再推定と、固定したパラメータのもとで観測へ条件付ける処理を
混同しないことが、更新の意味を追ううえで重要です。

## 3. 予測を意思決定スコアへ接続する

BoTorchの標準獲得関数は、モデルを保持し、候補入力から`posterior()`を呼んで
平均・分散や標本を取得します。モデル側が獲得関数の種類を選ぶ構造ではありません。
AnalyticのEI・UCBと、joint posteriorから標本を取るMC獲得関数では、
予測分布に要求する情報も異なります。

異種の出力と目的・制約を接続する補助層が
[acquisition/composition.py][composition]です。
`resolve_acquisition_composition()`は、目的と制約が参照する出力について、
それを所有するモデルと、そのモデル内部での出力番号を解決します。
この計画を作るだけでは、モデル間に共通の同時事後分布は生成されません。

例えば`make_qei_acquisition()`は、回帰の目的をBoTorchのMC objectiveへ接続し、
上流の`qExpectedImprovement`を構成します。
標準獲得関数の数式計算をrobotorchanが複製しているわけではありません。
目的の向き、出力の所有関係、標本の処理は[Acquisition Guide](04_acquisition_function.md)で追います。

## 4. 候補最適化の入口が担当すること

[optim/dispatch.py][dispatch]の`optimize_acqf()`は、
獲得関数と探索領域を受け取り、数値最適化のbackendを選びます。
最初に境界、候補数、候補制約、変数空間と要求された機能の整合を検査します。
この検査は、効用を定義する処理とも、実際に探索する処理とも別です。

既定のBoTorch経路では、[optim/backends/botorch.py][backend]の
`optimize_acqf_botorch()`へ進みます。
ここで共通の候補制約を上流の不等式・等式・非線形制約へ渡し、
数値探索そのものはBoTorchの`optimize_acqf()`へ委譲します。
非線形制約では、可行な初期解または初期化器が必要になるため、接続前の検査も行います。

返される候補と獲得関数値は、実験対象を評価した結果ではありません。
このdispatcherは、実験の実行、観測の追加、代理モデルの再学習を行いません。
したがって、BOの反復を理解するときは次の境界を保ちます。

```text
観測を保持するモデル
  → 獲得関数が候補の予測分布を取得
  → 候補最適化が獲得関数を繰り返し評価
  → 候補を呼び出し側へ返す
  → 外部の実験・シミュレーション
  → 新しい観測をモデル更新へ接続
```

## 5. バッチ・混合変数・状態を持つ探索

候補数qは、同時に選ぶ実験点の数です。
数値探索の多初期値や、MC標本数、モデルのbatch次元とは異なります。
複数候補間の依存を使う効用では、q点を一つの意思決定として評価します。

未観測のpending点と、将来観測を仮定するfantasyも実観測とは異なります。
これらの扱いは[バッチ・先読みGuide](optimization/08_batch_one_shot.md)で説明します。
TuRBOなどの探索状態もモデルのパラメータとは別であり、
[信頼領域Guide](optimization/10_trust_region.md)が担当します。

混合変数では、予測モデルのカテゴリ共分散と、候補探索の離散演算を分けます。
SAASや加法モデルなどの高次元モデル、多目的のEHVI・NEHVIも、
この接続構造の中で予測や効用の構成を変えるものです。
個別の部品の性質を、BO全体を自動実行する機能と同一視しません。

## 6. 実装の担当範囲と関連文書

robotorchanはモデルの共通機能を拡張し、目的・制約の接続とbackend選択を実装します。
標準GPの推論、標準BO獲得関数、BoTorch backendの数値探索は上流への委譲です。
他のbackendを選ぶ場合は、その実装で制約や離散変数をどう処理するかを別に読みます。

次は[Kernel Guide](03_kernel.md)で共分散部品、
[Acquisition Guide](04_acquisition_function.md)で効用評価の内部を確認できます。
実験を含む反復と可視化は[標準GP Notebook][notebook]、
実装上の境界条件は[学習契約テスト][fit-test]・[dispatcherのテスト][dispatch-test]を参照してください。

[single-task]: ../../src/robotorchan/models/standard/single_task.py
[base]: ../../src/robotorchan/models/base.py
[composition]: ../../src/robotorchan/acquisition/composition.py
[dispatch]: ../../src/robotorchan/optim/dispatch.py
[backend]: ../../src/robotorchan/optim/backends/botorch.py
[notebook]: ../../examples/notebooks/01_single_task_gp.ipynb
[fit-test]: ../../tests/models/contracts/test_exact_gp_fit.py
[dispatch-test]: ../../tests/optim/contracts/test_optimizer_dispatch.py
