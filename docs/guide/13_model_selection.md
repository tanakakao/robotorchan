# Model Selection：問題記述から互換性判定へ

[Theoryのモデル選択](../theory/13_model_selection.md)は、観測形式、仮定、
予測対象、意思決定に必要な分布を区別します。この章では、その区別が`ProblemSpec`、model registry、互換性判定へどう表れるかを説明します。
利用者向けのモデル比較は[モデル選択資料](../getting_started/model_selection.md)を参照してください。

## 理論上の条件を表すオブジェクト

[problem.py](../../src/robotorchan/workflow/problem.py)の`ProblemSpec`は、
入力型、task構造、出力型、fidelity、頑健性、preferenceなどの要件を保持します。
[models/capabilities.py](../../src/robotorchan/models/capabilities.py)の
`ModelRegistryEntry`はモデルの実装情報とcapabilityを結び付けます。
[registry.py](../../src/robotorchan/models/registry.py)がその登録を管理します。

理論上の仮定すべてをmetadataへ圧縮しているわけではありません。例えば「高次元」という要件は、軸方向の疎性、加法性、低次元表現のどれがデータに正しいかを推定するものではありません。

## 条件を照合し、不適合理由を保持する

[selector.py](../../src/robotorchan/workflow/selector.py)の`_check_model()`は
`ProblemSpec`と各entryのcapabilityを比較し、理由を`reasons`へ蓄積します。入力型・task型・preference・multi-fidelityには一致を要求し、
multi-output・structured output・頑健性などには必要な能力の存在を要求します。

`ModelSelectionResult`はモデル名、互換性、理由を持つimmutableなdataclassです。`evaluate_models()`はregistryを順に走査して結果を返し、
`select_compatible_models()`が互換な名前だけを取り出します。順序はregistryの順序であり、性能によるランキングではありません。
モデルの構築・学習・データ依存の精度比較はこの処理に含まれません。

## モデルから意思決定までの条件を分離する

[recommendation.py](../../src/robotorchan/workflow/recommendation.py)の
`recommend_compatible_workflows()`は、モデル候補に対して獲得関数とoptimizerの互換性を照合します。
その先では[acquisition/compatibility.py][acq-compat]と
[optimizer_compatibility.py][opt-compat]が、それぞれの計算要件を扱います。

モデルがposteriorを持つだけでは、Gaussian marginal、joint sample、fantasization、入力勾配のすべてを提供することにはなりません。
例えば`ModelListGP`の独立出力合成とMultiTaskのtask共分散、分類のlatent posteriorと予測クラス確率では、獲得関数へ渡す量が異なります。この層の分離が、
Theoryの観測形式・予測対象・decision purposeの区別に対応します。

## capabilityの所有権と設計上の限界

`ProblemSpec`、registryからのfiltering、理由の集約はrobotorchan独自の接続処理です。BoTorchへモデル選択アルゴリズムを委譲する実装ではありません。一方、
選ばれたGPの推論や標準の獲得関数は、各章で説明する上流実装が担当します。

互換性判定は、実装が宣言する条件への静的な照合です。周辺尤度・交差検証などを実行してモデルを自動選択したり、組合せの較正・学習収束・最適化性能を推定したりはしません。
学習経路の差は[GP](02_gaussian_process.md)、[変分GP](09_variational_gp.md)、
[選好学習](10_preference_learning.md)、[分類](24_classification.md)へ分けて追えます。

[acq-compat]: ../../src/robotorchan/acquisition/compatibility.py
[opt-compat]: ../../src/robotorchan/acquisition/optimizer_compatibility.py
