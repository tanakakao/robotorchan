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

[selector.py](../../src/robotorchan/workflow/selector.py)の`model_requirement_reasons()`は
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


## 統一Factoryによる候補提案

Factoryは既存クラスを生成する入口であり、学習や候補提案を暗黙に実行しません。

```python
import torch
from robotorchan.models import create_model, list_models
from robotorchan.acquisition import create_acquisition, list_acquisitions
from robotorchan.optim import create_optimizer, list_optimizers

X = torch.linspace(0.1, 0.9, 6, dtype=torch.double).unsqueeze(-1)
Y = -((X - 0.6) ** 2)
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
model = create_model("SingleTaskGP", X, Y)
model.fit()
acq = create_acquisition("qLogExpectedImprovement", model=model, best_f=Y.max())
optimize = create_optimizer("sobol", raw_samples=64, seed=17)
candidates, values = optimize(acq, bounds, q=2)
```

文字列には公開Factoryの一覧にある名前を使います。従来のクラスを直接
importして構築する方法も引き続き利用できます。Optimizer Factoryは
呼び出し可能な最適化関数を返します。Search Strategy Factoryは
状態を保持し得る既存のStrategyインスタンスを返す点が異なります。

## 問題設定から構成を選ぶ

```python
from robotorchan.workflow import ProblemSpec, evaluate_components, recommend
from robotorchan.study import create_study

spec = ProblemSpec(purpose="bayesian_optimization", q=2)
inventory = evaluate_components(spec)
(selection,) = recommend(
    spec,
    model="SingleTaskGP",
    acquisition="qLogExpectedImprovement",
    optimizer="sobol",
)
print(selection.compatibility)
print(selection.required_arguments())
study = create_study(
    search_space={"x": (0.0, 1.0)},
    recommendation=selection,
    optimizer_kwargs={"raw_samples": 64, "seed": 17},
    seed=17,
)
study.optimize(lambda trial: -((trial.params["x"] - 0.6) ** 2), n_trials=3)
for trial in study.ask_batch(2):
    study.tell(trial, -((trial.params["x"] - 0.6) ** 2))
```

判定は compatible / conditional / incompatible / unknown の4種類です。
conditional は追加の引数やposterior等の条件が残ることを示します。
recommend は性能順位ではなく静的な適合候補を返します。
Studyは単目的のスカラー観測を対象とし、すべてのFactory構成を
そのままStudyへ渡せるわけではありません。

詳細は[Capability Workflow](../capability_workflow.md)を参照してください。
