# 最適化器の公開 API

BoTorch の獲得関数体系を保ち、任意の名前付き optimizer 選択を追加します。入口は `from robotorchan.optim import optimize_acqf` です。既定の `optimizer="botorch"` は BoTorch 標準 backend に委譲します。`botorch.optim.optimize_acqf` の直接利用も有効です。

## 名前付き backend

`botorch`、`torch_adam`、`torch_adamw`、`torch_sgd`、`random`、`sobol`、`de`、`cmaes`、`ga`、`pso`、`hybrid` を選べます。

backend 固有の制御は `optimizer_options`、BoTorch と意味が同じ共通引数は明示的な引数に置きます。未対応の組合せは引数を無視せず拒否します。

Mixed GA とベクトル値 NSGA-II は `robotorchan.optim.backends` の専用 API です。通常の scalar continuous 呼び出しと同一視しません。追加の runtime 依存が必要なのはこの集合では CMA-ES のみで、`cmaes` extra を導入してください。

## 構造化変数空間

通常の bounds、q、制約、fixed features と併せて `MixedVariableSpace` で整数・カテゴリを宣言できます。連続変数は暗黙です。

実行前に変数型と capability を照合し、未対応の座標を丸め・無視しません。固定値も整数・カテゴリ領域に対して検証します。

backend の直接利用と低水準引数も公開されています。`MixedVariableSpace` は共通 dispatch の契約であり、低水準引数を置き換えません。曖昧な二重宣言を拒否する backend に両方を渡すとエラーになります。

対応する Random・Sobol・DE・GA・PSO・Hybrid へ変数空間を渡します。勾配 Torch は整数・カテゴリ、PSO はカテゴリを拒否します。`optimizer="botorch"` は連続最適化です。BoTorch の mixed 列挙には `fixed_features_list` が明示される専用 mixed backend を使います。

## 候補制約

`robotorchan.optim` から公開しています。

```python
import torch

from robotorchan.optim import CandidateConstraints, optimize_acqf

constraints = CandidateConstraints(
    nonlinear_inequality_constraints=((lambda x: x.new_tensor(0.8) - x[0], True),),
)

candidate, value = optimize_acqf(
    acquisition,
    bounds,
    q=1,
    constraints=constraints,
    batch_initial_conditions=feasible_initial_conditions,
)
```


非線形 tuple は BoTorch 標準の `(callable, is_intrapoint)` です。実行可能性は `g(X) >= 0`、点内 callable の入力は `[d]`、点間は `[q, d]` です。

型 alias `LinearConstraint`、`NonlinearConstraint`、`NonlinearConstraintCallable` も公開しますが、標準 tuple を別 class で包む必要はありません。

出力制約は獲得関数固有の BoTorch 契約に従い、`CandidateConstraints` を経由しません。

## BoTorch 初期値 generator

既定経路は `batch_initial_conditions`、`ic_generator`、`ic_gen_kwargs` を公開します。特に非線形候補制約では実行可能な restart が必要なため、明示的な初期値またはそれを構成する generator を指定できます。

BoTorch の意味を保って同 backend だけへ渡します。他の名前付き最適化器は generator 引数を無視せず拒否します。

### 非線形の実行可能性エラー

明示的初期値の形状、dtype、device、有限性、box、非線形実行可能性を全 restart で検証し、不正なら最適化前に失敗します。

任意の非線形実行可能領域が非空であることは証明しません。generator が実行可能点を作れない場合や、妥当な初期化後に BoTorch・SciPy が失敗した場合は元のエラーを伝えます。無制約探索への暗黙の fallback はありません。
