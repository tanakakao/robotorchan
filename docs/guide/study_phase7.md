# Study API: Mixed / q-batch / CandidateConstraints

このガイドでは、Studyを使って複数候補をまとめて提案し、実験結果を後から登録する手順を説明します。

## 1. 初期候補をまとめて提案する

```python
from robotorchan.study import create_study

study = create_study(
    search_space={"temperature": (100.0, 300.0), "pressure": (1.0, 10.0)},
    seed=42,
)
trials = study.ask_batch(3)
for trial in trials:
    result = -(trial.params["temperature"] - 200.0) ** 2
    study.tell(trial, result)
```

`ask_batch(q)` は q 個のTrialをRUNNING状態で登録します。評価順序と `tell()` の順序は一致する必要はありません。

## 2. 整数・カテゴリ変数を指定する

`MixedVariableSpace` は公開候補座標に対する変数種別を定義します。カテゴリは数値コードで表現します。

```python
import torch
from robotorchan.optim.domains.variable_space import MixedVariableSpace
from robotorchan.study import create_study

space = MixedVariableSpace(
    bounds=torch.tensor([[0.0, 1.0], [1.0, 4.0]], dtype=torch.double),
    integer_dims=(1,),
)
study = create_study(
    search_space={"x": (0.0, 1.0), "count": (1.0, 4.0)},
    variable_space=space,
)
trials = study.ask_batch(2)
```

## 3. 入力制約を指定する

`CandidateConstraints` は入力候補の制約です。獲得関数が扱う出力制約とは異なります。
BoTorch形式の線形不等式は `sum(coefficients * X[indices]) >= rhs` を意味します。

```python
import torch
from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.study import create_study

constraints = CandidateConstraints(
    inequality_constraints=(
        (torch.tensor([0]), torch.tensor([1.0]), 0.8),
    ),
)
study = create_study(
    search_space={"x": (0.0, 1.0)},
    constraints=constraints,
    seed=1,
)
trial = study.ask()
assert trial.params["x"] >= 0.8 - 1e-7
study.tell(trial, trial.params["x"])
```

初期候補は最大256回のSobolサンプリングを行い、制約を満たす候補群を採用します。等式制約など可行領域が非常に狭い場合、実行可能な解が存在しても候補生成に失敗する場合があります。

## 4. q候補間の制約

`indices` を `[n_terms, 2]` とすると、1列目がq内の候補番号、2列目が変数番号です。
例えば `x_0 + x_1 >= 1.5` を表現できます。候補間制約は候補群全体で評価されるため、単独の `add_trial()` ではなく `ask_batch()` を使用してください。

## 制限事項

- `Study.optimize()` は逐次実行（q=1）です。
- 初期探索は制約付きSobol棄却サンプリングです。一般の等式制約向け専用ソルバーではありません。
- モデルベースのMixed/q-batchでは、指定したモデル・獲得関数・Optimizerの対応能力に依存します。
- 制約付き候補は最適化器から返った後にも検証します。
