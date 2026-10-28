# 混合変数の Genetic Algorithm

BoTorch 型の数値 bounds と構造化次元を受け取る進化的最適化器です。

- 連続次元は実数値のまま扱う。
- `integer_dims` は初期化・変異後に丸め、合法領域へ収める。
- `categorical_values` はカテゴリの合法な符号集合を指定する。

カテゴリに順序距離を仮定しません。交叉は親のカテゴリを継承し、変異は宣言集合から直接抽出します。整数の交叉も親値を継承し、変異は離散的な 1 step 移動と領域修復を使います。

推奨の契約は sampling backend と共通の `MixedVariableSpace` です。低水準の `integer_dims`・`categorical_values` も使えますが、`variable_space` と併用できません。整数列とカテゴリ列は重複できません。

同契約を使う場合 `fixed_features` を検証し、joint q-batch 全体へ展開して各変異後に復元します。個体は `q x d` 全体を表し、joint 獲得関数の意味を保ちます。集団評価は Tensor device 上でバッチ化され、bounds と獲得関数が CUDA にあれば CUDA で実行します。

```python
from robotorchan.optim.backends import optimize_acqf_mixed_ga

candidates, value = optimize_acqf_mixed_ga(
    acq_function=acq,
    bounds=bounds,
    q=3,
    integer_dims=[1, 4],
    categorical_values={2: [0.0, 1.0, 2.0]},
    population_size=128,
    generations=100,
    seed=123,
)
```


探索空間戦略ではなく数値最適化 backend であり、上位の Mixed・tree search 戦略から再利用できます。

候補制約は scalar penalty でなく実行可能性優先です。実行不可能な個体間は総違反量で順位付けします。[候補制約](candidate-constraints.md)を参照してください。

## 混合変数に依存する制約

領域修復と fixed features 適用後の元座標で制約を評価します。連続・整数・カテゴリの組合せに依存する制約にも別 DSL は不要です。

`CandidateConstraints` の線形不等式・等式、非線形不等式に対応します。個体が q-batch 全体を表すため、点内・点間の意味を保ちます。カテゴリに応じて連続の許容領域が変わる制約でも、有限ラベルを補間せず合法値で評価します。実行可能性を獲得値より優先し、違反候補間は総違反量で比較します。
