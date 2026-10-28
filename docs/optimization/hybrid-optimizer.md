# 大域探索と局所探索を組み合わせる最適化

微分不要の大域探索と BoTorch の局所改良を組み合わせます。

```text
DE or CMA-ES
    -> promising joint q x d candidate
    -> BoTorch optimize_acqf with explicit batch_initial_conditions
    -> refined candidate
```


多峰的な獲得関数や初期値に敏感な勾配法に対して、大域段階で有望な候補を探します。局所段階では BoTorch の制約付き最適化を使います。

```python
from robotorchan.optim.backends import optimize_acqf_hybrid

candidates, value = optimize_acqf_hybrid(
    acq_function=acq,
    bounds=bounds,
    q=3,
    global_optimizer="de",
    global_options={"maxiter": 100},
    local_options={"maxiter": 200},
    constraints=constraints,
    seed=123,
)
```


大域段階には `de`、`cmaes`、`mixed_ga` または互換 custom callable を指定できます。候補制約は両段階へ渡し、backend 固有の処理を保ちます。[候補制約](candidate-constraints.md)を参照してください。

現在の局所段階は restart を 1 回だけ受け付けます。同じ大域候補の複製は初期値の多様性を増やさないため、`num_restarts > 1` は拒否します。異なる大域初期値からの restart は将来の拡張です。

`fixed_features` は両段階へ渡します。`variable_space` に整数・カテゴリを宣言する場合、BoTorch mixed 局所段階用の `mixed_fixed_features_list` も必要です。各設定は共通 `fixed_features` または個別設定で全構造化次元を固定してください。大域最適化器にも宣言した変数型への対応が必要で、この経路の `cmaes` は `variable_space` を受け付けません。未対応の組合せは明示的に拒否します。
