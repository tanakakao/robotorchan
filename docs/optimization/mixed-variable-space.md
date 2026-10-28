# 混合変数の探索空間

`MixedVariableSpace` は公開された元の候補座標で変数の意味を定義します。surrogate のカテゴリカーネル、候補制約、数値最適化 backend とは別の層です。

各次元の型は一つだけです。

- 構造化 metadata にない次元は連続値。
- `integer_dims` は `ceil(lower) .. floor(upper)` の全整数。
- `categorical_values` は明示的な有限・順序なし集合。

型の重複、範囲外の次元、合法整数のない bounds、空・重複カテゴリ、非有限値、bounds 外のカテゴリを拒否します。

```python
space = MixedVariableSpace(
    bounds,
    integer_dims=(1,),
    categorical_values={2: (0.0, 1.0, 2.0)},
)
```


## q-batch の意味

metadata は平坦化した q-batch 座標ではなく公開特徴次元ごとに定義します。q > 1 では各候補行に同じ変数の意味を適用します。点間の関係は `CandidateConstraints` が担います。

## Fixed features

`validate_fixed_features` は同じ元座標で固定値を検証します。整数固定値は整数、カテゴリ固定値は宣言集合の要素である必要があり、不正な固定値を backend が暗黙に修復することを防ぎます。

## 責務の境界

この契約は合法な値を定め、探索手法は定めません。カテゴリ変異や構造化次元の扱いは backend の capability と runtime 検証に従います。対応状況は [DE](differential-evolution.md) など各 backend の説明を参照してください。

BoTorch の `fixed_features_list` を置き換えません。`MixedSpaceStrategy` は BoTorch による列挙経路です。共通の変数空間から backend 固有表現を導出・検証する場合も、BoTorch の制約の意味を変えません。
