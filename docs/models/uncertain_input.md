# Input uncertainty

Input uncertainty は「出力観測にノイズがある」問題とは異なり、学習点の座標やカテゴリ自体が不確かという問題です。

## Continuous uncertainty

`UncertainInputSingleTaskGP` は連続 training input の Gaussian uncertainty を covariance に反映します。

`MixedUncertainInputSingleTaskGP` は continuous uncertainty と nominal categorical input を分離します。カテゴリコードへ Gaussian jitter を加えるモデルではありません。

詳細: [Uncertain input GP](uncertain_input_gp.md)、[Mixed uncertain input GP](mixed_uncertain_input_gp.md)

## Categorical uncertainty

`UncertainCategoricalSingleTaskGP` はカテゴリを simplex 上の確率として表し、expected categorical covariance を利用します。候補生成時に確率 mixture を最適化することと、観測カテゴリの不確かさを表現することは同義ではありません。

詳細: [Uncertain categorical GP](uncertain_categorical_gp.md)

## 選択上の注意

input uncertainty、heteroskedastic output noise、robust risk/scenario evaluation は別レイヤーです。何が不確かなのかを明確にしてからモデルを選択してください。

Notebook: [Uncertain input](../../examples/notebooks/17_uncertain_input_gp.ipynb)  
Theory: [Uncertain-input GP](../theory/16_uncertain_input_gp.md)
