# Preference learning

`PairwiseGP` は絶対的な回帰 target ではなく、「A が B より好ましい」という pairwise comparison を学習します。

官能評価、デザイン選好、ランキングなど、絶対スコアを安定して付けにくい問題に向きます。通常の `train_Y` を使う回帰モデルとは入力契約が異なり、`datapoints` と `comparisons` を使用します。

Preference learning は noisy regression の代替ではありません。観測された情報そのものが比較である場合に選択します。

Notebook: [Pairwise GP](../../examples/notebooks/07_pairwise_gp.ipynb)  
Theory: [Preference Learning](../theory/10_preference_learning.md)
