# Contamination-mixture GP

大部分の観測を狭い Gaussian 成分、少数の汚染観測を広い Gaussian 成分で表現します。Student-t の全体的な heavy tail、疎な外れ値補正、入力依存ノイズとは異なる観測機構です。

## 利用方法と制約

`ContaminatedSingleTaskGP` は `contamination_probability`、`inlier_scale`、`outlier_scale`、`num_inducing` を受け取ります。確率は 0 と 1 の間、scale は正で `outlier_scale > inlier_scale` が必要です。`training_loss()` による変分学習を使い、`posterior()` は潜在応答を返します。`contamination_diagnostic(X, Y)` は潜在応答の平均を使った診断近似であり、汚染所属を完全に積分した事後確率ではありません。

## 理論・実装・検証を辿る

[理論](../theory/14_robust_gaussian_process.md)と[実装ガイド](../guide/14_robust_gaussian_process.md)で仮定・推論方法・近似を確認できます。[ソース](../../src/robotorchan/models/robust/contaminated.py)から API と処理を確認してください。

過去の設計条件・数式・検証要件は[設計記録](../../.github/project-records/model-design/contaminated_gp.md)に原文を保存しています。そこでの将来計画や Phase の完了条件は現在の対応保証ではありません。現在のモデル選択は[モデル索引](README.md)を参照してください。

### 学習の安定化

既定の `training_loss()` は一次元 Gaussian marginal の Gauss–Hermite 求積（最低32点）で期待対数混合尤度を近似します。`num_likelihood_samples` は求積点数の下限を制御し、`training_loss(num_likelihood_samples=...)` を明示すると従来のMonte Carlo評価を利用できます。`fit(steps=400, lr=0.02, warmup_steps=100)` は通常成分の標準偏差を一時的に広げ、指定した標準偏差に戻すオプションです。既定の `warmup_steps=0` では温度調整を行いません。収束や精度はデータごとに検証してください。
