# 回帰目的と二値 Pass 制約

`strength_pass` は Strength を最大化する連続目的と二値 Pass/Fail 出力制約を組み合わせます。

- Strength：`1 - (x0-0.8)^2 - (x1-0.3)^2`。
- Pass margin：`g(X) = x1 - 0.25 - 0.5*x0`。
- label：`strength_pass_labels(X) = (g(X) >= 0)`。
- 既知の制約付き最適点：`X=(0.66,0.58)`、Strength `0.902`。

実行可能性指標には決定的な符号付き margin を保存します。分類学習には `strength_pass_labels` の 0/1 label を使い、latent GP sample や予測 Pass 確率と混同しないでください。

`register_regression_binary_problems(registry)` で登録し、seed 付き `run_benchmark` で実行します。`simple_regret_curve`・`cumulative_feasibility_rate` は真値評価に使えます。

fixture と評価テストの提供は、分類器学習や制約付き qEI・qNEI の合成を含みません。学習と確率的実行可能性は、この既知真値に対して別途比較してください。
