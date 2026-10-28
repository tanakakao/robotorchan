# 異種出力の多目的 benchmark

Strength・Conductivity を最大化する 2 つの連続回帰目的と二値出力制約を組み合わせます。

- `strength_conductivity_pass_tradeoff`：Pass/Fail 制約 1 本。
- `strength_conductivity_two_pass`：Pass/Fail 制約 2 本。

入力 x0・x1 は [0,1]、目的関数は次です。

```text
Strength     = 1 - (x0 - 0.8)^2 - 0.25*(x1 - 0.3)^2
Conductivity = 1 - (x0 - 0.2)^2 - 0.25*(x1 - 0.7)^2
```


真値の実行可能性は `g(X) >= 0` です。第 1 margin は `x1 - 0.25 - 0.5*x0`、第 2 は `0.8 - (x0-0.5)^2 - (x1-0.5)^2` です。

分類の 0/1 label は `multiobjective_pass_labels(X)`・`multiobjective_two_labels(X)` から取得します。符号付き margin で二値分類器を訓練しないでください。

runner は真値目的と margin を保存し、`hypervolume_curve` が reference point `(0,0)` に対する実行可能な支配領域を測ります。これは評価指標であり獲得関数ではありません。

問題・label・再現性テスト・実行可能 hypervolume の提供と、異種 surrogate の学習や制約付き qEHVI・qNEHVI の統合検証は別です。
