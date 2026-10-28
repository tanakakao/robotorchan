# 複数の学習対象となる出力制約

Yield を最大化する 2 問題です。

- `yield_two_binary_constraints`：二値 Pass/Fail 条件 2 本。
- `yield_binary_continuous_constraints`：二値 Pass/Fail と連続温度制限。

目的は `1-(x0-0.8)^2-(x1-0.2)^2`、x0・x1 は [0,1]、制約真値は `g(X) >= 0` です。

| 制約 | 真値 margin | 意味 |
| --- | --- | --- |
| Pass | `x1 - 0.2 - 0.4*x0` | 二値実行可能性 |
| Reliability | `0.4 - x0` | 第 2 の二値実行可能性 |
| Temperature | `0.8 - x1` | 連続の上限 |

第 1 問題は Pass・Reliability を使い、最適点 `(0.4, 0.36)`、Yield `0.8144` です。第 2 は Pass・Temperature を使い、最適点 `(0.8/1.16, 0.2+0.4*(0.8/1.16))`、Yield 約 `0.911724` です。

`binary_constraint_labels(X)` は第 1 問題の 2 列の 0/1 label を返します。混合問題では最初の列で Pass 分類器を、Temperature margin で回帰モデルを学習します。

符号付き margin は真値の実行可能性評価用で、分類訓練 label ではありません。fixtures・registry・テストの提供だけでは、surrogate 学習や joint 予測実行可能性を検証したことにはなりません。
