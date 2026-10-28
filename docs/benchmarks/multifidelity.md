# 多忠実度 benchmark 問題

設計変数は `x = X[..., 0]`、連続 fidelity は `s = X[..., 1]`、ともに [0, 1] です。最高 fidelity は s=1、低費用近似は s<1 です。

| 問題 | 最高 fidelity の目的 | 低 fidelity の追加項 |
| --- | --- | --- |
| multifidelity_quadratic | (x - 0.7)^2 | (1-s)*(0.25 + 0.2*(x-0.2)^2) |
| multifidelity_oscillatory | (x-0.65)^2 + 0.05*sin(6*pi*x)^2 | (1-s)*(0.1 + 0.3*(x-0.1)^2) |

共通費用は `0.1 + 0.9*s^2` で、s=0 では 0.1、s=1 では 1.0 です。`BenchmarkTrajectory.cumulative_cost` で費用比較できます。

quadratic の大域最小値は `(x,s)=(0.7,1)` で 0 です。oscillatory は未認定の解析最適値を掲載せず `optimal_value` を未設定にします。

最後の座標は物理設計変数ではなく fidelity です。最高 fidelity だけの比較では s=1 に固定してください。汎用 Random は両座標を抽出する基準法で、fidelity-aware 獲得戦略ではありません。runner は評価数と費用を記録しますが金額予算を強制しません。

`register_multifidelity_problems(registry)` で登録します。
