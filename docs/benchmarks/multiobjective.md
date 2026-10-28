# 多目的 benchmark 問題

| 問題 | 変数数 | 目的数 | 方向 | Pareto 情報 |
| --- | ---: | ---: | --- | --- |
| Biobjective linear | 1 | 2 | 両方最大化 | 解析 front の標本 |
| Branin-Currin | 2 | 2 | 両方最大化 | reference point のみ |
| DTLZ2 | 7 | 3 | すべて最小化 | 解析 front の標本 |
| ZDT1 | 6 | 2 | 両方最小化 | 解析 front の標本 |

Branin-Currin は標準の negative-Branin・Currin、入力 [0, 1]^2、最大化 reference point `(-310, -1)` を使います。解析的な真の Pareto front は認定せず `reference_front` を持ちません。Currin の x2=0 では指数項の連続極限を使い NaN を避けます。

`register_multiobjective_problems(registry)` で登録します。比較では reference point を固定し、最小化の向きを一貫して変換してください。問題定義だけで多目的獲得戦略の品質は示せません。

## Front と方向

`biobjective_linear` は `(x, 1-x)`、`zdt1` は `(u, 1-sqrt(u))`、`dtlz2` は正の単位球面が解析 front です。保存された `reference_front` は有限標本で、連続 front の厳密表現ではありません。`branin_currin` には reference point のみあります。

reference point は元の目的方向で定義します。最大化の例では `(-0.1, -0.1)`、最小化では `(1.1, ...)` です。最大化用 hypervolume 実装へ渡す前に、目的値と reference point の両方を `problem.to_maximization(...)` で揃えてください。

```python
from robotorchan.benchmarks import (
    BenchmarkProblemRegistry,
    register_multiobjective_problems,
)

registry = BenchmarkProblemRegistry()
register_multiobjective_problems(registry)
problem = registry.create("zdt1")
print(problem.reference_front.shape)
```


これらの定義だけでは qEHVI・qNEHVI の接続の正しさや最適化性能を保証しません。
