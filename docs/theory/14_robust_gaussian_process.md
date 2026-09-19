# Robust Gaussian Process

## 直感

標準 GP の Gaussian likelihood は、残差が概ね正規分布に従うときに扱いやすい一方、少数の
極端な観測や裾の重い誤差があると posterior が強く引っ張られます。robust GP では
「外れ値がある」という一語ではなく、外れ値が生じる統計的機構を区別します。

## Gaussian likelihood の弱点

標準的な回帰では

```text
y_i = f(x_i) + epsilon_i
epsilon_i ~ Normal(0, sigma^2)
```

を仮定します。二乗誤差に対応するため、大きな残差の影響が急速に増えます。

## Student-t likelihood

Student-t 分布は Gaussian より heavy tail を持ちます。

```text
y_i | f_i ~ StudentT(nu, f_i, sigma)
```

自由度 `nu` が小さいほど大きな残差を許容します。「観測の一部だけが別分布」というより、
残差分布全体を heavy-tailed と考える場合に自然です。

robotorchan: `StudentTSingleTaskGP`, `MixedStudentTSingleTaskGP`。

## Contamination model

contamination model は nominal observation と gross-error component の mixture を考えます。

```text
p(y_i | f_i) = (1 - pi) p_nominal(y_i | f_i)
             + pi p_contaminated(y_i | f_i)
```

少数の観測が別の高分散機構から生じるという解釈を持てます。

robotorchan: `ContaminatedSingleTaskGP`, `MixedContaminatedSingleTaskGP`。

## Relevance pursuit

relevance pursuit は少数の観測へ sparse な correction / relevance を割り当てる考え方です。
Student-t のように全観測の likelihood を同じ heavy-tailed family に変更する仮定とは異なります。

robotorchan: `RobustRelevancePursuitSingleTaskGP`,
`MixedRobustRelevancePursuitSingleTaskGP`。

## Mixed variables

Mixed 版でも robust mechanism 自体の理論は変わりません。違いは input covariance が continuous
distance だけでなく categorical identity を保持する点です。カテゴリコードを連続値として
jitter することはしません。

## BO との関係

robust surrogate は posterior mean だけでなく posterior uncertainty も変えるため、EI/UCB/NEI
などの acquisition に間接的に影響します。一方、CVaR や worst-case objective は
observation robustness ではなく decision objective 側の risk aggregation です。

## 使い分け

- 残差全体が heavy-tailed: Student-t
- nominal + gross-error mixture: Contaminated GP
- 少数点へ sparse correction を置きたい: Relevance Pursuit
- noise variance 自体が入力で変わる: [Heteroskedastic noise](15_heteroskedastic_noise.md)

実装ガイドは [Robust / Noise / Uncertainty model guide](../robust-model-support.md) を参照してください。
