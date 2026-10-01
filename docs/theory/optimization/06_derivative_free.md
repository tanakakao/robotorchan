# Derivative-free Acquisition Optimization

## 6.1 目的

獲得関数最適化では、候補点 X に対して

    X* = argmax alpha(X),  X in search-space^q

を解きます。

gradient-free backend は candidate gradient を必要とせず、
acquisition value の評価だけで探索します。
これは acquisition が非微分可能な場合だけでなく、integer / categorical variable、
tree posterior、repair を含む探索空間でも有用です。

ただし「gradient-free」は「構造を無視してよい」という意味ではありません。
candidate の型、制約、q-batch semantics は backend ごとに保存する必要があります。

## 6.2 Random search

Random search は有限個の候補を sampling し、その中で acquisition value が
最大の q-batch を選択します。

robotorchan の optimize_acqf_sampling(method="random") は [N, q, d] の q-batch を生成し、
各 batch に対して1つの scalar acquisition value を要求します。

continuous dimension は bounds 内の一様乱数です。
integer dimension は legal integer set、categorical dimension は allowed category set から
直接一様に生成します。

したがって mixed space でも、category code 間に人工的な距離を仮定しません。

## 6.3 Sobol search

Sobol sequence は low-discrepancy sequence により、擬似乱数より均一な
space-filling を狙う quasi-Monte Carlo sampling です。

robotorchan では q-batch 全体を qd 次元の Sobol point として生成し、
unit hypercube から bounds へ写像します。

integer / categorical dimension では Sobol coordinate を equal-width inverse-CDF bin に
写像します。これにより structured variable の legal value を保ちながら、
Sobol coordinate の層化を利用します。

## 6.4 Sampling optimizer の性質

Sampling optimizer の利点は次です。

- candidate gradient が不要
- local optimum に依存する restart geometry を持たない
- mixed variable を legal set から直接生成できる
- acquisition evaluation を batch 化しやすい
- deterministic seed による再現性を持たせやすい

一方、有限 sample なので global optimum を保証しません。
dimension と q が増えるほど qd 次元空間の被覆には多くの sample が必要です。

## 6.5 q-batch は joint decision variable

q > 1 の acquisition では1点ずつ独立に sampling するのではなく、
X = (x_1, ..., x_q) を1つの decision variable として評価します。

qEI や qNEHVI のような joint acquisition は candidate 間の相互作用を持つため、
各点の marginal acquisition を個別最大化することとは一般に一致しません。

robotorchan の sampling / DE / CMA-ES / GA / PSO は scalar acquisition optimization では
この joint q-batch semantics を維持します。

## 6.6 Fixed features

sampling backend は candidate 生成後に fixed feature を適用します。

mixed variable の fixed feature は variable semantics と整合する必要があります。
integer dimension には整数、categorical dimension には allowed category を指定します。

## 6.7 Constraint handling

gradient-free optimizer でも candidate constraint は無視できません。

| Backend family | 現在の主な constraint semantics |
| --- | --- |
| DE | native SciPy nonlinear constraint |
| GA / Mixed GA | feasibility-first |
| CMA-ES | penalty |
| PSO | penalty |
| Random / Sobol | candidate constraint API は未提供 |

penalty method は acquisition value から lambda × violation を引いた score を最大化します。

feasibility-first は feasible candidate を infeasible candidate より優先し、
infeasible 同士では violation の小ささを優先します。

詳細は [Candidate Constraint Optimization](05_constraints.md) を参照してください。

## 6.8 Structured repair

DE や PSO のような continuous representation を使う backend で integer variable を扱う場合、
round と clip により評価前に integer feasibility を回復します。

このとき optimizer が動く continuous coordinate と、
acquisition が実際に見る repaired coordinate は一致しません。
同じ整数値へ複数の proposal が写像されるため、plateau が生じます。

unordered categorical variable では単純な rounding は category semantics を保存しません。
現在の DE / PSO は categorical variable を拒否し、Mixed GA または sampling を使います。

## 6.9 Tree posterior との関係

tree ensemble のように candidate coordinate に関する滑らかな gradient を
前提にしない posterior では gradient-free search が自然です。

TreeEnsembleSearchStrategy はこの責務に位置付けられます。

ただし derivative-free optimization は tree model 専用ではありません。
GP acquisition でも非滑らかな objective transform、structured variable、
constraint handling などを理由に選択できます。

## 6.10 Population method との違い

Random / Sobol search は生成した candidate set を基本的に更新しません。

DE、CMA-ES、GA、PSO は評価結果を次世代の proposal distribution や population 更新へ
利用します。したがって同じ evaluation budget でも adaptive search です。

詳細は [Evolutionary and Population-based Optimization](07_evolutionary.md) を参照してください。

## 6.11 選択の考え方

sampling は robust baseline として有用です。
population method は acquisition landscape を反復的に探索できますが、
hyperparameter と stopping condition が増えます。

continuous で gradient が信頼できる場合は gradient-based optimizer が第一候補になり得ます。
mixed / non-smooth / tree-based では derivative-free backend の価値が高くなります。

最終的な選択では、gradient availability だけでなく

    variable type
     -> q-batch semantics
     -> candidate constraints
     -> acquisition output shape
     -> evaluation budget
     -> backend capability

を確認します。

## 参考文献

1. Sobol, I. M. (1967).
   *On the Distribution of Points in a Cube and the Approximate Evaluation of Integrals*.
   USSR Computational Mathematics and Mathematical Physics, 7(4), 86-112.
2. Niederreiter, H. (1992).
   *Random Number Generation and Quasi-Monte Carlo Methods*. SIAM.
3. Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B.,
   Wilson, A. G., & Bakshy, E. (2020).
   *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*. NeurIPS 33.
4. BoTorch documentation. *Optimization* and acquisition-function optimization.
