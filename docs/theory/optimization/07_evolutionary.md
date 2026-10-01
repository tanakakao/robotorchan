# Evolutionary and Population-based Optimization

## 7.1 共通構造

population-based optimizer は複数の candidate を保持し、
評価結果を次の population 更新へ利用します。

scalar acquisition optimization では acquisition function を joint [q, d] batch 上で最大化し、
robotorchan の DE、CMA-ES、GA、Mixed GA、PSO は q-batch を内部的に
[qd] の decision vector として扱います。

NSGA-II は例外で、scalar acquisition dispatcher ではなく、
vector-valued objective の Pareto set を探索する別 primitive です。

## 7.2 Differential Evolution

Differential Evolution (DE) は population 内の個体差分を mutation direction に使います。

代表的な mutation は

    v_i = x_r1 + F (x_r2 - x_r3)

です。

trial vector を crossover で作り、objective が改善すれば target vector と置き換えます。

robotorchan の optimize_acqf_de は SciPy Differential Evolution を利用します。
acquisition maximization は内部で負の acquisition value の minimization に変換されます。

### Integer variable

integer dimension は acquisition / constraint evaluation 前に round と clip で repair します。
unordered categorical variable は数値距離を仮定しないため明示的に拒否します。

### Constraint

candidate constraint は SciPy NonlinearConstraint へ violation evaluator を接続し、
violation が0の candidate を feasible として扱います。

このため現在の DE は単なる penalty backend ではなく、
optimizer-native constraint route を持ちます。

## 7.3 CMA-ES

Covariance Matrix Adaptation Evolution Strategy (CMA-ES) は
探索分布 N(m, sigma^2 C) の mean、global step size、covariance を
高評価 sample から適応させます。

coordinate-wise gradient を使わず、変数間の相関を covariance に学習できることが特徴です。

robotorchan の optimize_acqf_cmaes は optional cmaes dependency を利用し、
q-batch を flatten した continuous vector を探索します。

現在の backend は integer / categorical variable 用の native semantics を持ちません。

candidate constraint は

    J(X) = -alpha(X) + lambda V(X)

という penalty objective で処理します。
したがって penalty coefficient は探索挙動に影響します。

## 7.4 Genetic Algorithm

Genetic Algorithm (GA) は selection、crossover、mutation、elitism を組み合わせます。

robotorchan の continuous GA は real-valued population を使い、
tournament selection、arithmetic crossover、Gaussian mutation を行います。

constraint がある場合は scalar penalty だけに依存せず、feasibility-first ranking を使います。

概念的な ordering は次の通りです。

1. feasible candidate を infeasible candidate より優先
2. feasible 同士は acquisition value が高い方を優先
3. infeasible 同士は violation が小さい方を優先

これにより acquisition scale と constraint violation scale の単純な混合を避けます。

## 7.5 Mixed Genetic Algorithm

Mixed GA は continuous / integer / categorical variable ごとに operator を変えます。

continuous では arithmetic crossover と実数 mutation を利用できます。
integer は integer feasibility を維持する mutation / repair を行います。

categorical は親から legal category を継承するか、allowed set から再sampleします。
category code の差を metric として mutation magnitude に使いません。

この type-aware operator が、continuous GA に最後だけ rounding を追加する方法との
本質的な違いです。

robotorchan では unordered categorical を含む mixed search の主要 population backend です。

## 7.6 Particle Swarm Optimization

Particle Swarm Optimization (PSO) は particle の位置と速度を

    v_i(t+1) = w v_i(t)
             + c1 r1 (p_i - x_i(t))
             + c2 r2 (g - x_i(t))

    x_i(t+1) = x_i(t) + v_i(t+1)

で更新します。

p_i は personal best、g は global best です。

robotorchan では bounds 内へ clip し、integer dimension は評価前に repair します。
categorical variable は現在サポートしません。

constraint は acquisition value から lambda × violation を引いた penalty score に反映されます。

## 7.7 NSGA-II

NSGA-II は multi-objective evolutionary algorithm です。
population を Pareto dominance に基づく non-dominated front へ分け、
同じ front 内では crowding distance により diversity を維持します。

robotorchan の optimize_vector_nsga2 は [population, d] の candidate を受け取り、
[population, m]、m >= 2 の vector objective を返す contract です。
全 objective は maximize として扱います。

返り値は近似 Pareto set と対応する objective values です。

### qEHVI / qNEHVI との違い

qEHVI や qNEHVI は multi-objective BO の acquisition ですが、
候補 X に対して最終的には scalar acquisition utility を返します。

したがって

    multi-objective black-box problem
        !=
    vector-valued acquisition optimization

です。

qEHVI / qNEHVI の candidate optimization に NSGA-II が必須なわけではありません。
optimize_vector_nsga2 は最適化対象そのものが vector-valued な場合の別 primitive です。

## 7.8 q-batch flattening

DE、CMA-ES、GA、Mixed GA、PSO は joint q-batch を内部的に flatten できます。

    X in R^(q x d)  <->  z in R^(qd)

ただし flattening は数学的な q-batch semantics を失わせるものではありません。
acquisition evaluation 時には [q, d] へ戻し、batch 全体を同時に評価します。

mixed variable の dimension ownership も q 個すべてへ展開する必要があります。

## 7.9 Exploration と exploitation

population method 自体の exploration / exploitation と、
BO acquisition の exploration / exploitation は別レイヤーです。

例えば UCB の beta は surrogate uncertainty に基づく BO utility を変えます。
一方、GA mutation rate や CMA-ES covariance は、その acquisition landscape を
どのように探索するかを変えます。

したがって optimizer population diversity を増やすことは、
BO acquisition の epistemic exploration parameter を増やすことと同義ではありません。

## 7.10 Constraint semantics の比較

| Backend | Constraint semantics | Integer | Categorical |
| --- | --- | --- | --- |
| DE | native feasibility | repair | no |
| CMA-ES | penalty | no | no |
| GA | feasibility-first | no | no |
| Mixed GA | feasibility-first | yes | yes |
| PSO | penalty | repair | no |
| NSGA-II | vector APIではcandidate constraintなし | no | no |

同じ「constraint対応」でも enforcement semantics が異なります。
backend 選択では boolean capability だけでなく方式まで確認する必要があります。

## 7.11 Acquisition optimization での意味

population optimizer は expensive black-box objective を直接大量評価するためではなく、
通常は学習済み surrogate から計算される比較的安価な acquisition landscape を探索します。

そのため population size × generations が大きくても、
実験回数が同じだけ増えるわけではありません。

ただし MC acquisition や大規模 GP posterior では acquisition evaluation 自体が
高コストになり得るため、optimizer evaluation budget は無視できません。

## 7.12 Backend 選択

代表的な選択軸は次です。

| 状況 | 適合しやすい backend |
| --- | --- |
| robust baseline / space filling | Random / Sobol |
| continuous non-smooth | DE / CMA-ES / GA / PSO |
| integerを含む | DE / PSO / Mixed GA / sampling |
| unordered categoricalを含む | Mixed GA / sampling |
| scale依存penaltyを避けたいconstraint | DE / GA / Mixed GA |
| 真のvector objective | NSGA-II |

これは性能順位ではありません。
実際の選択は dimension、q、constraint、mixed structure、
acquisition evaluation cost、budget に依存します。

## 参考文献

1. Storn, R., & Price, K. (1997).
   *Differential Evolution - A Simple and Efficient Heuristic for Global Optimization
   over Continuous Spaces*. Journal of Global Optimization, 11, 341-359.
2. Hansen, N., & Ostermeier, A. (2001).
   *Completely Derandomized Self-Adaptation in Evolution Strategies*.
   Evolutionary Computation, 9(2), 159-195.
3. Holland, J. H. (1975).
   *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
4. Kennedy, J., & Eberhart, R. (1995).
   *Particle Swarm Optimization*. Proceedings of ICNN, 1942-1948.
5. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
   *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*.
   IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
6. Deb, K. (2000).
   *An Efficient Constraint Handling Method for Genetic Algorithms*.
   Computer Methods in Applied Mechanics and Engineering, 186, 311-338.
