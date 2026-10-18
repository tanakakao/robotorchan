# Evolutionary and Population-based Optimization

## 7.1 共通構造

population-based optimizer は複数の candidate を保持し、
評価結果を次の population 更新へ利用します。

scalar acquisition optimization では acquisition function を joint [q, d] batch 上で最大化し、
population-based method では q-batch を [qd] の decision vector として
表現できます。

一方、NSGA-II は vector-valued objective の Pareto set を探索する手法です。

## 7.2 Differential Evolution

Differential Evolution (DE) は population 内の個体差分を mutation direction に使います。

代表的な mutation は

    v_i = x_r1 + F (x_r2 - x_r3)

です。

trial vector を crossover で作り、objective が改善すれば target vector と置き換えます。

acquisition maximization は符号を反転すれば minimization としても定式化できます。

### Integer variable

integer dimension では評価前に round と clip による repair が可能です。
unordered categorical variable には別途 type-aware な演算が必要です。

### Constraint

DE に対しては feasibility-first selection、penalty、native constraint handling など
複数の制約処理方式を組み合わせられます。

## 7.3 CMA-ES

Covariance Matrix Adaptation Evolution Strategy (CMA-ES) は
探索分布 N(m, sigma^2 C) の mean、global step size、covariance を
高評価 sample から適応させます。

coordinate-wise gradient を使わず、変数間の相関を covariance に学習できることが特徴です。

標準的な CMA-ES は連続空間の探索分布を前提とします。
離散変数を扱う場合は別途 encoding や離散化の設計が必要です。

candidate constraint は

    J(X) = -alpha(X) + lambda V(X)

という penalty objective で処理します。
したがって penalty coefficient は探索挙動に影響します。

## 7.4 Genetic Algorithm

Genetic Algorithm (GA) は selection、crossover、mutation、elitism を組み合わせます。

連続空間の GA では tournament selection、arithmetic crossover、
Gaussian mutation などの演算を利用できます。

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

type-aware operator は unordered categorical を含む mixed search に適しています。

## 7.6 Particle Swarm Optimization

Particle Swarm Optimization (PSO) は particle の位置と速度を

    v_i(t+1) = w v_i(t)
             + c1 r1 (p_i - x_i(t))
             + c2 r2 (g - x_i(t))

    x_i(t+1) = x_i(t) + v_i(t+1)

で更新します。

p_i は personal best、g は global best です。

box bounds に対する clipping や integer dimension の repair を組み込めます。
categorical variable には別途離散的な位置更新規則が必要です。

constraint は acquisition value から lambda × violation を引いた penalty score に反映されます。

## 7.7 NSGA-II

NSGA-II は multi-objective evolutionary algorithm です。
population を Pareto dominance に基づく non-dominated front へ分け、
同じ front 内では crowding distance により diversity を維持します。

複数目的の評価ベクトルに対して Pareto 優越関係を定義し、
非劣解集合の近似と多様性の維持を目指します。

### qEHVI / qNEHVI との違い

qEHVI や qNEHVI は multi-objective BO の acquisition ですが、
候補 X に対して最終的には scalar acquisition utility を返します。

したがって

    multi-objective black-box problem
        !=
    vector-valued acquisition optimization

です。

qEHVI / qNEHVI の candidate optimization に NSGA-II が必須なわけではありません。
NSGA-II は最適化対象そのものが vector-valued な場合の選択肢です。

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

population-based search では native feasibility、penalty、feasibility-first、
repair などを区別します。制約処理方式は探索アルゴリズム名だけで一意には決まりません。
詳細は [Candidate Constraint Optimization](05_constraints.md) を参照してください。

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
