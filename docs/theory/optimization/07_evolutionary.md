# Evolutionary and Population-based Optimization

[微分不要の探索章](06_derivative_free.md)と[候補制約章](05_constraints.md)を前提とします。
集団サイズとBOの候補数qは別です。前者は数値探索で比較する解の数、後者は一度に評価する実験候補の数です。

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

ここで差分の係数 $F>0$ を定め、通常は互いに異なる個体の添字 $r_1,r_2,r_3$ を選びます。
例えば基準個体が $(0.2,0.4)$、差分を取る二個体が $(0.7,0.2),(0.1,0.5)$、$F=0.5$ なら、
変異ベクトルは $(0.5,0.25)$ です。これはまだ境界や制約を満たすとは限りません。

交叉の一例では、座標ごとに確率CRで変異ベクトルを採用し、それ以外は元の個体を使います。
$0\le CR\le1$ とし、少なくとも一座標を変異側から取る規則を加えると、交叉が元の個体だけを選ぶことを防げます。
ただし両者の座標値が一致する場合など、結果が同じになる可能性は残ります。
同じ決定論的な効用で改善した試行だけを受理すれば、個体の評価値は減りませんが、集団の大域収束までは保証しません。
毎回異なるモンテカルロ近似値を使う場合は、この単調性も真の期待効用については成立するとは限りません。

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

候補制約を扱う一つの構成は、

    J(X) = -alpha(X) + lambda V(X)

というペナルティ目的を最小化する方法です。
したがってペナルティ係数は探索挙動に影響します。
集団サイズを $\lambda$、平均の更新に使う上位数を $1\le\mu\le\lambda$ とします。
探索分布の平均は、例えば上位の標本を $x_{i:\lambda}$、正の重みを $w_i$ として
$m_{new}=\sum_{i=1}^{\mu}w_i x_{i:\lambda}$、$\sum_iw_i=1$ と更新します。
共分散や歩幅の更新には順位と探索履歴も使います。単に全標本の共分散を計算する操作とは異なります。
この探索共分散は提案の形を定める量であり、GPの関数値の事後共分散ではありません。

## 7.4 Genetic Algorithm

Genetic Algorithm (GA) は selection、crossover、mutation、elitism を組み合わせます。

連続空間の GA では tournament selection、arithmetic crossover、
Gaussian mutation などの演算を利用できます。

制約がある場合には、例えばペナルティの加算とは別に、実行可能性を優先する順位付けを選べます。

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

ここでp_iは個体の過去の最良位置、gは群れの最良位置です。wは慣性、c1・c2は引力の係数、
r1・r2は通常、座標ごとに生成する一様乱数です。最良位置と乱数の役割を分けます。

box bounds に対する clipping や integer dimension の repair を組み込めます。
categorical variable には別途離散的な位置更新規則が必要です。

制約を、獲得効用から違反量のペナルティを引いたスコアへ反映する設計も可能です。

## 7.7 NSGA-II

NSGA-II は multi-objective evolutionary algorithm です。
population を Pareto dominance に基づく non-dominated front へ分け、
同じ front 内では crowding distance により diversity を維持します。

複数目的の評価ベクトルに対して Pareto 優越関係を定義し、
非劣解集合の近似と多様性の維持を目指します。
混雑距離では、各目的ごとに並べた近隣の値の差を、その目的の集団内範囲で割って足します。
両端の点を優先する規則を用い、範囲が零の目的は零除算しないよう別に扱います。
この距離は目的空間の多様性の指標であり、元の入力空間の距離そのものではありません。

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
| 単位に依存するペナルティを避けたい制約 | 実行可能性優先などの比較規則を組み込む |
| 真のvector objective | NSGA-II |

これは性能順位ではありません。
実際の選択は dimension、q、constraint、mixed structure、
acquisition evaluation cost、budget に依存します。

## この章で覚えておくこと

集団法は複数の探索解を更新する方法であり、BOの実験候補数を増やす操作ではありません。
差分・共分散・速度・遺伝的演算が表す探索の幾何を、予測モデルの不確実性と区別します。
制約処理と離散演算は別に定め、大域最適性を手法名だけから仮定しません。

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
