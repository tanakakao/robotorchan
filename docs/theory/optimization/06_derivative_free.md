# Derivative-free Acquisition Optimization

[基礎章](01_foundations.md)と[候補制約章](05_constraints.md)を前提とします。
本章では、勾配を使わないことと、変数型や制約を考慮しないことを区別します。

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

joint q-batch では各サンプルが q 個の候補を含み、
その集合全体に対して acquisition value を評価します。

continuous dimension は bounds 内の一様乱数です。
integer dimension は legal integer set、categorical dimension は allowed category set から
直接一様に生成します。

したがって mixed space でも、category code 間に人工的な距離を仮定しません。

連続値の一様乱数を整数へ丸める操作は、整数集合からの一様抽出とは異なる場合があります。
例えば $[0,2]$ の一様乱数を最も近い整数へ丸めると、0・1・2の確率は $1/4,1/2,1/4$ です。
整数を等確率にしたければ、直接離散抽出するなど、意図した分布で生成します。

## 6.3 Sobol search

Sobol sequence は low-discrepancy sequence により、擬似乱数より均一な
space-filling を狙う quasi-Monte Carlo sampling です。

joint q-batch 全体を qd 次元の Sobol point として生成し、
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

非空の有限標本集合を $\mathcal S$、非空でコンパクトな可行領域を $\mathcal F$ とし、
全ての標本が可行で、効用がL-Lipschitz連続なら、
被覆半径 $h=\sup_{X\in\mathcal F}\min_{Z\in\mathcal S}\|X-Z\|$ に対して

$$
\max_{X\in\mathcal F}\alpha(X)-\max_{Z\in\mathcal S}\alpha(Z)\le Lh
$$

と評価できます。最適点に近い標本との効用差をLipschitz性で抑えるためです。
ただし未知のLや被覆半径を、有限個の評価だけで正確に把握できるとは限りません。
非滑らかな木モデルなどに、この連続性の上界をそのまま適用しません。

## 6.5 q-batch は joint decision variable

q > 1 の acquisition では1点ずつ独立に sampling するのではなく、
X = (x_1, ..., x_q) を1つの decision variable として評価します。

qEI や qNEHVI のような joint acquisition は candidate 間の相互作用を持つため、
各点の marginal acquisition を個別最大化することとは一般に一致しません。

この joint q-batch semantics は数値最適化手法にかかわらず維持する必要があります。

## 6.6 Fixed features

fixed feature は候補生成時に固定するか、生成後に対応座標を置き換えられます。

mixed variable の fixed feature は variable semantics と整合する必要があります。
integer dimension には整数、categorical dimension には allowed category を指定します。

## 6.7 Constraint handling

gradient-free optimizer でも candidate constraint は無視できません。

| Constraint handling | 原理 |
| --- | --- |
| Native | 探索演算に制約を組み込む |
| Feasibility-first | 実行可能性と違反量を優先して比較 |
| Penalty | 目的値から制約違反ペナルティを差し引く |
| Repair / projection | 評価前に実行可能領域へ写像 |
| Rejection | 実行不可能な候補を棄却 |

penalty method は acquisition value から lambda × violation を引いた score を最大化します。

feasibility-first は feasible candidate を infeasible candidate より優先し、
infeasible 同士では violation の小ささを優先します。

例えば独立な提案の可行確率がpなら、一つの可行点を得るまでの提案回数の期待値は $1/p$ です。
等式制約が低次元の面を作る場合など、pが零なら棄却を繰り返しても厳密な可行点は得られません。
詳細は [Candidate Constraint Optimization](05_constraints.md) を参照してください。

## 6.8 Structured repair

DE や PSO のような continuous representation を使う backend で integer variable を扱う場合、
round と clip により評価前に integer feasibility を回復します。

このとき optimizer が動く continuous coordinate と、
acquisition が実際に見る repaired coordinate は一致しません。
同じ整数値へ複数の proposal が写像されるため、plateau が生じます。

unordered categorical variable では単純な rounding は category semantics を保存しません。
unordered categorical variable には離散集合上の探索演算が必要です。

## 6.9 Tree posterior との関係

tree ensemble のように candidate coordinate に関する滑らかな gradient を
前提にしない posterior では gradient-free search が自然です。

tree surrogate に対しても random search や population search を利用できます。

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

## この章で覚えておくこと

勾配を使わない探索でも、候補の分布、変数型、制約の設計は必要です。
有限集合から最良点を選ぶ品質は、効用の規則性と集合の被覆に依存します。
棄却や丸めを追加したときは、実際に生成される候補の分布と可行性を確認します。

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
