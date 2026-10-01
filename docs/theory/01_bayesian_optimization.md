# 1. ベイズ最適化とは

ベイズ最適化（Bayesian Optimization; BO）は、**評価に時間や費用がかかる未知の関数を、
できるだけ少ない評価回数で最適化するための逐次的な意思決定方法**です。

この章では、具体例と直感を先に示し、その後で数式による定式化を行います。
数式を省略するのではなく、記号が何を意味するかを確認しながら進みます。

## まず具体例から考える

材料開発で、温度と圧力を変えながら材料強度を最大化する問題を考えます。

```text
入力 x = 温度、圧力
評価 f(x) = その条件で実験して得られる材料強度
```

1回の実験に数時間かかる、試料が高価である、装置を長時間占有する、といった場合、
多数の条件を無計画に試すことはできません。

そこで、これまでの実験結果から**「次にどの条件を試すと有益か」**を判断します。
BO は、現在もっとも良さそうな条件だけでなく、まだ十分に調べていない領域の価値も考えながら、
限られた評価回数を配分します。

## 1.1 ベイズ最適化が対象とする問題

ベイズ最適化（Bayesian Optimization; BO）は、**評価コストが高いブラックボックス関数を、できるだけ少ない評価回数で最適化するための逐次最適化手法**です。

典型的には、次の問題を考えます。

\[
x^* = \arg\max_{x \in \mathcal{X}} f(x)
\]

または最小化問題

\[
x^* = \arg\min_{x \in \mathcal{X}} f(x)
\]

です。

ここで、各記号は次を表します。

- \(x\): 温度、圧力、組成などの入力条件
- \(\mathcal{X}\): 入力として許される探索空間
- \(f(x)\): 入力 \(x\) に対する目的関数値
- \(x^*\): 目的関数を最大または最小にする入力

\(\arg\max\) は最大値そのものではなく、**最大値を与える入力がどこか**を表します。
例えば最大強度が 120 MPa なら、値 120 ではなく、その強度を与える温度・圧力が
\(x^*\) に対応します。

ここで `f(x)` は、解析式が分からない、勾配が利用できない、あるいは1回の評価に高いコストがかかる関数です。

例えば材料・製造分野では、次のような問題が該当します。

- 実験条件から性能を最大化する
- 組成とプロセス条件から特性を最大化する
- シミュレーション条件を最適化する
- 装置条件を調整して歩留まりを改善する
- 高価な試作回数を減らしながら最適条件を探索する

## 1.2 通常の最適化との違い

勾配法では、目的関数の勾配

\[
\nabla f(x)
\]

を利用して最適点へ移動します。

一方、ベイズ最適化では目的関数そのものではなく、観測データから作った**代理モデル（surrogate model）**を使って、次に評価すべき点を選びます。

BOでは、最適化対象の真の関数 `f(x)` は直接には分かりません。代わりに、これまでの観測

\[
\mathcal{D}_n = \{(x_i, y_i)\}_{i=1}^n
\]

を使って `f(x)` の分布を推定します。

\(\mathcal{D}_n\) は、入力 \(x_i\) と観測値 \(y_i\) の組を \(n\) 個集めたデータ集合です。
つまり「どの条件を試し、その結果がどうだったか」という履歴を表します。

観測値にはノイズが含まれる場合があり、一般には

\[
y_i = f(x_i) + \epsilon_i
\]

と書けます。

ここで \(f(x_i)\) は潜在的な真の関数値、\(\epsilon_i\) は観測ノイズです。
典型的な GP 回帰では

\[
\epsilon_i \sim \mathcal{N}(0, \sigma_{\mathrm{noise}}^2)
\]

のような Gaussian noise を仮定することがあります。ただし、これは BO 自体に必須の仮定ではなく、
heteroskedastic noise や heavy-tailed noise など異なる観測モデルも考えられます。

## 1.3 ベイズ最適化を構成する2つの要素

ベイズ最適化は大きく分けて次の2要素から構成されます。

### Surrogate model

未知の目的関数 `f(x)` を近似する確率モデルです。

典型的には Gaussian Process（GP）が使われます。

GPはある点 `x` に対して単一の予測値だけでなく、

- 予測平均
- 予測分散

を返します。

この「予測の不確実性」が、次の評価点を合理的に選ぶために重要です。

GP の場合、単一点については典型的に

\[
f(x) \mid \mathcal{D}_n
\sim
\mathcal{N}\left(\mu_n(x), \sigma_n^2(x)\right)
\]

と表せます。\(\mu_n(x)\) は予測平均、\(\sigma_n^2(x)\) は予測分散です。

厳密には GP が与えるのは各点の平均・分散だけではなく、複数入力に対する
**joint posterior distribution** です。この候補点間の相関は batch BO や
Monte Carlo acquisition で重要になります。

### Acquisition function

代理モデルの予測分布を使って、次に評価する候補点の価値を数値化する関数です。

\[
\alpha(x; \mathcal{D}_n)
\]

と書き、次の評価点を

\[
x_{n+1} = \arg\max_{x \in \mathcal{X}} \alpha(x; \mathcal{D}_n)
\]

として選びます。

ここで最大化しているのは高価な真の目的関数 \(f(x)\) ではなく、
surrogate model から安価に計算できる獲得関数 \(\alpha(x)\) であることが重要です。

## 1.4 探索と活用

BOの中心的な考え方は、**探索（exploration）と活用（exploitation）のバランス**です。

### 活用

予測平均が良い場所を優先します。

「今までの情報から見て、ここが最も良さそう」という領域を調べる考え方です。

### 探索

予測の不確実性が大きい場所を優先します。

「まだよく分かっていないので、調べる価値がある」という領域を調べます。

活用だけを行うと局所解に偏りやすく、探索だけを行うと評価回数を浪費します。

獲得関数は、この両者を1つのスコアとして扱います。

ただし、すべての獲得関数が探索項と活用項へ明示的に分解できるわけではありません。
Expected Improvement、Knowledge Gradient、entropy-based method などは、
それぞれ異なる意思決定原理から候補の価値を定義します。

## 1.5 ベイズ最適化の基本ループ

BOは以下の逐次ループで進みます。

1. 初期データを取得する
2. surrogate model を学習する
3. posterior distribution を構築する
4. acquisition function を計算する
5. acquisition function を最大化する
6. 得られた候補点で目的関数を評価する
7. 新しい観測をデータへ追加する
8. 停止条件を満たすまで繰り返す

数式で表すと、まず

\[
\mathcal{D}_n
\]

から posterior

\[
p(f \mid \mathcal{D}_n)
\]

を構築します。これを posterior（事後分布）と呼びます。

Bayes の定理との関係を概念的に書けば、

\[
p(f \mid \mathcal{D}_n)
\propto
p(\mathcal{D}_n \mid f)\,p(f)
\]

です。\(p(f)\) はデータを見る前の prior、
\(p(\mathcal{D}_n \mid f)\) は likelihood、
\(p(f \mid \mathcal{D}_n)\) はデータを見た後の posterior です。

そこから acquisition function

\[
\alpha(x; \mathcal{D}_n)
\]

を計算します。

その後、

\[
x_{n+1}=\arg\max_x \alpha(x; \mathcal{D}_n)
\]

を選び、実験またはシミュレーションで

\[
y_{n+1}=f(x_{n+1})+\epsilon
\]

を取得します。

そして

\[
\mathcal{D}_{n+1}=\mathcal{D}_n\cup\{(x_{n+1},y_{n+1})\}
\]

として更新します。

## 1.6 なぜ Gaussian Process がよく使われるのか

BOでは、単に予測精度が高いだけでは十分ではありません。

次点を選択するためには、**どこが分かっていて、どこが分かっていないか**をモデルが表現できる必要があります。

Gaussian Process は予測値を

\[
f(x_*) \mid \mathcal{D}
\]

という確率分布として扱えるため、各点で

- 平均 `\mu(x)`
- 分散 `\sigma^2(x)`

を得られます。

GP は有限個の入力点に対する関数値を joint Gaussian distribution としてモデル化するため、
観測データで条件付けした posterior から平均だけでなく covariance も得られます。

ただし「不確実性を出せる = 必ず正しい不確実性」という意味ではありません。
kernel、likelihood、prior、推論方法などの仮定がデータに適していなければ、
posterior uncertainty も不適切になり得ます。

この性質により、例えば UCB は

\[
\alpha_{\mathrm{UCB}}(x)
=
\mu(x)+\beta^{1/2}\sigma(x)
\]

のように、予測平均と不確実性を同時に利用できます。

## 1.7 ベイズ最適化が向いている問題

BOは特に次の条件で有効です。

- 1回の評価が高価
- 評価可能回数が限られている
- 勾配が得られない
- 目的関数がブラックボックス
- 評価結果を逐次的な意思決定に利用できる
- surrogate model を学習する価値がある程度の構造を目的関数が持つ

一方、1回の評価が非常に安く、何百万回でも実行可能な場合は、
BO より単純な探索法の方が適することがあります。

また、「何次元以上なら BO が使えない」という固定された境界はありません。
難しさは入力次元だけでなく、評価可能なデータ数、有効次元、変数間の構造、
kernel や prior、surrogate model、candidate-search strategy にも依存します。

## 1.8 高次元問題

標準的な GP ベース BO は、入力次元が増えるにつれて難しくなります。

高次元では、

- カーネル距離が識別しにくくなる
- 必要なデータ数が増える
- acquisition optimization が難しくなる

といった問題が発生します。

このため robotorchan では、SAAS GP や Additive GP などの高次元向けモデルも扱います。

## 1.9 Multi-objective Bayesian Optimization

目的が複数ある場合は、単一の最良値ではなく Pareto front を探索します。

例えば

\[
\mathbf{f}(x)
=
(f_1(x), f_2(x), \ldots, f_m(x))
\]

を同時に最適化します。

目的同士が競合する場合、ある目的を改善すると別の目的が悪化することがあります。そのため、単一の最適解ではなく非劣解集合を求めます。

BoTorch では EHVI / NEHVI 系の獲得関数が代表的です。

## 1.10 制約付きベイズ最適化

実務では、性能だけでなく制約を満たす必要があります。

例えば

\[
\max_x f(x)
\]

subject to

\[
g_j(x) \le 0
\]

という問題です。

材料・製造では、コスト、強度、安全性、組成和、装置上限などが制約として現れます。

制約には少なくとも、候補 \(x\) だけから判定できる既知の
**input / candidate constraint** と、評価して初めて結果が分かる
**outcome constraint** を区別する必要があります。

未知の outcome constraint では feasibility probability などを acquisition に組み込む方法があります。
既知の candidate constraint は candidate optimization 側で直接扱える場合があります。

## 1.11 Batch Bayesian Optimization

複数の実験を同時に実施できる場合は、1点ではなく `q` 点を同時に選びます。

\[
X_{\mathrm{next}}
=\{x_1,\ldots,x_q\}
\]

単純に「1点用 acquisition の上位 \(q\) 点」を選べばよいとは限りません。
候補同士の依存関係や joint utility を考慮する必要があります。

BoTorch の `qEI`, `qUCB`, `qNEHVI` などは batch BO を扱えます。

### 1.11.1 Asynchronous Bayesian Optimization

複数の評価が同時に終了するとは限りません。一部の実験が実行中でも次の候補を決めたい場合、
未完了候補を pending points として意思決定へ反映します。

これは複数点をまとめて選ぶ batch BO と関連しますが、同じ概念ではありません。

## 1.12 robotorchan との関係

robotorchan では、BO全体を以下のように分けて考えます。

```text
データ
  ↓
Surrogate model
  ↓
Posterior distribution
  ↓
Acquisition function
  ↓
Acquisition optimization
  ↓
Candidate
  ↓
実験・シミュレーション
  ↓
データ更新
```

`docs/models.md` はこのうち surrogate model の選択を中心に説明しています。

本理論ガイドでは、その背景となる GP・Kernel・Multi-task・Multi-Fidelity などを順に説明します。

## 1.12.1 実装では BO loop をさらに分解する

初心者が最初に覚える中心構造は、

```text
データ
 -> surrogate model
 -> uncertainty を含む予測
 -> acquisition function
 -> 次の候補
```

です。

理論上は surrogate と acquisition の2要素で説明できても、実装では次の責務を分けます。

```text
model
 -> posterior
 -> transform / objective
 -> sampling
 -> acquisition
 -> initialization
 -> acquisition optimization
 -> evaluation
 -> model update
```

batch / asynchronous BO では pending points と fantasization がこの loop に加わります。
TuRBO のような trust-region method では candidate optimization 側に stateful search geometry が加わります。

この分解は単なるソフトウェア設計ではありません。
posterior sample の依存構造、objective の適用位置、initial conditions、
pending-point semantics などが BO の意思決定結果を変え得るためです。

詳細な実装境界は [理論ガイドREADME](README.md) の
「BO 周辺基盤の理論と実装ガイド」を参照してください。

## 1.13 この章で覚えておくこと

BO の本質は、単に高精度な予測モデルを作ることではありません。
**限られた評価予算の中で、現在の情報を使って次の評価を逐次的に決めること**にあります。

```text
高価な未知関数 f(x)
        ↓ 少数の観測
データ D_n
        ↓
確率的 surrogate model
        ↓
posterior: 予測 + 不確実性
        ↓
acquisition function
        ↓
次に評価する候補
        ↓
新しい観測を追加して繰り返す
```

## 1.14 次に読む章

BOを理解するためには、surrogate model がどのように予測平均と不確実性を計算しているかを理解することが重要です。

次章では Gaussian Process の基本理論を説明します。

→ [2. Gaussian Process](02_gaussian_process.md)

## 参考文献

1. Mockus, J. (1989). *Bayesian Approach to Global Optimization*.
   Kluwer Academic Publishers.
2. Jones, D. R., Schonlau, M., and Welch, W. J. (1998).
   Efficient global optimization of expensive black-box functions.
   *Journal of Global Optimization*, 13, 455-492.
3. Snoek, J., Larochelle, H., and Adams, R. P. (2012).
   Practical Bayesian optimization of machine learning algorithms.
   *Advances in Neural Information Processing Systems 25*.
4. Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., and de Freitas, N. (2016).
   Taking the human out of the loop: A review of Bayesian optimization.
   *Proceedings of the IEEE*, 104(1), 148-175.
5. Balandat, M. et al. (2020).
   BoTorch: A framework for efficient Monte-Carlo Bayesian optimization.
   *Advances in Neural Information Processing Systems 33*.
