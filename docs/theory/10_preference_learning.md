# 10. Preference Learning

## 10.1 この章で扱うこと

通常の回帰では、各入力 `x` に対して数値 target `y` を観測します。

一方、Preference Learning では絶対スコアではなく、

```text
A の方が B より好ましい
B の方が C より好ましい
```

という **比較結果** だけを観測します。

robotorchan では BoTorch の

```python
PairwiseGP
```

を thin wrapper として提供しています。

本章では、

- preference data
- latent utility
- pairwise comparison likelihood
- probit model
- GP prior
- Laplace approximation
- `PairwiseLaplaceMarginalLogLikelihood`
- posterior utility
- pairwise BO
- Expected Utility of Best Option (EUBO)
- BOPE との関係
- robotorchan API

を順に整理します。

---

## 10.2 なぜ絶対値ではなく比較を使うのか

人間は、絶対的な数値評価より比較の方が答えやすい場合があります。

例えば、

- 官能評価
- デザイン評価
- 材料の見た目・触感
- 複数性能を総合した専門家判断
- UI / UX の好み
- レコメンド
- ランキング

などです。

```text
このサンプルを 0〜100 点で評価してください
```

より、

```text
サンプル A と B ならどちらが良いですか？
```

の方が安定して回答できることがあります。

Preference Learning は、この比較結果から背後にある **潜在的な utility function** を推定します。

---

## 10.3 潜在 utility

各候補 `x` に、直接は観測できない utility

\[
f(x)
\]

が存在すると仮定します。

例えば

\[
f(x_A) > f(x_B)
\]

なら、理想的には A が B より好まれます。

ただし人間の判断や測定にはノイズがあるため、比較結果は必ずしも deterministic ではありません。

```text
latent utility
f(A) = 1.4
f(B) = 1.1

本来は A > B

しかし判断ノイズにより
B > A
になることもある
```

この uncertainty を likelihood で表現します。

---

## 10.4 Pairwise comparison data

BoTorch / robotorchan の `PairwiseGP` では、入力候補を

```python
datapoints
```

比較結果を

```python
comparisons
```

として与えます。

例えば

```python
datapoints = torch.tensor(
    [
        [0.1],
        [0.4],
        [0.8],
    ],
    dtype=torch.double,
)

comparisons = torch.tensor(
    [
        [2, 1],
        [1, 0],
    ],
    dtype=torch.long,
)
```

なら、

```text
2番目の候補 > 1番目の候補
1番目の候補 > 0番目の候補
```

という意味です。

各行は

```text
[winner, loser]
```

です。

---

## 10.5 `comparisons` は target value ではない

通常回帰では

```python
train_X
train_Y
```

を使います。

Preference Learning では

```python
datapoints
comparisons
```

を使います。

重要なのは、`comparisons` の値 `0, 1, 2, ...` は utility 値ではなく **datapoints の行index** だということです。

```text
comparisons[k] = [i, j]

意味:
候補 i が候補 j より好まれた
```

です。

---

## 10.6 比較確率

2つの候補 `x_i`, `x_j` の utility を

\[
f_i = f(x_i),\qquad f_j = f(x_j)
\]

とします。

Preference model では、

\[
P(i \succ j)
\]

つまり「i が j より好まれる確率」を utility difference

\[
f_i-f_j
\]

から定義します。

utility difference が大きく正なら、i が勝つ確率は高くなります。

```text
f_i - f_j >> 0
  → i が選ばれる確率 ≈ 1

f_i - f_j ≈ 0
  → ほぼ 50 / 50

f_i - f_j << 0
  → j が選ばれる確率 ≈ 1
```

---

## 10.7 Probit likelihood

BoTorch の `PairwiseGP` は default で **probit likelihood** を使います。

基本的な形は

\[
P(i \succ j)
=
\Phi\left(
\frac{f_i-f_j}{\sqrt{2}}
\right)
\]

です。

ここで

- \(\Phi\) は標準正規分布のCDF
- \(f_i-f_j\) は latent utility difference

です。

Chu & Ghahramani の定式化では denominator にノイズscale \(\sigma\) を含めますが、BoTorch はこの scale を明示的に likelihood parameter として分離せず、function scale を kernel の `ScaleKernel` で表現します。

---

## 10.8 Probit と Logistic

pairwise comparison model では logistic model もよく使われます。

例えば Bradley–Terry 型なら

\[
P(i \succ j)
=
\sigma(f_i-f_j)
\]

です。

ここで \(\sigma\) は sigmoid です。

一方 `PairwiseGP` の default は probit です。

```text
Bradley–Terry / logistic
  → sigmoid

PairwiseGP default
  → Gaussian CDF / probit
```

どちらも utility difference から preference probability を作る考え方は同じです。

---

## 10.9 GP prior on latent utility

utility function `f(x)` 自体には Gaussian Process prior を置きます。

\[
f \sim \mathcal{GP}(m(x),k(x,x'))
\]

つまり preference data を直接 GP に入れるのではなく、

```text
GP prior
  ↓
latent utility f(x)
  ↓
pairwise likelihood
  ↓
observed preference
```

という階層構造です。

---

## 10.10 なぜ通常の GP regression と違うのか

通常の Gaussian regression なら

\[
y=f(x)+\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,\sigma^2)
\]

であり、Gaussian prior と Gaussian likelihood の組み合わせなので posterior は解析的に Gaussian になります。

Preference Learning では likelihood が

\[
\Phi(f_i-f_j)
\]

のような nonlinear function になるため、posterior を exact Gaussian として解析的には求められません。

そこで `PairwiseGP` は **Laplace approximation** を使います。

---

## 10.11 Laplace approximation

観測 comparison data を \(D\) とすると、latent utilities `f` の posterior は

\[
p(f\mid D)
\propto
p(D\mid f)p(f)
\]

です。

しかし preference likelihood が non-Gaussian なので、この posterior は Gaussian ではありません。

Laplace approximation では、まず posterior mode

\[
f_{MAP}
=
\arg\max_f p(f\mid D)
\]

を求めます。

その周辺を2次近似して

\[
p(f\mid D)
\approx
\mathcal{N}(f_{MAP},\Sigma_{Laplace})
\]

とします。

---

## 10.12 Laplace approximation の直感

```text
本当の posterior
  → non-Gaussian

posterior mode を探す
        ↓
      f_MAP
        ↓
その周辺の曲率を Hessian で評価
        ↓
Gaussian で局所近似
```

です。

このため `PairwiseGP` は名前に GP とありますが、標準 `SingleTaskGP` と posterior fitting の仕組みが異なります。

---

## 10.13 MAP utility

`PairwiseGP` 内部では preference comparisons と GP prior の両方を考慮して latent utility の MAP 解を求めます。

概念的には

\[
f_{MAP}
=
\arg\max_f
\left[
\log p(D\mid f)
+
\log p(f)
\right]
\]

です。

第一項は比較結果への適合、第二項はGP priorによる正則化です。

---

## 10.14 Laplace covariance

posterior mode の周辺で negative log posterior の Hessian を使うと、近似 covariance は概念的に

\[
\Sigma_{Laplace}
\approx
\left(
K^{-1}+W
\right)^{-1}
\]

となります。

ここで

- \(K\) は GP prior covariance
- \(W\) は preference likelihood の曲率

です。

比較情報が多い領域では likelihood curvature が強くなり、utility uncertainty が縮みます。

---

## 10.15 PairwiseGP の default kernel

BoTorch の `PairwiseGP` は default で scaled RBF kernel を使います。

```text
RBF kernel
  → input similarity

ScaleKernel
  → latent utility scale
```

probit preference model では observation-noise scale と utility scale の識別が通常回帰ほど単純ではないため、BoTorch は kernel output scale を使う設計です。

---

## 10.16 utility の絶対値は重要ではない

pairwise data で観測されるのは utility difference です。

例えば、

\[
f(x)\rightarrow f(x)+c
\]

として全候補に同じ定数を足しても、

\[
(f_i+c)-(f_j+c)=f_i-f_j
\]

なので preference probability は変わりません。

したがって utility の **絶対的な基準点** は識別されません。

---

## 10.17 relative scale と評価指標

Preference Learning では通常の RMSE だけでなく、ranking quality を見るのが自然です。

例えば

- Kendall's tau
- Spearman rank correlation
- pairwise accuracy
- top-k recovery

などです。

BoTorch の preference tutorial でも、latent value の絶対値ではなく ranking correlation を評価に利用しています。

---

## 10.18 duplicate datapoints の consolidation

Preference dataset では同じ候補が複数回現れることがあります。

BoTorch の `PairwiseGP` は numerical stability のため、近接・重複 datapoints を内部で consolidation できます。

wrapper の引数には

```python
consolidate_rtol
consolidate_atol
```

があります。

robotorchan はこの upstream behavior をそのまま利用します。

---

## 10.19 raw data と consolidated data

robotorchan では constructor に渡された値を

```python
model.raw_datapoints
model.raw_comparisons
```

として保持します。

これは **caller input の provenance** です。

一方 BoTorch model 内部では duplicate consolidation 後の

```python
model.datapoints
model.comparisons
```

が使われる場合があります。

したがって両者は必ずしも同じ shape になるとは限りません。

---

## 10.20 robotorchan の `PairwiseGP`

基本構築は次の通りです。

```python
import torch
from robotorchan.models import PairwiseGP


datapoints = torch.tensor(
    [[0.1], [0.4], [0.8]],
    dtype=torch.double,
)
comparisons = torch.tensor(
    [[2, 1], [1, 0]],
    dtype=torch.long,
)

model = PairwiseGP(
    datapoints=datapoints,
    comparisons=comparisons,
)
```

---

## 10.21 constructor

robotorchan wrapper は BoTorch 0.18.1 と同じ主要引数を公開します。

```python
PairwiseGP(
    datapoints,
    comparisons,
    likelihood=None,
    covar_module=None,
    input_transform=None,
    jitter=1e-6,
    xtol=None,
    consolidate_rtol=0.0,
    consolidate_atol=1e-4,
    maxfev=None,
)
```

`datapoints` または `comparisons` が `None` の場合、upstream では prior-only model として構築できます。

---

## 10.22 `make_mll()`

robotorchan では

```python
mll = model.make_mll()
```

で

```python
PairwiseLaplaceMarginalLogLikelihood
```

を返します。

実体は概念的に

```python
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood

mll = PairwiseLaplaceMarginalLogLikelihood(
    model.likelihood,
    model,
)
```

です。

---

## 10.23 fitting

通常の robotorchan exact GP と同様、

```python
from botorch.fit import fit_gpytorch_mll

mll = model.make_mll()
fit_gpytorch_mll(mll)
```

で hyperparameter fitting できます。

ただし objective は Exact Gaussian MLL ではなく、**Laplace approximation に基づく marginal likelihood** です。

---

## 10.24 fitting の二重構造

PairwiseGP では fitting を理解するとき、2段階を区別するとよいです。

```text
inner problem
  latent utility f_MAP を求める

outer problem
  Laplace-approximated evidence を使って
  kernel hyperparameters を調整
```

つまり標準回帰GPより内部処理が複雑です。

---

## 10.25 posterior

学習後は通常の BoTorch model と同様に

```python
posterior = model.posterior(test_X)
```

で latent utility posterior を得られます。

ここで posterior が表しているのは

```text
観測スコア y
```

ではなく

```text
潜在 utility f(x)
```

です。

---

## 10.26 posterior mean の解釈

```python
mean = posterior.mean
```

が高い候補ほど、モデルは「より好まれやすい」と推定しています。

ただし absolute scale より ranking が重要です。

```text
posterior mean
x1: -0.8
x2:  0.2
x3:  1.1
```

なら

```text
x3 > x2 > x1
```

という ranking interpretation が自然です。

---

## 10.27 observation noise の扱い

通常の regression GP では `posterior(..., observation_noise=True)` で observation noise を含む predictive distribution を考えます。

PairwiseGP の probit preference model では noise と utility scale の識別が異なるため、通常回帰と同じ意味で observation-noise variance を解釈しません。

したがって preference model の uncertainty と regression noise を同一視しないことが重要です。

---

## 10.28 comparison graph

pairwise comparisons は graph として考えると理解しやすいです。

```text
A → B
A → C
C → D
B → D
```

矢印を

```text
winner → loser
```

とすると、observed comparisons は directed graph になります。

十分に連結された graph なら候補間の相対順位を推定しやすくなります。

---

## 10.29 comparison graph が疎すぎる場合

例えば

```text
A > B
C > D
```

だけで A/B group と C/D group の間に comparison が一つもなければ、2グループ間の relative utility は data からほとんど決まりません。

GP prior による input similarity が橋渡しする場合はありますが、comparison design 自体も重要です。

---

## 10.30 transitivity

latent utility model は基本的に

```text
A > B
B > C
→ A > C が高確率
```

という transitive ranking を想定します。

しかし実際の人間 preference では

```text
A > B
B > C
C > A
```

のような cyclic preference が起きることがあります。

強い非推移性がある問題では、単一 scalar utility という仮定自体を見直す必要があります。

---

## 10.31 noisy comparisons

同じ pair を複数回評価して

```text
A > B
A > B
B > A
A > B
```

となることは自然です。

PairwiseGP は deterministic ranking table ではなく probabilistic likelihood を使うため、このような矛盾を uncertainty として扱えます。

---

## 10.32 preference data を作るときの注意

比較 query では次を意識します。

- 差が明らかすぎる pair ばかり選ばない
- ほぼ同じ候補だけにも偏らない
- graph connectivity を確保する
- 同一 pair の再評価で noise を把握する
- evaluator drift を考慮する
- 複数 evaluator を混ぜるなら個人差を意識する

---

## 10.33 preference learning と classification の違い

比較結果は binary なので classification に見えます。

しかし通常の binary classification とは入力構造が違います。

```text
binary classification
x → class label

preference learning
(x_i, x_j) → i > j
```

さらに PairwiseGP は pair 自体ではなく、各候補に latent utility function `f(x)` を置いています。

---

## 10.34 preference learning と ranking

Preference Learning は ranking problem と密接です。

ただし目的は単に既存候補を並べ替えることだけではありません。

GP を使うことで、未評価の新しい `x` に対しても

\[
f(x)
\]

を予測できます。

この点が Bayesian Optimization に繋がります。

---

## 10.35 Preferential Bayesian Optimization

通常 BO では

```text
x を評価
  ↓
y = objective(x) を観測
  ↓
GP regression
```

です。

Preference BO では

```text
候補 x_i, x_j を提示
  ↓
どちらが良いか比較を観測
  ↓
PairwiseGP
  ↓
latent utility posterior
  ↓
次の比較候補を選ぶ
```

となります。

---

## 10.36 なぜ通常の EI をそのまま使わないのか

EI は通常、観測された objective value と incumbent を前提にします。

Preference Learning では absolute objective value を観測していません。

そのため

```text
best observed y
```

という概念が直接ありません。

Preference BO では、utility model に適した acquisition function を使います。

---

## 10.37 EUBO

BoTorch には preference optimization 向け acquisition function として

```python
AnalyticExpectedUtilityOfBestOption
```

があります。

EUBO は **Expected Utility of the Best Option** の略です。

候補集合を提示したとき、その中で最終的に選べる最良 option の期待 utility を高くするように query を設計します。

---

## 10.38 q=2 の直感

最も分かりやすい preference query は2候補比較です。

```text
candidate A
candidate B
   ↓
どちらが良い？
```

BoTorch の preference BO tutorial では `q=2` として EUBO を最適化する例があります。

```text
PairwiseGP posterior
      ↓
EUBO
      ↓
2候補を選ぶ
      ↓
比較結果を取得
      ↓
comparisonsに追加
```

となります。

---

## 10.39 EUBO と pure uncertainty sampling の違い

単に最も不確実な2点を比較する方法は information gathering には有効ですが、必ずしも最良候補の発見に効率的とは限りません。

EUBO は utility maximization という decision objective に沿って comparison query を選ぶ考え方です。

```text
uncertainty sampling
  → knowledge acquisition 重視

EUBO
  → best option utility 重視
```

です。

---

## 10.40 preference BO loop

概念的には

```python
model = PairwiseGP(datapoints, comparisons)
fit_gpytorch_mll(model.make_mll())

# preference acquisition を構築
# acquisition を最大化して next_X を選ぶ
# next_X 内の pairwise comparison を取得
# datapoints / comparisons に追加
# refit
```

となります。

---

## 10.41 新しい比較の index 更新

既存 datapoints が `n` 個あり、新しく2点を追加した場合、新しい comparison index は既存行数を offset として加える必要があります。

```text
既存 datapoints: 0 ... n-1
新規 point A: n
新規 point B: n+1

A > B
→ [n, n+1]
```

この index 管理は preference BO 実装でよく間違える点です。

---

## 10.42 `condition_on_observations`

`PairwiseGP` は `condition_on_observations` をサポートしますが、通常回帰モデルと違い、新しい `Y` も pairwise comparisons である必要があります。

つまり

```text
X = new datapoints
Y = scalar targets
```

ではなく

```text
X = new datapoints
Y = comparison indices
```

です。

---

## 10.43 raw data API

robotorchan では

```python
model.raw_datapoints
model.raw_comparisons
```

を提供します。

通常の supervised wrappers の

```python
raw_train_X
raw_train_Y
```

とは命名を分けています。

Preference data の semantics を明確にするためです。

---

## 10.44 input normalization

PairwiseGP でも kernel を使うため、連続入力なら unit cube への normalization が有効です。

例えば

```python
from botorch.models.transforms.input import Normalize

model = PairwiseGP(
    datapoints=datapoints,
    comparisons=comparisons,
    input_transform=Normalize(d=datapoints.shape[-1]),
)
```

とできます。

---

## 10.45 outcome standardization はどうするか

通常回帰では `Standardize` を使えますが、Preference Learning では直接観測する scalar target がありません。

したがって

```text
observed Y を standardize
```

という考え方はそのまま適用できません。

latent utility の scale は likelihood と kernel の中で扱われます。

---

## 10.46 hyperparameter fitting

PairwiseGP の kernel hyperparameters も preference data に基づいて学習されます。

例えば lengthscale は

```text
input のどの距離スケールで preference utility が変化するか
```

を表します。

ただし comparison 数が少ないと、通常回帰以上に hyperparameter uncertainty が大きくなりやすい点に注意します。

---

## 10.47 comparison 数 m と datapoint 数 n

Preference dataset では

- datapoints 数 `n`
- comparisons 数 `m`

を分けて考えます。

全 pair を比較すると

\[
\frac{n(n-1)}{2}
\]

件になるため、`n` が大きいと全比較は非現実的です。

Preference BO / Active Learning の目的の一つは、少ない comparison で有益な ranking を学ぶことです。

---

## 10.48 repeated comparison

同じ pair を繰り返し評価することにも意味があります。

```text
A vs B を1回だけ
```

より

```text
A vs B を複数回
```

比較することで、judge noise を反映できます。

ただし repeated comparison に budget を使いすぎると exploration が減ります。

---

## 10.49 evaluator variability

複数人が評価する場合、

```text
Person 1: A > B
Person 2: B > A
```

は measurement noise ではなく preference heterogeneity かもしれません。

単一 `PairwiseGP` は基本的に1つの latent utility function を想定するため、個人差が大きい場合は

- evaluator ID を task として扱う
- hierarchical model
- multi-task preference model

などの拡張が必要です。

---

## 10.50 context-dependent preference

好みが context に依存する場合、単純な

\[
f(x)
\]

では不十分です。

例えば

\[
f(x,c)
\]

として context `c` を入力に含める必要があります。

```text
材料Aが良い
```

ではなく

```text
用途Cでは材料Aが良い
```

という問題です。

---

## 10.51 cyclic preference が強い場合

単一 scalar latent utility では

```text
A > B
B > C
C > A
```

を自然に説明しにくいです。

このような場合、

- context dependence
- evaluator heterogeneity
- non-transitive preference model

を検討します。

PairwiseGP を無理に当てはめないことが重要です。

---

## 10.52 preference learning と multi-objective

Multi-objective BO と Preference Learning は別概念です。

```text
Multi-objective BO
  → 複数の数値 objective を観測

Preference Learning
  → 総合的な好みを comparison として観測
```

ただし両者は組み合わせられます。

---

## 10.53 BOPE

BoTorch には Bayesian Optimization with Preference Exploration、つまり **BOPE** の考え方があります。

複数 outcome

\[
y=f(x)\in\mathbb{R}^m
\]

は測定できるが、それをどう総合評価するかの utility

\[
g(y)
\]

が分からない場合を考えます。

```text
x
 ↓
outcome model f
 ↓
y1, y2, ..., ym
 ↓
preference model g
 ↓
utility
```

という2段階モデルです。

---

## 10.54 BOPE と直接 Preference BO の違い

直接 Preference BO では

```text
x → latent utility
```

を直接 PairwiseGP で学習します。

BOPE では

```text
x → outcomes y

y → utility
```

を分けます。

複数の物理量は測れるが、人間がどの trade-off を好むか分からない場合に BOPE が有効です。

---

## 10.55 材料開発での例

例えば材料候補について

- 強度
- 靭性
- コスト
- 加工性
- 外観

を総合して「どちらを採用したいか」を専門家が判断するとします。

絶対 utility score を定義しにくくても、

```text
候補Aと候補BならA
```

という判断は可能です。

このとき PairwiseGP で expert utility を学習することができます。

---

## 10.56 通常回帰を使った方がよい場合

信頼できる absolute scalar objective が取れるなら、通常回帰の方が単純です。

```text
毎回、正確な強度値が測れる
→ SingleTaskGP

最終評価が人の総合判断で数値化しにくい
→ PairwiseGP を検討
```

です。

Preference Learning は「比較形式でしか情報が得られない / 比較の方が信頼できる」場合に使うのが自然です。

---

## 10.57 scalarization との違い

複数 objective がある場合、重み付き和

\[
u(y)=w^T y
\]

を明示できるなら preference model は不要かもしれません。

一方、重みを人が明示できず comparison なら答えられる場合、Preference Learning が有効です。

---

## 10.58 PairwiseGP の計算コスト

PairwiseGP では latent utility MAP と Laplace approximation が必要なため、standard exact GP regression より fitting が重くなる場合があります。

特に

- datapoints 数
- comparisons 数
- duplicate consolidation
- hyperparameter optimization

が大きいと計算負荷が増えます。

---

## 10.59 numerical stability

constructor には

```python
jitter=1e-6
```

があります。

これは Cholesky 等の numerical stability のために使われます。

また

```python
xtol
maxfev
```

は latent utility MAP を求める数値解法の収束制御に関係します。

通常は default から始め、収束問題が出た場合に調整します。

---

## 10.60 comparison noise と scale

probit preference model では utility scale と noise scale が完全に独立には識別できません。

BoTorch は likelihood 内で明示的な \(\sigma\) を持たせる代わりに、kernel output scale 側で function scale を扱います。

そのため通常 regression GP の

```text
learned noise = measurement noise variance
```

のような解釈をそのまましないことが重要です。

---

## 10.61 model diagnostics

Preference model では次を確認します。

1. pairwise accuracy
2. Kendall tau / Spearman correlation
3. held-out comparison log likelihood
4. posterior ranking stability
5. top candidate stability
6. comparison graph connectivity
7. repeated comparison consistency
8. hyperparameter stability
9. utility posterior uncertainty
10. BO performance

---

## 10.62 held-out comparison accuracy

comparison dataset を train / validation に分け、validation pair `(i,j)` について

```text
posterior mean utility_i > posterior mean utility_j ?
```

を見るだけでも簡単な診断になります。

ただし uncertainty を無視するため、可能なら predictive preference probability を使う方がよいです。

---

## 10.63 ranking metric

ground truth ranking がある simulation では Kendall tau が便利です。

```text
true order
A > B > C > D

predicted order
A > C > B > D
```

のような順位差を評価できます。

Preference model は absolute utility recovery より ranking recovery が重要な場合が多いためです。

---

## 10.64 uncertainty の利用

PairwiseGP は point ranking だけでなく posterior uncertainty を持ちます。

これにより

```text
どちらが優れているか分からない領域
```

を active に比較できます。

この uncertainty-aware query design が、単なるランキング手法と preference BO の大きな違いです。

---

## 10.65 preference Active Learning

目的が「最適候補を見つける」ではなく「ranking model を効率よく学ぶ」なら、Active Learning として比較 query を選ぶこともできます。

例えば

- entropy
- pairwise uncertainty
- information gain
- disagreement

などを基準にできます。

BO と Active Learning では query selection objective が異なる点に注意します。

---

## 10.66 BO と Active Learning の違い

```text
Preference BO
  → 最良 utility candidate を見つけたい

Preference Active Learning
  → preference function / ranking を正確に学びたい
```

したがって acquisition function も変わります。

EUBO は前者寄りです。

---

## 10.67 candidate set から比較を選ぶ場合

連続最適化ではなく、既存候補集合

\[
\mathcal{C}=\{x_1,\dots,x_N\}
\]

から pair を選ぶ場合、全 pair に acquisition score を計算して

\[
(i^*,j^*)=
\arg\max_{i\neq j}\alpha(x_i,x_j)
\]

とすることもできます。

候補数が大きいと pair 数は \(O(N^2)\) なので効率化が必要です。

---

## 10.68 preference query batch

一度に複数 pair を人へ提示する場合、comparison batch 内の冗長性も問題になります。

```text
A vs B
A vs C
A vs D
```

ばかり選ぶより、情報が分散するように query を選ぶ方がよい場合があります。

これは batch acquisition の問題です。

---

## 10.69 PairwiseGP と classifier の使い分け

例えば product A/B のどちらを選ぶかだけを予測したいなら pair-input classifier でも実装できます。

しかし

- unseen candidate の utility を推定したい
- ranking を作りたい
- BO に使いたい
- posterior uncertainty が欲しい

なら latent utility model である PairwiseGP が自然です。

---

## 10.70 robotorchan Notebook

実行例は

```text
examples/notebooks/07_pairwise_gp.ipynb
```

にあります。

Notebook では

1. latent utility を用意
2. comparison data を生成
3. PairwiseGP を構築
4. Laplace MLL を fitting
5. posterior utility を可視化
6. posterior-best point を確認

という流れを実行できます。

---

## 10.71 raw-data provenance

Notebook / wrapper では

```python
model.raw_datapoints
model.raw_comparisons
```

を確認できます。

これは model fitting 後の consolidated state ではなく constructor input の snapshot です。

再現性やデバッグではこの区別が重要です。

---

## 10.72 simple example

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import PairwiseGP


torch.set_default_dtype(torch.double)

x = torch.linspace(0.0, 1.0, 12).unsqueeze(-1)
true_utility = torch.sin(2 * torch.pi * x).squeeze(-1)

pairs = []
for i in range(11):
    j = i + 1
    if true_utility[i] >= true_utility[j]:
        pairs.append([i, j])
    else:
        pairs.append([j, i])

comparisons = torch.tensor(pairs, dtype=torch.long)

model = PairwiseGP(
    datapoints=x,
    comparisons=comparisons,
)

mll = model.make_mll()
fit_gpytorch_mll(mll)
```

---

## 10.73 posterior ranking

```python
test_X = torch.linspace(0.0, 1.0, 200).unsqueeze(-1)
posterior = model.posterior(test_X)
mean = posterior.mean.squeeze(-1)

best_idx = mean.argmax()
best_x = test_X[best_idx]
```

これにより比較データしか与えていなくても、latent utility posterior に基づいて好ましい領域を推定できます。

---

## 10.74 よくある誤解

### 「comparisons の数字が score」

違います。datapoints の index です。

### 「[i, j] は i < j の意味」

違います。`i` が winner、`j` が loser です。

### 「PairwiseGP は binary classifier」

単純な classifier ではなく、候補ごとの latent utility GP を学習します。

### 「utility の絶対値に意味がある」

基本的には relative scale / ranking が重要です。

### 「PairwiseGP は Exact Gaussian regression」

違います。probit likelihood と Laplace approximation を使います。

### 「EI をそのまま使えばよい」

absolute objective observations がないので preference-specific acquisition を検討します。

---

## 10.75 モデル選択フロー

```text
評価対象がある
   ↓
信頼できる absolute scalar score を取得できる？
 ├─ Yes
 │    └─ SingleTaskGP 等
 │
 └─ No
      ↓
      pairwise comparison なら安定して得られる？
       ├─ Yes
       │    └─ PairwiseGP
       │
       └─ No
            └─ 別の preference / ordinal model を検討
```

さらに

```text
複数 outcome は測れる
しかし総合 utility が分からない
        ↓
       BOPE
```

も候補です。

---

## 10.76 実務比較表

| 状況 | 第一候補 |
|---|---|
| scalar target を直接測れる | `SingleTaskGP` |
| 人の比較評価のみ | `PairwiseGP` |
| ranking を学びたい | `PairwiseGP` |
| 官能評価 | `PairwiseGP` |
| 複数 outcome + preference | BOPE |
| ordinal score が直接ある | ordinal model を検討 |
| evaluator ごとの差が大きい | hierarchical / multitask preference model を検討 |
| cyclic preference が強い | non-transitive model を検討 |

---

## 10.77 robotorchan で覚えておく API

```python
model = PairwiseGP(
    datapoints=datapoints,
    comparisons=comparisons,
)

model.raw_datapoints
model.raw_comparisons
model.supports_mll
model.make_mll()
model.posterior(test_X)
```

`make_mll()` は

```python
PairwiseLaplaceMarginalLogLikelihood
```

を返します。

---

## 10.78 数学的な全体像

Preference Learning の構造は

\[
f\sim\mathcal{GP}(m,k)
\]

\[
P(i\succ j\mid f)
=
\Phi\left(
\frac{f_i-f_j}{\sqrt{2}}
\right)
\]

\[
p(f\mid D)
\propto
p(D\mid f)p(f)
\]

です。

この posterior は非Gaussianなので、

\[
p(f\mid D)
\approx
\mathcal{N}(f_{MAP},\Sigma_{Laplace})
\]

と Laplace approximation します。

---

## 10.79 ベイズ最適化との接続

通常 BO:

```text
objective value
      ↓
regression GP
      ↓
EI / UCB / KG
```

Preference BO:

```text
pairwise comparison
      ↓
PairwiseGP
      ↓
latent utility posterior
      ↓
EUBO / preference acquisition
```

という違いです。

---

## 10.80 まとめ

Preference Learning は、**絶対値ではなく比較から潜在 utility function を学ぶ方法**です。

robotorchan では

```python
PairwiseGP
```

を利用します。

特に重要なのは次の点です。

1. `comparisons` は `[winner, loser]` の index pair
2. latent utility `f(x)` に GP prior を置く
3. default likelihood は probit
4. non-Gaussian posterior を Laplace approximation する
5. `make_mll()` は `PairwiseLaplaceMarginalLogLikelihood`
6. utility の absolute value より ranking が重要
7. preference BO では EI より preference-specific acquisition を考える
8. BoTorch では EUBO が利用できる
9. multiple outcomes と preference を分ける場合は BOPE が有力
10. evaluator heterogeneity や cyclic preference が強い場合は単一 utility 仮定を見直す

次章では、画像・波形・空間場などの structured output を扱う [Structured Output GP](11_structured_output.md) を説明します。

## 参考

- Chu, Ghahramani, *Preference Learning with Gaussian Processes*, ICML 2005
- Brochu, Cora, de Freitas, *A Tutorial on Bayesian Optimization of Expensive Cost Functions*, 2010
- Lin et al., *Preference Exploration for Efficient Bayesian Optimization with Multiple Outcomes*, AISTATS 2022
- Astudillo et al., *qEUBO: A Decision-Theoretic Acquisition Function for Preferential Bayesian Optimization*, AISTATS 2023
- BoTorch `PairwiseGP`
- BoTorch preference BO tutorial
- BoTorch BOPE tutorial
- [Gaussian Process](02_gaussian_process.md)
- [Acquisition Function](04_acquisition_function.md)
