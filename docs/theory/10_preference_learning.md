# 10. Preference Learning

## 10.1 この章で扱うこと

通常の回帰では、各入力 `x` に対して数値 target `y` を観測します。

一方、Preference Learning では絶対スコアではなく、


```text
A の方が B より好ましい
B の方が C より好ましい
```

という **比較結果** だけを観測します。

本章では、

- preference data
- latent utility
- pairwise comparison likelihood
- probit model
- GP prior
- Laplace approximation
- `Laplace 近似対数周辺尤度`
- posterior utility
- pairwise BO
- Expected Utility of Best Option (EUBO)
- BOPE との関係

を順に整理します。

---

### 回帰との違いを最初に図で見る

通常の回帰とPreference Learningでは、**観測しているもの**が違います。


~~~text
Regression
x ──→ y
      ↑
      絶対的な数値を観測

Preference Learning
x_i ─┐
     ├──→ i が j より好まれた
x_j ─┘
          ↑
          相対的な比較だけを観測
~~~

Preference LearningでGPが直接説明しようとするのはwinner/loserのindexそのものではなく、
その比較を生み出す背後のlatent utility $`f(x)`$ です。

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


$$
f(x)
$$

が存在すると仮定します。

例えば


$$
f(x_A) > f(x_B)
$$

なら、モデル上はAがBより好まれる確率が高くなります。

ただし比較likelihoodが確率的なので、$`f(x_A) > f(x_B)`$ だからといって
観測される比較結果が必ずA勝利になるわけではありません。

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

比較データは、候補集合 $X=(x_1,\ldots,x_n)$ と勝者・敗者の添字対
$(i,j)$ から構成され、$x_i$ が $x_j$ より好まれたことを表します。

---

## 10.5 `comparisons` は target value ではない

通常回帰では

`観測入力` / `観測応答`

を使います。

Preference Learning では

`datapoints` / `comparisons`

を使います。

重要なのは、`comparisons` の値 `0, 1, 2, ...` は utility 値ではなく **datapoints の行index** だということです。


```text
comparisons[k] = [i, j]

意味:
候補 i が候補 j より好まれた
```

です。

さらに、同じdatapointを複数のcomparisonで再利用できます。つまりcomparisonの行数と
候補点の行数は一致する必要がありません。このindex表現によって、同じ候補を毎回
複製せずに比較graphを表現できます。

---

## 10.6 比較確率

2つの候補 `x_i`, `x_j` の utility を


$$
f_i = f(x_i),\qquad f_j = f(x_j)
$$

とします。

Preference model では、


$$
P(i \succ j)
$$

つまり「i が j より好まれる確率」を utility difference


$$
f_i-f_j
$$

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

基本的な形は


$$
P(i \succ j)
=
\Phi\left(
\frac{f_i-f_j}{\sqrt{2}}
\right)
$$

です。

ここで

- $`\Phi`$ は標準正規分布のCDF
- $`f_i-f_j`$ は latent utility difference

です。

---

## 10.8 Probit と Logistic

pairwise comparison model では logistic model もよく使われます。

例えば Bradley–Terry 型なら


$$
P(i \succ j)
=
\sigma(f_i-f_j)
$$

です。

ここで $`\sigma`$ は sigmoid です。


```text
Bradley–Terry / logistic
  → sigmoid

probit を選ぶ場合
  → Gaussian CDF / probit
```

どちらも utility difference から preference probability を作る考え方は同じです。

---

## 10.9 GP prior on latent utility

utility function `f(x)` 自体には Gaussian Process prior を置きます。


$$
f \sim \mathcal{GP}(m(x),k(x,x'))
$$

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

### 比較データは「点」だけでなく「辺」の情報を持つ

通常の回帰では、1観測はおおむね1つの入力点へ対応します。
Preference Learningでは、1観測が**2候補間の辺**を追加します。


~~~text
datapoints = graph nodes
comparisons = directed edges
winner → loser
~~~

したがって同じ候補数でも、どのpairを比較したかによって得られる情報は大きく変わります。
これは後半のquery designやPreference BOにつながります。

## 10.10 なぜ通常の GP regression と違うのか

通常の Gaussian regression なら


$$
y=f(x)+\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,\sigma^2)
$$

であり、Gaussian prior と Gaussian likelihood の組み合わせなので posterior は解析的に Gaussian になります。

Preference Learning では likelihood が


$$
\Phi(f_i-f_j)
$$

のような nonlinear function になるため、posterior を exact Gaussian として解析的には求められません。

そこで pairwise GP は **Laplace approximation** を使います。

---

## 10.11 Laplace approximation

観測 comparison data を $`D`$ とすると、latent utilities `f` の posterior は


$$
p(f\mid D)
\propto
p(D\mid f)p(f)
$$

です。

しかし preference likelihood が non-Gaussian なので、この posterior は Gaussian ではありません。

Laplace approximation では、まず posterior mode


$$
f_{MAP}
=
\arg\max_f p(f\mid D)
$$

を求めます。

その周辺を2次近似して


$$
p(f\mid D)
\approx
\mathcal{N}(f_{MAP},\Sigma_{Laplace})
$$

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

このため pairwise GP は名前に GP とありますが、標準 単一タスク GP と posterior fitting の仕組みが異なります。

---

## 10.13 MAP utility

pairwise GP 内部では preference comparisons と GP prior の両方を考慮して latent utility の MAP 解を求めます。

概念的には


$$
f_{MAP}
=
\arg\max_f
\left[
\log p(D\mid f)
+
\log p(f)
\right]
$$

です。

第一項は比較結果への適合、第二項はGP priorによる正則化です。

---

## 10.14 Laplace covariance

posterior mode の周辺で negative log posterior の Hessian を使うと、近似 covariance は概念的に


$$
\Sigma_{Laplace}
\approx
\left(
K^{-1}+W
\right)^{-1}
$$

となります。

ここで

- $`K`$ は GP prior covariance
- $`W`$ は preference likelihood の曲率

です。

比較情報が多い領域では likelihood curvature が強くなり、utility uncertainty が縮みます。

---

## 10.16 utility の絶対値は重要ではない

pairwise data で観測されるのは utility difference です。

例えば、


$$
f(x)\rightarrow f(x)+c
$$

として全候補に同じ定数を足しても、


$$
(f_i+c)-(f_j+c)=f_i-f_j
$$

なので preference probability は変わりません。

したがって比較likelihoodだけからはutilityの **絶対的な基準点** は識別されません。

またutilityのscaleについても、likelihoodのnoise conventionやkernel scaleとの関係があるため、
通常の回帰targetと同じ物理単位を持つ量として解釈しないことが重要です。
Preference modelでは主に**差・順位・選好確率**に意味があります。

---

## 10.17 relative scale と評価指標

Preference Learning では通常の RMSE だけでなく、ranking quality を見るのが自然です。

例えば

- Kendall's tau
- Spearman rank correlation
- pairwise accuracy
- top-k recovery

などです。

---

## 10.18 duplicate datapoints の consolidation

同じ入力が繰り返し現れる場合、比較関係を保ちながら重複候補を統合できます。
この操作は観測数とユニークな候補数の違いに注意が必要です。

---

## 10.19 raw data と consolidated data

観測時の候補・比較と、重複統合後に推論へ用いる候補・比較は区別します。
統合前後で添字付けが変化し得ます。

---

---

---

## 10.23 fitting

---

## 10.24 fitting の二重構造

pairwise GP では fitting を理解するとき、2段階を区別するとよいです。


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

比較観測から推論するのは、絶対的な観測スコアではなく潜在効用 $u(x)$ の事後分布です。

---

## 10.26 posterior mean の解釈

潜在効用の事後平均による順位付けは可能ですが、絶対水準より候補間の相対関係が重要です。

---

## 10.27 observation noise の扱い

pairwise GP の probit preference model では noise と utility scale の識別が異なるため、通常回帰と同じ意味で
observation-noise variance を解釈しません。

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

graph connectivityは、離れた候補群の相対utilityをデータから結び付けるうえで重要です。
ただしGP priorはinput similarityを通じても情報を共有するため、graphだけでposteriorの
識別性や精度が完全に決まるわけではありません。

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

pairwise GPはdeterministic ranking tableではなくprobabilistic likelihoodを使うため、
このような矛盾した観測にも有限のlikelihoodを与えられます。

ただし矛盾がそのまま1個の「uncertainty parameter」へ変換されるわけではありません。
強い評価者差や時間変化があるなら、単一latent utility modelの仮定自体も確認します。

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

さらに pairwise GP は pair 自体ではなく、各候補に latent utility function `f(x)` を置いています。

---

## 10.34 preference learning と ranking

Preference Learning は ranking problem と密接です。

ただし目的は単に既存候補を並べ替えることだけではありません。

GP を使うことで、未評価の新しい `x` に対しても


$$
f(x)
$$

を予測できます。

この点が Bayesian Optimization に繋がります。

---

### Preference LearningとPreference BOは別の層

ここまでの pairwise GP は、**比較データからutility posteriorを作るsurrogate model**です。
これだけでは「次にどの2候補を比較するか」は決まりません。


~~~text
pairwise GP
  → preference surrogate

Preference acquisition
  → 次に比較する候補を評価

Acquisition optimization
  → 実際のquery pairを決定
~~~

通常のBOでGPとEIが別物なのと同じく、Preference LearningとPreferential BOも
分けて考えます。

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
pairwise GP
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

Expected Utility of the Best Option（EUBO）は、提示する候補集合で得られる
最大潜在効用の期待値を評価する獲得基準です。

---

## 10.38 q=2 の直感

最も分かりやすい preference query は2候補比較です。


```text
candidate A
candidate B
   ↓
どちらが良い？
```


```text
pairwise GP posterior
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

潜在効用を比較観測から推論し、次の比較候補を選び、比較結果を追加して
事後分布を更新する反復過程です。

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

pairwise GP は `condition_on_observations` をサポートしますが、通常回帰モデルと違い、新しい `Y` も pairwise comparisons
である必要があります。

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

## 10.44 input normalization

入力の尺度が大きく異なる場合、カーネルの距離尺度を適切に設定するため
正規化が有効になることがあります。

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

pairwise GP の kernel hyperparameters も preference data に基づいて学習されます。

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


$$
\frac{n(n-1)}{2}
$$

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

単一 pairwise GP は基本的に1つの latent utility function を想定するため、個人差が大きい場合は

- evaluator ID を task として扱う
- hierarchical model
- multi-task preference model

などの拡張が必要です。

---

## 10.50 context-dependent preference

好みが context に依存する場合、単純な


$$
f(x)
$$

では不十分です。

例えば


$$
f(x,c)
$$

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

pairwise GP を無理に当てはめないことが重要です。

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

複数 outcome


$$
y=f(x)\in\mathbb{R}^m
$$

は測定できるが、それをどう総合評価するかの utility


$$
g(y)
$$

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

を直接 pairwise GP で学習します。

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

このとき pairwise GP で expert utility を学習することができます。

---

## 10.56 通常回帰を使った方がよい場合

信頼できる absolute scalar objective が取れるなら、通常回帰の方が単純です。


```text
毎回、正確な強度値が測れる
→ 単一タスク GP

最終評価が人の総合判断で数値化しにくい
→ pairwise GP を検討
```

です。

Preference Learning は「比較形式でしか情報が得られない / 比較の方が信頼できる」場合に使うのが自然です。

---

## 10.57 scalarization との違い

複数 objective がある場合、重み付き和


$$
u(y)=w^T y
$$

を明示できるなら preference model は不要かもしれません。

一方、重みを人が明示できず comparison なら答えられる場合、Preference Learning が有効です。

---

## 10.58 pairwise GP の計算コスト

pairwise GP では latent utility MAP と Laplace approximation が必要なため、standard exact GP regression より
fitting が重くなる場合があります。

特に

- datapoints 数
- comparisons 数
- duplicate consolidation
- hyperparameter optimization

が大きいと計算負荷が増えます。

---

## 10.59 numerical stability

Laplace近似や行列分解では数値的な安定化が必要です。
安定化の強さと最適化収束精度は近似結果に影響します。

---

## 10.60 comparison noise と scale

probit preference model では utility scale と noise scale が完全に独立には識別できません。

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

pairwise GP は point ranking だけでなく posterior uncertainty を持ちます。

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


$$
\mathcal{C}=\{x_1,\dots,x_N\}
$$

から pair を選ぶ場合、全 pair に acquisition score を計算して


$$
(i^*,j^*)=
\arg\max_{i\neq j}\alpha(x_i,x_j)
$$

とすることもできます。

候補数が大きいと pair 数は $`O(N^2)`$ なので効率化が必要です。

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

## 10.69 pairwise GP と classifier の使い分け

例えば product A/B のどちらを選ぶかだけを予測したいなら pair-input classifier でも実装できます。

しかし

- unseen candidate の utility を推定したい
- ranking を作りたい
- BO に使いたい
- posterior uncertainty が欲しい

なら latent utility model である pairwise GP が自然です。

---

## 10.71 raw-data provenance

比較観測の原データと、推論用に整理したデータを区別することは
推定結果の再現性と解釈に重要です。

---

## 10.72 simple example

---

## 10.73 posterior ranking

潜在効用の事後平均を最大にする点は、一つの意思決定基準です。
ただし事後不確実性を考慮する探索基準とは異なります。

---

### 「comparisons の数字が score」

違います。datapoints の index です。

### 「[i, j] は i < j の意味」

違います。`i` が winner、`j` が loser です。

### 「pairwise GP は binary classifier」

単純な classifier ではなく、候補ごとの latent utility GP を学習します。

### 「utility の絶対値に意味がある」

基本的には relative scale / ranking が重要です。

### 「pairwise GP は Exact Gaussian regression」

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
 │    └─ 単一タスク GP 等
 │
 └─ No
      ↓
      pairwise comparison なら安定して得られる？
       ├─ Yes
       │    └─ pairwise GP
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

## 10.76 観測形式とモデル仮定の比較表

| 観測・仮定 | 対応する考え方 |
|---|---|
| scalar targetを直接観測 | regression GP |
| pairwise comparisonを観測 | pairwise GP |
| 複数outcomeは観測でき、utilityを比較で学ぶ | BOPE型の分離 |
| ordinal scoreを直接観測 | ordinal model |
| evaluator差を明示的に表現したい | hierarchical / multitask preference model |
| 強いcyclic preferenceがある | scalar utility仮定を再検討 |

---

## 10.78 数学的な全体像

Preference Learning の構造は


$$
f\sim\mathcal{GP}(m,k)
$$


$$
P(i\succ j\mid f)
=
\Phi\left(
\frac{f_i-f_j}{\sqrt{2}}
\right)
$$


$$
p(f\mid D)
\propto
p(D\mid f)p(f)
$$

です。

この posterior は非Gaussianなので、


$$
p(f\mid D)
\approx
\mathcal{N}(f_{MAP},\Sigma_{Laplace})
$$

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
pairwise GP
      ↓
latent utility posterior
      ↓
EUBO / preference acquisition
```

という違いです。

---

## 10.80 この章で覚えておくこと

Preference Learning は、**絶対値ではなく比較から潜在 utility function を学ぶ方法**です。

pairwise GP

を利用します。

特に重要なのは次の点です。

1. `comparisons` は `[winner, loser]` の index pair
2. latent utility `f(x)` に GP prior を置く
4. non-Gaussian posterior を Laplace approximation する
6. utility の absolute value より ranking が重要
7. preference BO では EI より preference-specific acquisition を考える
8. BoTorch では EUBO が利用できる
9. multiple outcomes と preference を分ける場合は BOPE が有力
10. evaluator heterogeneity や cyclic preference が強い場合は単一 utility 仮定を見直す

次章では、画像・波形・空間場などの structured output を扱う [Structured Output GP](11_structured_output.md) を説明します。

## 参考文献

1. Chu, W., and Ghahramani, Z. (2005).
   Preference learning with Gaussian processes.
   *Proceedings of the 22nd International Conference on Machine Learning*, 137-144.
2. Lin, Z. J. et al. (2022).
   Preference exploration for efficient Bayesian optimization with multiple outcomes.
   *Proceedings of AISTATS*.
3. Astudillo, R. et al. (2023).
   qEUBO: A decision-theoretic acquisition function for preferential Bayesian optimization.
   *Proceedings of AISTATS*.
4. BoTorch pairwise GP, preference BO, and BOPE documentation.
5. [Gaussian Process](02_gaussian_process.md)
6. [Acquisition Function](04_acquisition_function.md)
