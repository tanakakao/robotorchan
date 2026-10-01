# 6. Multi-Fidelity

## 6.1 Multi-Fidelity とは

ベイズ最適化では、目的関数を1回評価するだけでも大きなコストがかかることがあります。

しかし実務では、同じ現象について精度とコストが異なる複数の評価方法を利用できる場合があります。

```text
低精度シミュレーション   : 安い / 速い / 誤差が大きい
高精度シミュレーション   : 高い / 遅い / 誤差が小さい
実験                     : 非常に高い / 最も信頼したい
```

このような「評価の忠実度」を **fidelity** と呼び、複数のfidelityから得られる情報を
統合して最適化する方法が **Multi-Fidelity Bayesian Optimization (MFBO)** です。

通常のBOが

$$
x^*=\arg\max_x f(x)
$$

を考えるのに対して、Multi-Fidelityでは

$$
y=f(x,s)+\epsilon
$$

のようにfidelity変数 $`s`$ を追加します。

ここで重要なのは、最終的に欲しい解が

$$
x^*=\arg\max_x f(x,s_{target})
$$

であることです。

つまり、**低fidelityそのものを最適化したいわけではありません**。低コストの評価を利用して、target fidelityでの最適解を効率よく見つけることが目的です。

---

### Fidelityは「別の目的」ではなく「同じ対象を見る解像度」

初心者はfidelityを「目的変数が複数ある」と捉えやすいですが、基本的には異なります。
例えば同じ材料強度を、

~~~text
簡易simulation   → 安いが粗い
高精度simulation → 高価だが精密
実験              → 最終的に信頼したい
~~~

という異なる方法で調べていると考えます。

低fidelityの値がtargetと完全に一致する必要はありません。重要なのは、
**低fidelityを観測することでtarget fidelityについての不確実性が減るか**です。

## 6.2 Single-Fidelity BOとの違い

Single-Fidelity BOでは、各候補 $`x`$ に対して評価方法は1つです。

```text
Candidate x
   ↓
Expensive evaluation
   ↓
y
```

Multi-Fidelityでは、候補だけでなく「どのfidelityで評価するか」も選択対象になります。

```text
Candidate x
   +
Fidelity s
   ↓
Evaluation at (x, s)
   ↓
y
```

したがって次の評価点は概念的に

$$
(x_{next},s_{next})
$$

です。

MFBOでは、

1. **どの設計点を評価するか**
2. **どのfidelityで評価するか**

を同時に考えます。

---

## 6.3 Fidelityの例

### 計算時間

```text
10 iteration
100 iteration
1000 iteration
```

iteration数が増えるほど精度が高くなる場合です。

### メッシュ解像度

```text
粗いmesh
中程度のmesh
細かいmesh
```

CAEや流体解析などで典型的です。

### データ量

```text
10 % data
50 % data
100 % data
```

機械学習モデルの性能評価などで利用できます。

### 物理モデル

```text
簡易モデル
中精度シミュレーション
高精度シミュレーション
実験
```

材料・製造分野では特に重要な構成です。

---

## 6.4 Fidelityは単なるカテゴリではない

例えば

```text
low
medium
high
```

という3水準があっても、これを単なるカテゴリ変数として扱うだけではfidelityの特徴を十分に利用できない場合があります。

Fidelityには通常、

```text
精度または近似レベル
評価コスト
target fidelityとの統計的な関係
```

という構造があります。

ただし「fidelityが高いほど必ず情報量が大きい」という単純な順序を
無条件に仮定するべきではありません。targetについてどれだけ有用かは、
設計点、モデル化した相関、観測ノイズなどにも依存します。

特にcontinuous fidelityなら

$$
s\in[0,1]
$$

として

```text
s = 0.1  : low fidelity
s = 0.5  : medium fidelity
s = 1.0  : target fidelity
```

のような連続的関係を持たせられます。

したがって、前章のMixed Variablesで説明したcategorical variableとfidelity variableは区別する必要があります。

---

## 6.5 Multi-Fidelity GPの基本思想

Multi-Fidelity GPでは、設計変数 $`x`$ とfidelity $`s`$ をまとめて入力とします。

$$
\tilde{x}=(x,s)
$$

そして

$$
f(x,s)\sim\mathcal{GP}(m(x,s),k((x,s),(x',s')))
$$

とモデル化します。

重要なのはkernelが、設計変数とfidelityの構造を通して異なる観測間の
covarianceを表現することです。概念的には

```text
設計点 x の関係
+
fidelity s の関係
        ↓
異なる (x, s) 間の covariance
```

と捉えられます。

そのため、低fidelityで観測した情報をtarget fidelityの予測へ伝播できます。

---

## 6.6 Fidelity間の相関

Multi-Fidelityが有効なのは、低fidelityと高fidelityの間に情報共有できる関係があるためです。

例えば

```text
low fidelity  : 全体傾向は合うがbiasがある
high fidelity : 正確だが高コスト
```

なら、low fidelityを多数評価して全体構造を把握し、必要な場所だけhigh fidelityを評価できます。

一方、low fidelityとtarget fidelityがほぼ無相関なら、low fidelityを大量に取得してもtargetの最適化には役立ちません。

したがってMFBOの本質は

> **安い評価が、target fidelityについてどれだけ情報を持つか**

にあります。

---

### 相関があることと「同じ値」であることは違う

例えばlow fidelityが常にtargetより小さめに出るbiasを持っていても、
設計変数に対する増減傾向が共有されていれば有用な場合があります。

~~~text
low    : 値そのものにはbiasがあるが、良い領域・悪い領域の傾向は似る
target : 絶対値は異なるが、lowから情報を受け取れる
~~~

したがってMFBOでは「値が近いか」だけでなく、
**targetのposterior更新に役立つ統計的依存関係があるか**を見る必要があります。

## 6.7 Target Fidelity

Multi-Fidelityで最終的に最適化したいfidelityをtarget fidelityと考えます。

例えば

$$
s_{target}=1
$$

なら、最終目的は

$$
x^*=\arg\max_x f(x,1)
$$

です。

低fidelity $`s<1`$ は、このtargetを効率よく推定するために利用します。

この視点を失うと、単に安いfidelityばかり評価する戦略になりかねません。

---

## 6.8 Fidelityと評価コスト

通常、fidelityが高いほど評価コストも高くなります。

評価コストを

$$
c(x,s)
$$

とします。

単純にはfidelityだけに依存して

$$
c(s)
$$

としても構いません。

例えば

```text
s = 0.2 → cost = 1
s = 0.5 → cost = 5
s = 1.0 → cost = 50
```

です。

MFBOでは評価価値とコストの両方を考えることが重要です。ただし、常に単純な
「情報量 ÷ コスト」を使うわけではありません。どのutilityを使うかは
acquisition設計によって異なります。

---

## 6.9 Cost-Aware Bayesian Optimization

候補 $`(x,s)`$ の価値を $`V(x,s)`$ とすると、直感的には

$$
\frac{V(x,s)}{c(x,s)}
$$

のように、単位コスト当たりの価値を評価できます。

```text
高fidelity
  ├─ 情報量: 大
  └─ cost  : 大

低fidelity
  ├─ 情報量: 小〜中
  └─ cost  : 小
```

重要なのは「最も安いfidelity」でも「最も正確なfidelity」でもなく、**現在のposteriorに対して最も費用対効果の高い評価**を選ぶことです。

BoTorchではcost-aware utilityをacquisition設計へ組み込むための部品が用意されています。

---

## 6.10 なぜ通常のEIだけでは不十分なのか

通常のEIは、基本的に候補を評価したときの目的値改善を考えます。

しかしMFBOでは、low fidelityを評価しても、それ自体はtarget fidelityでのbest valueを直接更新しない場合があります。

low fidelityの価値は

> low fidelityを観測した結果、target fidelityについての意思決定がどれだけ改善されるか

にあります。

このためMulti-Fidelityでは **Value of Information (VoI)** の考え方が特に重要になります。

---

## 6.11 Knowledge Gradient

Knowledge Gradient (KG) は、ある候補を観測することで「観測後の最適意思決定」がどれだけ改善されるかを評価する考え方です。

現在のposterior meanに基づく最良値を

$$
V_n=\max_x \mu_n(x,s_{target})
$$

とします。

候補 $`(x,s)`$ を観測した後のposteriorを $`\mu_{n+1}`$ とすると、KGは概念的に

$$
\operatorname{KG}(x,s)
=
\mathbb{E}_n
\left[
\max_{x'}\mu_{n+1}(x',s_{target})
\right]
-
\max_{x'}\mu_n(x',s_{target})
$$

です。

つまり、

```text
今の最良判断
    ↓
(x, s)を仮に観測
    ↓
posterior更新
    ↓
target fidelityでの最良判断
    ↓
どれだけ改善したか
```

を測ります。

このようにKGは、非target fidelityの観測を「将来のtarget上の意思決定改善」で
評価できるため、Multi-Fidelityへ拡張しやすい考え方です。

---

## 6.12 Multi-Fidelity Knowledge Gradient

Multi-Fidelity KGでは、target fidelity以外の観測も「targetでの意思決定改善」によって価値を持ちます。

さらにcost-awareにすれば、概念的には

$$
\alpha(x,s)
\approx
\frac{\operatorname{VoI}(x,s)}{c(x,s)}
$$

として評価できます。

これにより、

```text
序盤
  └─ 安価なlow / medium fidelityを多めに利用

重要領域が絞られる
  └─ high fidelityを増やす

終盤
  └─ target fidelityで最終確認
```

のような挙動を期待できます。

ただし実際の選択はposterior・相関・cost modelに依存し、必ずこの順になるわけではありません。

---

## 6.13 SingleTaskMultiFidelityGP

robotorchanではBoTorchのMulti-Fidelity GPを共通wrapper APIで利用できます。

```python
from robotorchan.models import SingleTaskMultiFidelityGP
```

例えば入力を

```text
0: temperature
1: pressure
2: fidelity
```

とすると、fidelity次元を指定してモデルを構築できます。

```python
model = SingleTaskMultiFidelityGP(
    train_X=train_X,
    train_Y=train_Y,
    data_fidelities=[2],
)
```

robotorchanのwrapperはBoTorchの予測挙動を維持しつつ、raw training data保持や
`make_mll()` などの共通規約を追加します。

---

### モデルがfidelityを理解しても、評価戦略までは決まらない

`SingleTaskMultiFidelityGP` が担うのは、異なるfidelity間で情報共有できる
posteriorを構築することです。それだけで

~~~text
low fidelityを何回使うか
いつtarget fidelityへ移るか
残りbudgetをどう配分するか
~~~

まで自動的に決まるわけではありません。

これらは acquisition、cost-aware utility、candidate optimization、停止条件などの
意思決定層で扱います。**Multi-Fidelity model と MFBO policy は同義ではありません。**

## 6.14 iteration_fidelity と data_fidelities

`SingleTaskMultiFidelityGP`では、fidelityの意味に応じて指定方法が分かれます。

### iteration_fidelity

反復回数や学習epochのように、iterationに対応するfidelityです。

概念例:

```text
iteration = 10
iteration = 100
iteration = 1000
```

### data_fidelities

データ量、解像度、物理モデル精度などのfidelityを入力次元として指定します。

概念例:

```text
mesh resolution
training data fraction
simulation accuracy
```

robotorchanのwrapperはBoTorchのconstructorに合わせて

```python
iteration_fidelity: int | None
data_fidelities: Sequence[int] | None
```

を受け取ります。

---

## 6.14.1 Mixed × Multi-Fidelity の現在の実装境界

現在の robotorchan には `MixedSingleTaskMultiFidelityGP` があります。

このモデルでは categorical design feature と fidelity feature を同じ役割として扱いません。
`cat_dims` は design-side categorical feature だけを表し、
`iteration_fidelity` / `data_fidelities` と重複すると validation error になります。

covariance も、

- 非 fidelity design dimensions: continuous / categorical mixed covariance
- fidelity dimensions: BoTorch の fidelity covariance

という責務分離です。

したがって「Mixed GP に fidelity 列を追加しただけ」ではありません。
fidelity dimension は構造列として保持され、categorical kernel の対象外です。

---

## 6.15 Fidelityを入力に含める意味

例えば通常の設計変数が2次元なら

$$
x=(x_1,x_2)
$$

ですが、fidelityを追加すると

$$
\tilde{x}=(x_1,x_2,s)
$$

になります。

ここで $`s`$ は単なる追加特徴量ではなく、**観測精度・情報源の構造を表す特殊な入力**です。

したがって、通常の `SingleTaskGP` へfidelity列を何も考えず追加することと、
`SingleTaskMultiFidelityGP` でfidelity構造を明示することは同じではありません。

---

## 6.16 Continuous Fidelity

Fidelityが連続値なら

$$
s\in[0,1]
$$

として扱えます。

例えば

```text
0.1 → very low fidelity
0.4 → low-medium fidelity
0.7 → high fidelity
1.0 → target fidelity
```

です。

この場合、候補生成では設計変数とfidelityを連続空間上で最適化する構成が可能です。

ただし、最終目的がtarget fidelityであることをacquisition側で正しく反映する必要があります。

---

## 6.17 Discrete Fidelity

実際の評価方法が

```text
cheap simulator
expensive simulator
experiment
```

の3種類しかない場合、fidelityは離散的です。

このとき、optimizerが存在しない中間fidelityを生成してはいけません。

例えば

```text
s ∈ {0.2, 0.6, 1.0}
```

だけが実行可能なら、その値に限定してacquisition optimizationを行います。

少数の離散fidelityなら、前章で説明したmixed / discrete optimizationと同様に、
fidelity値を固定した複数問題として候補探索する考え方が利用できます。

ただし、**fidelity列をcategorical design featureへ変えるという意味ではありません**。
モデル内部ではfidelityとして扱いながら、optimizer側で実行可能な離散値だけを
候補にする、という責務分離です。

つまり

```text
Multi-Fidelity model
        ↓
Acquisition Function
        ↓
continuous fidelity ? → continuous optimization
        ↓ No
 discrete fidelity     → mixed / discrete optimization
```

となります。

---

## 6.18 ModelとAcquisition Optimizationは別問題

Mixed Variablesと同様、Multi-Fidelityでもモデルとoptimizerの責務を分離することが重要です。

`SingleTaskMultiFidelityGP`は

```text
(x, fidelity)
      ↓
posterior
```

を定義します。

一方、

```text
どのxを選ぶか
どのfidelityを選ぶか
```

を決めるのはacquisition functionとacquisition optimizerです。

したがって

```text
SingleTaskMultiFidelityGP
        ↓
Posterior across fidelities
        ↓
MF acquisition / cost-aware utility
        ↓
Acquisition optimization
        ↓
(x_next, fidelity_next)
```

という構造で理解すると整理しやすくなります。

---

## 6.19 Projection to Target Fidelity

MFBOでは、現在の候補 $`(x,s)`$ を評価した価値を考える一方で、最終的な意思決定はtarget fidelityで行います。

そのため、入力をtarget fidelityへ射影する操作を考えることがあります。

概念的には

$$
P(x,s)=(x,s_{target})
$$

です。

例えば

```text
candidate   = [temperature, pressure, 0.3]
projection  = [temperature, pressure, 1.0]
```

とします。

この「観測するfidelity」と「最終評価するfidelity」の区別がMulti-Fidelity acquisitionの重要なポイントです。

---

## 6.20 Current Value

KG系の獲得関数では、現在の情報だけでtarget fidelity上の最適判断をしたときの価値を基準にします。

$$
\max_x \mu_n(x,s_{target})
$$

を求め、その後の仮想観測による改善量を評価します。

したがってMF-KGは通常の1-step improvementより計算量が大きくなります。

これは

```text
候補を評価
```

だけでなく

```text
候補を評価したらposteriorがどう変わるか
そのposteriorでtarget上の最良点はどこになるか
```

まで内部的に考えるためです。

---

## 6.21 Cost Model

評価コストが既知なら、fidelityからcostを計算する関数を利用できます。

例えば

$$
c(s)=a+bs^p
$$

のようなモデルです。

一方、実際の実験時間が設計条件にも依存するなら

$$
c(x,s)
$$

を学習することも考えられます。

```text
cost既知
  └─ deterministic cost model

cost不明
  └─ observed runtime / priceからcost modelを学習
```

目的値surrogateとcost modelは別のモデルとして設計できます。

---

## 6.22 Cost-Aware Utilityの注意点

「acquisition valueをcostで割ればよい」とだけ考えると危険です。

例えばcostが極端に小さいlow fidelityが、target fidelityについてほとんど情報を持たない場合があります。

```text
very cheap
+
almost no information
=
必ずしも有用ではない
```

したがって、

- targetへの情報量
- fidelity間相関
- cost
- 残りbudget

を合わせて考える必要があります。

---

## 6.23 Budgetという考え方

MFBOでは「何回評価できるか」より「総コストをいくら使えるか」の方が自然な場合があります。

$$
\sum_{i=1}^{n} c(x_i,s_i)\le B
$$

ここで $`B`$ は総budgetです。

例えば

```text
low fidelity  = 1 unit
high fidelity = 20 units
budget        = 100 units
```

なら、high fidelityだけなら5回ですが、low fidelityを組み合わせればより多くの情報を取得できます。

MFBOの性能比較でも、iteration数だけでなく累積costに対するregretやbest target valueを見ることが重要です。

---

## 6.24 Multi-Fidelityが有効な条件

Multi-Fidelityは常に有効ではありません。

特に有効なのは、

- target fidelityが高コスト
- low fidelityが十分安い
- fidelity間に強い情報共有がある
- low fidelityからtargetの傾向を推測できる
- 評価cost差が大きい

場合です。

逆に

```text
low fidelityがあまり安くない
low fidelityとtargetが無相関
low fidelityのbiasが複雑すぎる
high fidelity自体が十分安い
```

場合は、Single-Fidelity BOの方が単純で効率的なことがあります。

---

## 6.25 Multi-FidelityとMulti-Taskの違い

Multi-FidelityとMulti-Taskは似ています。

どちらも複数の関連情報源を利用するからです。

しかし目的が異なります。

### Multi-Fidelity

```text
low fidelity
high fidelity
```

には精度やcostの順序があり、通常は明確なtarget fidelityがあります。

### Multi-Task

```text
material A
material B
material C
```

のように複数task間で情報共有しますが、必ずしも「AよりBが高精度」という関係ではありません。

判断のポイントは

> **情報源に精度・コストの階層と最終targetがあるか**

です。

---

## 6.26 Multi-FidelityとMixed Variablesの組合せ

実際の問題では

```text
temperature : continuous
catalyst    : categorical
fidelity    : continuous / discrete
```

のようにMixed + Multi-Fidelityになることがあります。

この場合、

```text
モデル側
  └─ categorical structure + fidelity structure

optimizer側
  └─ categorical constraints + fidelity constraints
```

の両方を考える必要があります。

robotorchanには `MixedSingleTaskMultiFidelityGP` があり、categorical design featureと
fidelity featureを分離して扱います。`cat_dims` はdesign-side categoryを指し、
`iteration_fidelity` / `data_fidelities` と同じ列を指定することはできません。

したがってMixed × Multi-Fidelityでも

```text
surrogate  : category構造 + fidelity構造
acquisition: targetへの価値 + 必要ならcost
optimizer  : category値 + 実行可能fidelity値を守る
```

という責務分離が重要です。

---

## 6.27 Multi-FidelityとMulti-Objectiveの違い

Fidelityはobjectiveではありません。

例えば

```text
強度を最大化
コストを最小化
```

はmulti-objectiveです。

一方

```text
簡易simulation
高精度simulation
実験
```

は同じ目的値を異なるfidelityで評価しています。

したがって

```text
Multi-Objective
  └─ 最適化したい目的が複数

Multi-Fidelity
  └─ 同じtargetを評価する情報源の精度・costが複数
```

と区別します。

---

## 6.28 robotorchanでの基本例

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import SingleTaskMultiFidelityGP

torch.set_default_dtype(torch.double)

# [x1, x2, fidelity]
train_X = torch.tensor(
    [
        [0.10, 0.20, 0.25],
        [0.30, 0.60, 0.25],
        [0.70, 0.40, 0.50],
        [0.20, 0.80, 1.00],
        [0.80, 0.70, 1.00],
    ]
)
train_Y = torch.tensor([[0.20], [0.35], [0.55], [0.45], [0.80]])

model = SingleTaskMultiFidelityGP(
    train_X=train_X,
    train_Y=train_Y,
    data_fidelities=[2],
)

mll = model.make_mll()
fit_gpytorch_mll(mll)
```

この例では3列目がfidelityです。

モデル学習後は、目的に応じてMulti-Fidelity acquisition、cost model、
target fidelityへのprojection、acquisition optimizerを組み合わせます。

---

## 6.29 Raw Dataと共通API

robotorchan wrapperでは他のsupervised modelと同様に

```python
model.raw_train_X
model.raw_train_Y
model.raw_train_Yvar
model.raw_data
model.make_mll()
```

を利用できます。

ただし`raw_*`はwrapper構築時のprovenanceであり、`condition_on_observations()`後の現在状態そのものではありません。

---

## 6.30 実務上の選択目安

| 状況 | 基本方針 |
|---|---|
| 評価方法が1種類 | Single-Fidelity BO |
| 安価な近似評価 + 高価なtarget評価 | Multi-Fidelityを検討 |
| fidelityが連続 | continuous fidelityとして設計 |
| fidelityが少数の離散水準 | discrete / mixed optimizationを検討 |
| fidelity間の相関が弱い | MF化の効果を慎重に検証 |
| cost差が大きい | cost-aware acquisitionが重要 |
| target fidelityが明確 | targetへのprojection / VoIを意識 |
| 複数情報源だが精度順序がない | Multi-Taskも検討 |

---

### MFBOを使う前に確認する4つの質問

実装を選ぶ前に、少なくとも次を確認します。

1. **target fidelityは何か** — 最終的にどの評価方法の最適値が欲しいか。
2. **low fidelityはtargetへ情報を与えるか** — 安いだけでは不十分です。
3. **cost差は十分あるか** — MF化の複雑さに見合う節約余地があるか。
4. **実行可能なfidelity値は何か** — continuousか、有限個のdiscrete値か。

この4点が曖昧なまま `SingleTaskMultiFidelityGP` を選んでも、
MFBOとしての意思決定問題は定義できません。

## 6.31 よくある誤解

### 「低fidelityをたくさん取れば必ず得」

低fidelityがtargetについて情報を持たなければ有効ではありません。

### 「fidelityは普通のカテゴリ変数と同じ」

精度・cost・targetへの関係を持つ点が異なります。

### 「SingleTaskMultiFidelityGPを使えばcost-awareになる」

モデルはposteriorを定義するものです。cost-awareな意思決定はacquisition側で設計します。

### 「Multi-FidelityはMulti-Objectiveである」

異なります。Fidelityは評価方法の忠実度であり、objectiveそのものではありません。

### 「常にhigh fidelityを選ぶのが安全」

正確ですが、限られたbudgetでは探索効率が悪くなる可能性があります。

---

## 6.32 この章で覚えておくこと

Multi-Fidelity BOの中心は、

> **安い情報源を利用して、target fidelityでの最適解を少ない総コストで見つける**

ことです。

構造を整理すると

```text
Design x + Fidelity s
        ↓
SingleTaskMultiFidelityGP
        ↓
Posterior across fidelities
        ↓
Target-fidelity value of information
        +
Evaluation cost
        ↓
Multi-Fidelity acquisition
        ↓
Acquisition optimization
        ↓
(x_next, s_next)
```

となります。

特に重要なのは、

1. fidelity間に情報共有があること
2. target fidelityを明確にすること
3. modelとacquisitionを分けて考えること
4. evaluation costを無視しないこと
5. discrete fidelityではoptimizer側でも実行可能値を守ること

です。

次章では、複数の関連taskや複数outputの情報共有を扱う [Multi-task / Multi-output](07_multitask_multioutput.md) を説明します。

## 参考文献

1. Kennedy, M. C., and O'Hagan, A. (2000).
   Predicting the output from a complex computer code when fast approximations are available.
   *Biometrika*, 87(1), 1-13.
2. Kandasamy, K., Dasarathy, G., Oliva, J. B., Schneider, J., and Póczos, B. (2017).
   Multi-fidelity Bayesian optimisation with continuous approximations.
   *Proceedings of the 34th International Conference on Machine Learning*.
3. BoTorch `SingleTaskMultiFidelityGP`, multi-fidelity, and cost-aware documentation.
4. [Acquisition Function](04_acquisition_function.md)
5. [Mixed Variables](05_mixed_variables.md)
