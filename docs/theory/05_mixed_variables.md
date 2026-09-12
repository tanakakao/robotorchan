# 5. Mixed Variables

## 5.1 Mixed Variables とは

ベイズ最適化（Bayesian Optimization; BO）の探索変数がすべて連続値とは限りません。

材料・製造条件の例では、

```text
温度      : 600 ～ 1000 ℃       連続
圧力      : 0.1 ～ 2.0 MPa      連続
触媒      : A / B / C            カテゴリ
製法      : hot / cold           カテゴリ
保持回数  : 1 / 2 / 3 / 4        整数
```

のように、異なる種類の変数が同じ探索空間に含まれます。

このような探索空間を **mixed search space** と呼びます。

単純な連続変数BOとの違いは、主に2か所に現れます。

```text
1. Surrogate model
   └─ 変数間の類似度をどうモデル化するか

2. Acquisition optimization
   └─ 有効なカテゴリ・離散値を守りながら次候補をどう探索するか
```

この2つは別の問題です。

Mixed BOを理解するときは、

> **Mixedなモデルを作ること** と **Mixedな探索空間で獲得関数を最適化すること**

を分けて考えることが重要です。

## 5.2 連続変数だけの場合

通常の連続変数GPでは、例えば

\[
x=(x_1,x_2)\in\mathbb{R}^2
\]

に対して、入力間距離をkernelで評価します。

RBF kernelなら

\[
k(x,x')
=
\sigma_f^2
\exp\left(
-\frac{1}{2}
\sum_j
\frac{(x_j-x'_j)^2}{\ell_j^2}
\right)
\]

です。

このとき、

```text
x = (0.2, 0.5)
x' = (0.3, 0.5)
```

のような2点には自然な距離があります。

連続変数では、

```text
0.20 と 0.21 は近い
0.20 と 0.90 は遠い
```

という関係をそのままkernelへ反映できます。

## 5.3 カテゴリ変数では距離の意味が異なる

カテゴリ変数を

```text
触媒A = 0
触媒B = 1
触媒C = 2
```

と整数で符号化したとします。

ここで通常の連続kernelをそのまま使うと、

\[
|0-2|=2|0-1|
\]

なので、モデルは暗黙に

```text
A と C は
A と B の2倍遠い
```

と解釈してしまいます。

しかし、単なるカテゴリラベルにこの順序・距離関係はありません。

したがって、カテゴリ値を数値として保存することと、その数値差を連続距離としてモデル化することは別です。

## 5.4 Nominal と Ordinal を区別する

離散変数をすべて同じものとして扱うべきではありません。

### Nominal variable

順序を持たないカテゴリです。

```text
触媒 = A / B / C
装置 = Machine1 / Machine2 / Machine3
材料種 = X / Y / Z
```

`A < B < C` のような意味はありません。

### Ordinal variable

順序を持つカテゴリです。

```text
評価 = low / medium / high
粒度 = coarse / medium / fine
```

順序はありますが、隣接カテゴリ間の距離が等しいとは限りません。

### Integer variable

整数値です。

```text
積層数 = 1, 2, 3, 4, 5
繰返し回数 = 1, ..., 10
```

整数はカテゴリと異なり、通常は数値的な順序と距離に意味があります。

したがって

```text
categorical
ordinal
integer
```

は、探索空間を設計するときに区別する必要があります。

## 5.5 Categorical kernel の基本的な考え方

カテゴリ変数では、カテゴリが一致するかどうかを使った共分散を考えることができます。

最も単純な直感は

\[
k_{cat}(c,c')
=
\begin{cases}
1 & c=c'\\
\rho & c\neq c'
\end{cases}
\]

です。

ここで `ρ` は異なるカテゴリ間の相関を表します。

重要なのは、カテゴリ番号そのものの差

\[
|c-c'|
\]

を距離として使っていないことです。

例えば

```text
A = 0
B = 1
C = 2
```

と保存していても、`0` と `2` が `0` と `1` より必ず遠いとは仮定しません。

## 5.6 Continuous kernel と Categorical kernel

入力を

\[
x=(x_{cont},x_{cat})
\]

と分けて考えます。

例えば

```text
x_cont = [温度, 圧力]
x_cat  = [触媒]
```

です。

連続部分には

\[
k_{cont}(x_{cont},x'_{cont})
\]

カテゴリ部分には

\[
k_{cat}(x_{cat},x'_{cat})
\]

を使います。

Mixed GPでは、これらを組み合わせて共分散を構成します。

概念的には

```text
入力
 ├─ continuous dimensions
 │    └─ Matérn / RBFなど
 │
 └─ categorical dimensions
      └─ categorical covariance

             ↓

       Mixed covariance
```

と理解できます。

## 5.7 なぜ単純なOne-Hotだけでは十分でないのか

カテゴリ変数の一般的な前処理としてOne-Hot Encodingがあります。

例えば

```text
A → [1, 0, 0]
B → [0, 1, 0]
C → [0, 0, 1]
```

です。

予測モデルによっては非常に有効ですが、BOでは注意が必要です。

One-Hotした入力を通常の連続探索として最適化すると、獲得関数optimizerが

```text
[0.35, 0.20, 0.45]
```

のような、本来存在しない中間点を候補として生成できてしまいます。

つまり問題は2つあります。

```text
モデル側
  └─ One-Hot空間の距離をどう解釈するか

optimizer側
  └─ one-hot制約をどう保証するか
```

One-Hot Encodingそのものが誤りなのではなく、**モデル化と候補最適化の両方を整合させる必要がある**ということです。

## 5.8 MixedSingleTaskGP

robotorchanでは、連続変数とカテゴリ変数を含むsingle-task回帰に

```python
from robotorchan.models import MixedSingleTaskGP
```

を利用できます。

例えば入力列を

```text
0: 温度       continuous
1: 触媒       categorical
2: 圧力       continuous
3: 製法       categorical
```

とすると、

```python
model = MixedSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[1, 3],
)
```

のようにカテゴリ次元を指定します。

`cat_dims` は **カテゴリ変数が存在する入力次元のindex** です。

robotorchanの`MixedSingleTaskGP`はBoTorchの同名モデルを薄くwrapし、予測挙動を維持しながらraw data保持や`make_mll()`などの共通APIを追加しています。

## 5.9 カテゴリ値の表現

`MixedSingleTaskGP`へ入力するとき、カテゴリは数値コードとしてtensorに格納できます。

例えば

```text
A → 0
B → 1
C → 2
```

です。

ここで重要なのは、

> 数値コードはtensorへ格納するための表現であり、連続値としての大小関係を意味しない

という点です。

`cat_dims`によって、その次元をカテゴリとしてモデルへ伝えます。

実務では、元ラベルと数値コードの対応表を保存しておくことも重要です。

## 5.10 Integer variable はどう扱うか

整数変数はカテゴリ変数と同じではありません。

例えば

```text
焼成回数 = 1, 2, 3, 4, 5
```

では、通常

```text
1 と 2 は近い
1 と 5 は遠い
```

という数値的意味があります。

したがって、整数変数を単純にcategorical dimensionへ入れると、その距離構造を捨てることになります。

一方、連続値としてモデル化して候補生成後に丸める方法では、

```text
surrogate model : continuous
candidate       : integerへ丸める
```

という近似になります。

この方法は簡単ですが、丸め前後でacquisition valueが変わるため、厳密なmixed optimizationとは異なります。

整数変数の取り扱いは

- 値域の大きさ
- 整数間の滑らかさを仮定できるか
- 候補数
- acquisition optimizer

を考慮して決める必要があります。

## 5.11 モデル化とAcquisition Optimizationは別問題

Mixed BOで最も重要な点です。

`MixedSingleTaskGP`を使えば、自動的にカテゴリ制約を守った候補が生成されるわけではありません。

モデルは

```text
X → posterior distribution
```

を定義します。

一方、次候補を選ぶのは

\[
x_{next}
=
\arg\max_{x\in\mathcal{X}}
\alpha(x)
\]

というacquisition optimizationです。

したがって

```text
MixedSingleTaskGP
       ↓
Mixedなposterior
       ↓
Acquisition Function
       ↓
Mixed search spaceを守るoptimizer
       ↓
Candidate
```

という構造になります。

## 5.12 optimize_acqf_mixed の考え方

カテゴリや離散変数の組合せが有限で列挙可能な場合、BoTorchでは`optimize_acqf_mixed`を利用できます。

基本的な考え方は、離散部分を固定した複数の探索問題へ分けることです。

例えば

```text
触媒 = A / B / C
```

なら、

```text
触媒=A に固定して連続変数を最適化
触媒=B に固定して連続変数を最適化
触媒=C に固定して連続変数を最適化
```

し、それぞれの最良acquisition valueを比較します。

概念的には

\[
\max_{c\in\mathcal{C}}
\max_{x_{cont}}
\alpha(x_{cont},c)
\]

です。

これにより、カテゴリを連続値として補間せず、有効なカテゴリ値だけを評価できます。

## 5.13 fixed_features_list

`optimize_acqf_mixed`では、離散設定を`fixed_features_list`として表現します。

例えば入力が

```text
0: 温度
1: 触媒
2: 圧力
```

で、触媒コードが`0, 1, 2`なら概念的には

```python
fixed_features_list = [
    {1: 0.0},
    {1: 1.0},
    {1: 2.0},
]
```

とします。

各dictionaryは、探索中に固定する入力次元と値を表します。

カテゴリが複数ある場合は組合せを作ります。

例えば

```text
触媒 = A / B / C
製法 = hot / cold
```

なら最大で

\[
3\times2=6
\]

個のカテゴリ組合せがあります。

## 5.14 組合せ爆発

`optimize_acqf_mixed`の列挙方式は、カテゴリ組合せが少ない場合には明快です。

しかし、カテゴリ数や水準数が増えると

\[
N_{comb}
=
\prod_{j=1}^{p}L_j
\]

だけ組合せが増えます。

例えば5つのカテゴリ変数が各5水準なら

\[
5^5=3125
\]

組です。

この場合、すべての組合せについて連続最適化を行うのは高コストです。

したがってMixed BOでは、

```text
カテゴリ組合せが少ない
  └─ 列挙 + continuous optimization

カテゴリ組合せが多い
  └─ 離散探索、候補集合、heuristic、combinatorial methodなどを検討
```

という切り替えが必要になります。

## 5.15 Candidate set から選ぶ方法

実験可能な条件があらかじめ有限個に限定されている場合、連続最適化そのものが不要なことがあります。

候補集合を

\[
\mathcal{C}=\{x_1,\ldots,x_M\}
\]

とすると、

\[
x_{next}
=
\arg\max_{x\in\mathcal{C}}
\alpha(x)
\]

とすればよいからです。

これは材料候補、既存レシピ、設備制約付き条件などで非常に実用的です。

```text
全候補
  ↓
posterior
  ↓
acquisition value
  ↓
最大値の候補を選択
```

候補集合方式では、カテゴリ・整数・複雑な実行可能性を候補生成段階で保証できる利点があります。

## 5.16 Mixed model と候補集合方式は両立する

候補集合から選ぶ場合でも、surrogate model側でカテゴリ構造を適切に扱うことには意味があります。

例えば

```text
MixedSingleTaskGP
      ↓
有限候補集合にposteriorを計算
      ↓
各候補のacquisition value
      ↓
上位候補を選択
```

とできます。

したがって

```text
MixedSingleTaskGP = モデル
optimize_acqf_mixed = optimizerの一方式
candidate set = optimizerの別方式
```

であり、同一概念ではありません。

## 5.17 Batch BO との組合せ

一度に複数条件を実験する場合は、Mixed search spaceでもbatch BOが必要になります。

単純にacquisition value上位`q`点を独立に選ぶと、似た候補が集中する場合があります。

q-acquisitionでは候補集合

\[
X_q=\{x_1,\ldots,x_q\}
\]

を共同評価します。

Mixed search spaceでは

```text
カテゴリ制約
+
連続変数制約
+
batch内の相互作用
```

を同時に考える必要があるため、single candidateより最適化が難しくなります。

候補集合が有限なら、離散候補からbatchを選ぶ方法も有力です。

## 5.18 条件付き変数との違い

Mixed variableとhierarchical / conditional variableは似ていますが、別の問題です。

通常のMixed search spaceでは、すべての変数が常に意味を持つと考えます。

```text
触媒 = A / B / C
温度 = 500 ～ 1000
圧力 = 0.1 ～ 2.0
```

では、どの触媒でも温度と圧力が定義されています。

一方、

```text
製法 = A
  ├─ 温度
  └─ 時間

製法 = B
  ├─ 圧力
  └─ 流量
```

のように、親カテゴリによって有効な変数自体が変わる場合はhierarchical search spaceです。

robotorchanでは、このような構造に

- `HierarchicalConditionalKernelGP`
- `HierarchicalConditionalKernelMultiTaskGP`

を用意しています。

## 5.19 Multi-taskとの違い

カテゴリ変数とtask indexも混同しやすい概念です。

例えば

```text
触媒A / B / C
```

が「設計変数」であり、どの触媒を選ぶか自体を最適化したいならcategorical variableです。

一方、

```text
実験装置1 / 実験装置2
シミュレーション / 実験
材料系A / 材料系B
```

のようなデータソース間で情報共有したい場合はtaskとして扱う設計が候補になります。

判断基準は

> そのラベルを探索変数として最適化したいのか、それとも関連データ源として利用したいのか

です。

## 5.20 Multi-Fidelityとの違い

低精度・高精度シミュレーションなどのfidelityも離散値を取ることがあります。

しかし、単なるカテゴリと異なりfidelityには通常

```text
精度
計算コスト
情報量
```

の構造があります。

例えば

```text
low fidelity
medium fidelity
high fidelity
```

を単純な3カテゴリとして扱うと、fidelity間の関係やコスト構造を十分利用できません。

この場合は`SingleTaskMultiFidelityGP`などのMulti-Fidelityモデルを検討します。

## 5.21 Mixed BOでの制約

実務では、カテゴリの組合せによって実行不可能な条件が存在することがあります。

例えば

```text
材料=A かつ 製法=C は不可
装置=1 では 温度>900℃ は不可
```

です。

このような制約は、単純なbox boundsだけでは表現できません。

対処方法には、

- `fixed_features_list`から無効組合せを除外する
- 候補集合を事前に実行可能条件だけへ絞る
- 制約付きacquisitionを使う
- 条件付き探索空間としてモデル化する

などがあります。

「物理的に生成不可能な候補」と「生成できるが性能制約を満たすか不明な候補」は区別することが重要です。

前者はsearch spaceの制約、後者はconstrained BOの対象です。

## 5.22 入力変換の注意点

Mixed inputでは、すべての列を同じ方法で正規化してはいけません。

連続変数は

\[
[0,1]
\]

へ正規化することが有効ですが、カテゴリコードを連続値と同じ意味で正規化すると、カテゴリ表現との整合性に注意が必要です。

概念的には

```text
continuous dimensions
  └─ scale / normalize

categorical dimensions
  └─ category identityを保持
```

と考えます。

前処理・input transformを導入するときは、カテゴリ次元がどのように変換されるかを必ず確認します。

## 5.23 Mixed BOの典型フロー

連続 + カテゴリ変数のBOは、概念的には次のようになります。

```text
1. データ準備
   ↓
2. continuous / categorical dimensionsを定義
   ↓
3. MixedSingleTaskGPを学習
   ↓
4. posteriorを構築
   ↓
5. acquisition functionを構築
   ↓
6. mixed search spaceでacquisitionを最適化
   ↓
7. 有効なcandidateを取得
   ↓
8. 実験・シミュレーション
   ↓
9. データ追加
   ↓
10. 再学習
```

重要なのは、Step 3とStep 6が別々の責務であることです。

## 5.24 robotorchanでの基本例

モデル構築は通常のwrapperと同じ流れです。

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import MixedSingleTaskGP

torch.set_default_dtype(torch.double)

# [温度, 触媒コード, 圧力]
train_X = torch.tensor(
    [
        [0.20, 0.0, 0.40],
        [0.50, 1.0, 0.30],
        [0.80, 2.0, 0.70],
        [0.35, 0.0, 0.80],
        [0.65, 1.0, 0.55],
    ]
)
train_Y = torch.tensor([[0.30], [0.55], [0.40], [0.45], [0.70]])

model = MixedSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[1],
)

mll = model.make_mll()
fit_gpytorch_mll(mll)
```

ここで`cat_dims=[1]`により、2列目をカテゴリ次元として指定しています。

候補生成では、このモデルとは別にmixed search spaceに適したacquisition optimizationを選択します。

## 5.25 実務上の選択目安

| 探索空間 | 基本方針 |
|---|---|
| 連続変数のみ | `SingleTaskGP` + continuous acquisition optimization |
| 連続 + 少数カテゴリ | `MixedSingleTaskGP` + mixed optimization |
| 連続 + 多数のカテゴリ組合せ | Mixed model + candidate/discrete/combinatorial searchを検討 |
| 有限候補からのみ選択 | 候補集合上でacquisitionを評価 |
| 整数変数 | 数値構造を保つかcategorical化するかを問題ごとに判断 |
| 親カテゴリで有効変数が変化 | Hierarchical / conditional model |
| 離散fidelity | Multi-Fidelityとしての構造を優先して検討 |

## 5.26 よくある誤解

### 「カテゴリを0, 1, 2にすれば通常GPでよい」

単なるラベルなら、数値差に意味がないため適切ではありません。

### 「MixedSingleTaskGPを使えば候補も自動的にカテゴリになる」

モデルとacquisition optimizerは別です。有効な探索空間をoptimizer側でも保証する必要があります。

### 「整数はすべてカテゴリとして扱えばよい」

整数間の距離や順序に意味がある場合、その情報を失う可能性があります。

### 「One-Hot EncodingならMixed BOを考えなくてよい」

候補最適化時にone-hot制約をどう守るかという問題が残ります。

### 「カテゴリとtaskは同じ」

カテゴリは通常、最適化対象の設計変数です。taskは関連するデータ源・出力間で情報共有するための構造です。

## 5.27 まとめ

Mixed Variablesを扱うときは、次の3層を分けて考えると整理しやすくなります。

```text
Search Space
  └─ continuous / categorical / integer / conditional

Surrogate Model
  └─ それぞれの変数に適したcovariance structure

Acquisition Optimization
  └─ 有効な候補だけを探索
```

robotorchanでは、連続 + カテゴリ変数のsurrogateとして`MixedSingleTaskGP`を利用できます。

一方、候補生成は別の責務であり、カテゴリ組合せが少なければ`optimize_acqf_mixed`のような列挙ベースの方法、候補が有限ならcandidate set上での評価などを使い分けます。

この責務分離は、今後robotorchanでacquisition optimizerや候補選択機能を拡張するときにも重要な設計原則になります。

次章では、評価精度と評価コストが異なる複数の情報源を利用する [Multi-Fidelity](06_multi_fidelity.md) を説明します。

## 参考

- BoTorch v0.18.1 `MixedSingleTaskGP`
- BoTorch v0.18.1 `optimize_acqf_mixed`
- [Kernel](03_kernel.md)
- [Acquisition Function](04_acquisition_function.md)
