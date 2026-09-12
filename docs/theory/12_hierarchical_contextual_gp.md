# 12. Hierarchical / Contextual GP

## 12.1 この章で扱うこと

これまでの章では、

- 通常の連続入力
- カテゴリ変数
- Multi-Fidelity
- Multi-task
- 高次元
- Structured Output

などを扱ってきました。

本章では、さらに **入力空間そのものに構造がある問題** を扱います。

robotorchan では、主に次の3系統を対象とします。

1. **Hierarchical search space**
   - `HierarchicalConditionalKernelGP`
   - `HierarchicalConditionalKernelMultiTaskGP`
2. **Heterogeneous Multi-task**
   - `HeterogeneousMTGP`
3. **Contextual GP**
   - `SACGP`
   - `LCEAGP`
   - `LCEMGP`

これらは一見似ていますが、解こうとしている問題は異なります。

```text
Hierarchical
  → 親変数の値によって有効な子変数が変わる

Heterogeneous Multi-task
  → タスクごとに観測される特徴量集合が違う

Contextual
  → 入力を複数contextへ分解して構造を利用する
```

本章では、単なるAPI一覧ではなく、**何を仮定しているモデルなのか**を中心に整理します。

---

## 12.2 通常のGPが暗黙に仮定していること

通常のGPでは、入力

\[
x=(x_1,\dots,x_d)
\]

のすべての次元が、すべての候補で同じ意味を持つと考えます。

例えば RBF / Matérn kernel なら、概念的に

\[
k(x,x')
=
k(\|x-x'\|)
\]

のように、同じ入力次元同士の距離を計算します。

しかし実務では、

```text
装置方式Aでは温度だけ設定できる
装置方式Bでは圧力だけ設定できる
```

のように、**候補によって意味のある変数が変わる**ことがあります。

この場合、通常の距離計算では「無効な変数」まで比較してしまいます。

---

## 12.3 Hierarchical search space

階層探索空間では、ある変数の値が別の変数の有効 / 無効を決めます。

例えば

```text
process_type
 ├─ 0 → temperature が有効
 └─ 1 → pressure が有効
```

という構造です。

入力を

```text
[x, process_type, temperature, pressure]
```

とすると、

- `process_type == 0` では `temperature` が有効
- `process_type == 1` では `pressure` が有効

です。

この構造を無視すると、方式0同士を比較するときにも pressure の差がkernel距離へ入ってしまいます。

---

## 12.4 Hierarchical Conditional Kernel の考え方

Hierarchical Conditional Kernel は、比較する2点で**共通して有効な特徴量だけを利用する**ようにkernelを構成します。

単純化すると、

```text
候補A: process_type = 0
  → x, process_type, temperature が有効

候補B: process_type = 0
  → x, process_type, temperature が有効
```

なら、temperatureまで比較します。

一方、

```text
候補A: process_type = 0
候補B: process_type = 1
```

なら、branch固有変数 temperature / pressure をそのまま比較せず、共有された階層部分に基づいて共分散を作ります。

BoTorch の `HierarchicalConditionalKernel` は、この種の階層search space用kernelとして実装されています。

---

## 12.5 hierarchical dependency の表現

robotorchan / BoTorch では、階層関係を

```python
hierarchical_dependencies
```

で指定します。

形式は

```python
{
    parent_index: {
        parent_value: [active_child_indices],
    }
}
```

です。

例えば

```python
hierarchical_dependencies = {
    1: {
        0: [2],
        1: [3],
    }
}
```

なら

```text
feature 1 == 0
  → feature 2 が有効

feature 1 == 1
  → feature 3 が有効
```

です。

Notebookでは、

```text
[x, process_type, temperature, pressure]
```

に対してこの表現を使っています。

---

## 12.6 階層構造は単なるcategorical variableではない

`process_type` 自体はカテゴリ変数として扱えます。

しかし問題は、それだけではありません。

```text
Mixed GP
  → process_type がカテゴリである

Hierarchical GP
  → process_type の値によって
     他の入力次元の意味・有効性が変わる
```

という違いがあります。

したがって、

```text
continuous + categorical
```

だけなら `MixedSingleTaskGP` が自然ですが、

```text
categorical choice
  ↓
active parameter set changes
```

なら hierarchical model を検討します。

---

## 12.7 `HierarchicalConditionalKernelGP`

robotorchan の単一タスク版は

```python
from robotorchan.models import HierarchicalConditionalKernelGP

model = HierarchicalConditionalKernelGP(
    train_X=train_X,
    train_Y=train_Y,
    hierarchical_dependencies=hierarchical_dependencies,
)
```

です。

wrapper は BoTorch の予測挙動を維持しながら、

```python
model.raw_train_X
model.raw_train_Y
model.raw_train_Yvar
model.make_mll()
```

を提供します。

---

## 12.8 constructor の主要引数

robotorchan wrapper では主に

```python
HierarchicalConditionalKernelGP(
    train_X,
    train_Y,
    hierarchical_dependencies,
    eval_hierarchical_features=True,
    separate_hierarchical_features=True,
    train_Yvar=None,
    use_saas_prior=True,
    use_outputscale=True,
    input_transform=None,
    outcome_transform=DEFAULT,
)
```

を公開します。

特に重要なのは

- `hierarchical_dependencies`
- `eval_hierarchical_features`
- `separate_hierarchical_features`
- `use_saas_prior`

です。

---

## 12.9 `eval_hierarchical_features`

このフラグは、階層構造を構成する親feature自体をkernelで評価するかどうかに関係します。

親変数は単なるrouting情報ではなく、異なるbranch間の類似性にも寄与し得ます。

一般には default から始め、

```text
親変数そのものの差を共分散へ入れたいか
```

という観点で調整します。

---

## 12.10 `separate_hierarchical_features`

この設定は、階層feature群をkernel構造上どのように分離して扱うかに関係します。

実務では、まず default を使い、

- branch間の相関が不自然
- hyperparameterが不安定
- hierarchyが非常に深い

場合に構造を確認します。

---

## 12.11 SAAS priorとの組み合わせ

Hierarchical Conditional Kernel GP は

```python
use_saas_prior=True
```

を選べます。

これは階層構造と**高次元 sparsity**を同時に扱いたい場合に有効です。

例えば

```text
方式選択
  ↓
方式ごとに20個の設定変数
  ↓
実際に効く変数は少数
```

という問題では、

- hierarchy
- sparse effective dimensions

の両方が存在します。

ただし、SAAS priorを使えば常に良いわけではありません。

低次元かつデータが十分ある場合は、通常priorの方が学習しやすい場合があります。

Notebookでは説明を簡潔にするため `use_saas_prior=False` の例を使っています。

---

## 12.12 Multi-task版

複数の関連タスクが同じ階層探索空間を共有する場合、

```python
HierarchicalConditionalKernelMultiTaskGP
```

を使えます。

概念的には

```text
HierarchicalConditionalKernel
  ×
Task covariance
```

です。

BoTorch実装では、data covariance に hierarchical conditional kernel を使い、task correlation は `MultiTaskGP` と同様の task covariance で扱います。

---

## 12.13 `HierarchicalConditionalKernelMultiTaskGP`

基本形は

```python
model = HierarchicalConditionalKernelMultiTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    task_feature=task_feature,
    hierarchical_dependencies=hierarchical_dependencies,
)
```

です。

`train_X` は long-format で、1列がtask featureです。

例えば

```text
[x, process_type, temperature, pressure, task]
```

です。

---

## 12.14 task featureとhierarchical featureのindex

ここは重要です。

Notebookでは、`task_feature` を除いた入力featureに対して

```python
hierarchical_dependencies
```

を定義しています。

したがって、task列を末尾へ追加した場合でも、hierarchical featureのindexは単一タスク版と同じ考え方で管理できます。

実装時には

```text
raw input column index
vs
hierarchical feature index
vs
task feature index
```

を混同しないことが重要です。

---

## 12.15 どんな場面でHierarchical GPを使うか

典型例は、

- 装置方式ごとに設定項目が違う
- 材料種ごとに有効な配合条件が違う
- アルゴリズム選択ごとにハイパーパラメータが違う
- 前処理方法により後段パラメータが変わる
- 製造route選択により工程条件が変わる

です。

つまり

```text
choice changes the search subspace
```

という問題です。

---

## 12.16 Heterogeneous Multi-taskとは

次に、階層構造とは別の問題を考えます。

タスク0では

```text
[x0, x1, x2]
```

が観測される一方、タスク1では

```text
[x0, x2]
```

しか観測できないとします。

これは「同じfeatureが条件付きでactiveになる」のではなく、**タスクごとに入力feature set自体が違う**問題です。

この場合に `HeterogeneousMTGP` を使います。

---

## 12.17 MultiTaskGPとの違い

通常の `MultiTaskGP` は、基本的に各タスクで同じcovariate spaceを共有します。

```text
Task 0: x1, x2, x3
Task 1: x1, x2, x3
```

なら通常MultiTaskGPで自然です。

一方、

```text
Task 0: x1, x2, x3
Task 1: x1, x3
Task 2: x2, x4
```

のように入力feature集合が異なるなら、HeterogeneousMTGPが候補になります。

---

## 12.18 `feature_indices`

`HeterogeneousMTGP` の中心となるのが

```python
feature_indices
```

です。

例えば完全なfeature空間を

```text
[x0, x1, x2]
```

とし、

```text
Task 0 → [x0, x1, x2]
Task 1 → [x0, x2]
```

なら

```python
feature_indices = [
    [0, 1, 2],
    [0, 2],
]
```

です。

さらに

```python
full_feature_dim=3
```

を指定します。

---

## 12.19 `train_Xs`はtask featureを持たない

robotorchan / BoTorch の `HeterogeneousMTGP` constructorでは、タスクごとの入力を

```python
train_Xs=[train_X0, train_X1, ...]
```

として渡します。

この時点では task ID 列を追加しません。

モデル内部でタスク構造を組み立てます。

これは通常の long-format `MultiTaskGP` とAPIが異なる点です。

---

## 12.20 robotorchan `HeterogeneousMTGP`

基本形は

```python
model = HeterogeneousMTGP(
    train_Xs=[train_X0, train_X1],
    train_Ys=[train_Y0, train_Y1],
    train_Yvars=None,
    feature_indices=[
        [0, 1, 2],
        [0, 2],
    ],
    full_feature_dim=3,
)
```

です。

robotorchanではtaskごとのraw dataを

```python
model.raw_train_Xs
model.raw_train_Ys
model.raw_train_Yvars
```

として保持します。

---

## 12.21 `HeterogeneousMTGP.make_mll()`

このwrapperは

```python
mll = model.make_mll()
```

で `ExactMarginalLogLikelihood` を返します。

したがって

```python
from botorch.fit import fit_gpytorch_mll

fit_gpytorch_mll(model.make_mll())
```

で学習できます。

---

## 12.22 目的タスク

現行Notebookでは **task 0 を目的タスク** として使っています。

補助タスクの情報を利用してtask 0の予測を改善する、transfer-learning的な使い方です。

posterior評価では、完全feature空間の入力にtask featureを追加した形式を使います。

例えば

```python
test_X_with_task = torch.cat(
    [test_X, torch.zeros(test_X.shape[0], 1)],
    dim=-1,
)
posterior = model.posterior(test_X_with_task)
```

です。

---

## 12.23 HeterogeneousMTGPが有効な実務例

例えば

```text
新ライン:
  temperature, pressure, flow, atmosphere

旧ライン:
  temperature, flow
```

のような場合です。

旧ラインの大量データを完全に捨てず、共有featureを通じて新ラインへ情報転移できます。

材料開発なら

```text
高精度測定:
  composition + process + structure

過去DB:
  composition + process only
```

のようなケースにも考え方を応用できます。

---

## 12.24 HierarchicalとHeterogeneousの違い

この2つは混同しやすいです。

```text
Hierarchical
  → 1つの探索空間の中で
     parent choiceによってactive featuresが変わる

Heterogeneous MTGP
  → taskごとに
     観測されるfeature spaceが異なる
```

です。

判断基準は

```text
違いの原因が「branch」なのか「task」なのか
```

です。

---

## 12.25 Contextual GPとは

次に contextual GP を考えます。

入力全体を複数のcontextへ分解できる問題があります。

例えば

```text
工程Aの条件: [a0, a1]
工程Bの条件: [b0, b1]
```

があり、最終scoreが

\[
y
=
f_A(a_0,a_1)
+f_B(b_0,b_1)
\]

のような加法構造を持つと考えます。

このとき、入力をcontext単位へ分解してkernelを設計できます。

---

## 12.26 `decomposition`

SACGP / LCEAGP では、contextごとの入力列を

```python
decomposition
```

で指定します。

例えば

```python
decomposition = {
    "context_A": [0, 1],
    "context_B": [2, 3],
}
```

です。

入力

```text
[a0, a1, b0, b1]
```

を

```text
context_A → [a0, a1]
context_B → [b0, b1]
```

へ分けています。

---

## 12.27 SACGP

`SACGP` は **Structural Additive Contextual GP** と考えると理解しやすいモデルです。

各contextの寄与を加法的に分けます。

概念的には

\[
f(x)
=
\sum_{c=1}^{C} f_c(x_c)
\]

です。

kernelも対応して

\[
k(x,x')
=
\sum_{c=1}^{C} k_c(x_c,x_c')
\]

のような構造を持ちます。

---

## 12.28 SACGPの仮定

SACGPは強い構造仮定を置きます。

```text
overall reward
  = context A contribution
  + context B contribution
  + ...
```

です。

この仮定が妥当なら、フラットな高次元GPよりsample efficiencyを改善できる可能性があります。

一方、context間interactionが強い場合、単純な加法分解では不足します。

---

## 12.29 robotorchan `SACGP`

基本形は

```python
model = SACGP(
    train_X=train_X,
    train_Y=train_Y,
    train_Yvar=None,
    decomposition=decomposition,
)
```

です。

robotorchan共通の

```python
model.raw_train_X
model.raw_train_Y
model.raw_train_Yvar
model.make_mll()
```

が使えます。

---

## 12.30 LCEAGP

`LCEAGP` は latent context embedding を利用する additive contextual GPです。

SACGPではcontextを明示的に分離して扱いますが、LCEAGPはcontextごとの潜在embeddingを通じて、context間の関係を学習できます。

概念的には

```text
context ID / context features
        ↓
latent embedding z_c
        ↓
context similarity
        ↓
shared GP structure
```

です。

---

## 12.31 なぜcontext embeddingが必要なのか

例えば10個の工程segmentがあるとします。

それらを完全に独立に扱うと情報共有できません。

一方、すべて同一とみなすのも不自然です。

latent embeddingを学習すると、

```text
context 1 と 2 は似ている
context 3 は大きく異なる
```

という関係をデータから表現できます。

---

## 12.32 LCEAGP constructor

robotorchanでは

```python
LCEAGP(
    train_X,
    train_Y,
    train_Yvar,
    decomposition,
    train_embedding=True,
    cat_feature_dict=None,
    embs_feature_dict=None,
    embs_dim_list=None,
    context_weight_dict=None,
)
```

を利用できます。

特に

```python
train_embedding=True
```

ならcontext embeddingを学習します。

---

## 12.33 categorical context features

contextの補助情報がカテゴリとして与えられる場合、

```python
cat_feature_dict
```

でcontext属性を表現できます。

例えば

```text
line type
reactor family
material family
```

のようなcontext descriptorです。

単なるcontext IDより、意味のあるmetadataを使うことでcontext関係を学びやすくなる場合があります。

---

## 12.34 continuous embedding features

contextに連続descriptorがある場合は

```python
embs_feature_dict
```

などを利用できます。

例えば

```text
設備容量
炉容積
最大温度
測定分解能
```

のような特徴です。

これは

```text
context = opaque ID
```

ではなく

```text
context = feature vector
```

として情報共有する考え方です。

---

## 12.35 SACGPとLCEAGPの違い

単純化すると

```text
SACGP
  → context decompositionは既知
  → additive structureを利用
  → context間のlatent similarityを積極的には学習しない

LCEAGP
  → decompositionは既知
  → additive structureを利用
  → context embeddingによりcontext間関係を学習
```

です。

context数が少なく、それぞれ独立と考えられるならSACGPで十分な場合があります。

context数が多く、類似context間でtransferしたいならLCEAGPが有力です。

---

## 12.36 aggregated rewardという前提

SACGP / LCEAGP の重要な特徴は、contextごとの個別rewardを必ずしも観測しないことです。

例えば

```text
工程Aの寄与: 未観測
工程Bの寄与: 未観測

最終製品score: 観測
```

という状況です。

モデルは最終rewardを利用しながら、context構造をkernelへ組み込みます。

---

## 12.37 LCEMGP

一方、contextごとの出力を個別に観測できる場合があります。

例えば

```text
context A reward = y_A
context B reward = y_B
```

をそれぞれ測定できる場合です。

この場合、contextをtaskとして扱うmulti-output model

```python
LCEMGP
```

が候補になります。

---

## 12.38 LCEMGPの考え方

LCEMGPでは、

```text
input x
  +
context/task ID
```

からcontext別outputを学習します。

さらにcontext feature / embeddingを使ってtask間関係を表現します。

したがって、通常の `MultiTaskGP` よりcontext metadataを積極的に使える点が特徴です。

---

## 12.39 long-format data

LCEMGPの入力はMultiTaskGPと同様にlong-formatです。

例えば

```text
[x0, x1, task]
```

で

```text
task 0 = context A
task 1 = context B
```

とします。

```python
X_a = torch.cat([base_X, torch.zeros(n, 1)], dim=-1)
X_b = torch.cat([base_X, torch.ones(n, 1)], dim=-1)

train_X = torch.cat([X_a, X_b], dim=0)
train_Y = torch.cat([Y_a, Y_b], dim=0)
```

です。

---

## 12.40 context feature

contextのカテゴリ特徴を

```python
context_cat_feature
```

として与えられます。

例えば

```python
context_cat_feature = torch.tensor(
    [
        [0.0],
        [1.0],
    ]
)
```

のように、各contextに対応するfeatureを定義します。

連続的なcontext descriptorには

```python
context_emb_feature
```

も利用できます。

---

## 12.41 robotorchan `LCEMGP`

基本形は

```python
model = LCEMGP(
    train_X=train_X,
    train_Y=train_Y,
    task_feature=2,
    context_cat_feature=context_cat_feature,
)
```

です。

wrapper は通常のrobotorchan exact-GP規約に従います。

---

## 12.42 Contextual GPとMultiTaskGPの違い

MultiTaskGPでは、task correlation matrixそのものを学ぶことが中心です。

Contextual modelでは、task/contextについて

- decomposition
- context category
- context descriptor
- latent embedding

を利用します。

したがって、contextに意味のある構造がある場合にcontextual modelが有力です。

---

## 12.43 Contextual GPとHierarchical GPの違い

これも重要です。

```text
Hierarchical GP
  → input variableのactive / inactive構造

Contextual GP
  → input block / context structure
```

です。

例えば

```text
工程方式によって使う変数が変わる
```

ならhierarchicalです。

一方

```text
工程1、工程2、工程3の条件ブロックがあり
最終scoreがそれらの寄与から決まる
```

ならcontextualです。

---

## 12.44 Heterogeneous MTGPとContextual GPの違い

```text
Heterogeneous MTGP
  → taskごとにfeature setが違う

Contextual GP
  → context構造・context similarityをモデル化したい
```

です。

例えば旧設備と新設備でセンサー項目が違う場合はheterogeneousです。

複数設備に共通するmetadataを使って設備間の関係を学びたいならcontextual / multitask側の考え方が近くなります。

---

## 12.45 ベイズ最適化との接続

これらのモデルも最終的にはBoTorch `Model` としてposteriorを返すため、BO surrogateとして利用できます。

ただし acquisition optimization では、**探索空間の構造を壊さないこと**が重要です。

例えばhierarchical spaceでは、

```text
process_type = 0
なのに
pressureだけを自由に最適化する
```

ような無意味な候補を避ける必要があります。

modelがhierarchyを理解していても、candidate generation側の制約設計は別問題です。

---

## 12.46 hierarchical BOでのcandidate generation

Hierarchical GPを使う場合、candidate generator側でも

- branchを列挙する
- branchごとにactive dimensionsだけを最適化する
- inactive dimensionsを固定値へ置く
- postprocessで無効パラメータを無視する

といった設計が必要です。

つまり

```text
hierarchical surrogate
≠
hierarchical optimizer
```

です。

---

## 12.47 Mixed optimizationとの関係

hierarchical parentがcategorical / discreteなら、

```python
optimize_acqf_mixed
```

のようなmixed optimizerと組み合わせる考え方もあります。

ただし単なるカテゴリ列挙だけでは、子featureのactive/inactive semanticsまでは自動で解決しません。

fixed feature combinationをbranch単位で構成する設計が自然です。

---

## 12.48 contextual BO

Contextual GPをBOへ使う場合、何をcontrolできるかを明確にします。

例えば

```text
context = 工程
```

でも、

- context自体を選べる
- contextは外生的に決まる
- contextごとの条件だけ最適化する

では問題設定が異なります。

contextが外生変数なら、acquisitionは

\[
\alpha(x\mid c)
\]

として現在contextを条件に最適化する考え方になります。

---

## 12.49 transfer learningとしての利用

HeterogeneousMTGP / LCEMGP は、関連taskからtarget taskへ情報共有する意味でtransfer learningとして解釈できます。

ただし重要なのは

```text
related tasks
```

であることです。

関連性が低いtaskを強制的に共有すると negative transfer が起きます。

posterior改善だけでなく、task covarianceやvalidation性能を確認します。

---

## 12.50 negative transfer

補助taskを追加したのにtarget taskの予測が悪化する場合、

- task correlationが弱い
- feature mappingが不適切
- noise levelが大きく異なる
- distribution shiftが大きい
- context featureが不十分

などが考えられます。

multi-task / contextual modelは「情報を共有できる」モデルであり、「共有すれば必ず良くなる」モデルではありません。

---

## 12.51 normalization

これらのモデルでもinput scalingは重要です。

連続featureは一般にunit cube付近へnormalizeするとkernel hyperparameterの学習が安定しやすくなります。

Hierarchical MultiTask GPでは、taskごとのoutcome scaleが異なる場合、task単位でstandardizationを考えるのが自然です。

単純に全taskをまとめて標準化すると、task固有のscale差を不自然に混ぜる場合があります。

---

## 12.52 inactive featureの値

Hierarchical modelではinactive featureにもtensor上は値を入れる必要があります。

ただしその値は、そのbranchで意味を持ちません。

例えば

```text
process_type = 0
pressure = 0.73
```

でも、pressureがinactiveなら本来の関数値には影響しないという前提です。

このため、データ前処理時に

```text
inactive value = arbitrary number
```

が予測へ影響していないか確認することが重要です。

---

## 12.53 invalid configurationとの違い

hierarchical inactive featureと、物理的にinvalidな候補は別です。

```text
inactive
  → そのbranchでは意味を持たない

invalid
  → 実験・製造できない
```

invalid candidateはconstraintとしてcandidate generation側で除外すべきです。

---

## 12.54 diagnostics: Hierarchical GP

Hierarchical modelでは次を確認します。

1. branchごとの予測誤差
2. branchごとのposterior uncertainty
3. inactive featureを変えてもposteriorが不自然に変わらないか
4. parent choice境界で不連続性が妥当か
5. `hierarchical_dependencies` のindexが正しいか
6. branchごとのデータ数が極端に偏っていないか
7. SAAS prior有無で結果が安定するか

---

## 12.55 diagnostics: Heterogeneous MTGP

1. target task単独モデルとの比較
2. auxiliary task追加による改善量
3. feature mapping `feature_indices` の妥当性
4. task correlation
5. auxiliary task noise
6. target task posterior calibration
7. negative transfer有無

を確認します。

---

## 12.56 diagnostics: Contextual GP

1. decompositionが物理的に妥当か
2. additive assumptionが妥当か
3. context embeddingの安定性
4. context間transferが合理的か
5. unseen / sparse contextへの予測
6. context metadataの有効性
7. SACGPとLCEAGPのvalidation比較

を確認します。

---

## 12.57 additive assumptionの破れ

SACGP / LCEAGPで

\[
f(x_A,x_B)
\neq
f_A(x_A)+f_B(x_B)
\]

となる強いinteractionがある場合、加法モデルでは表現しにくくなります。

例えば

```text
工程Aの温度効果が
工程Bの圧力によって反転する
```

ならcontext間interactionがあります。

この場合、

- interaction kernel
- 通常GP
- DKL
- custom kernel

などを検討します。

---

## 12.58 context embeddingの過学習

context数に対してデータが少ない場合、embeddingを自由に学習すると過学習する可能性があります。

`train_embedding=True` を使う場合でも、

- embedding dimension
- context数
- 各contextのデータ量

のバランスを確認します。

既知context featureがあるなら、それを利用した方が識別しやすい場合があります。

---

## 12.59 モデル選択フロー

```text
入力・タスクに特殊な構造がある？
  ↓
親変数によってactive featureが変わる？
 ├─ Yes
 │    ↓
 │   複数taskも共有する？
 │    ├─ No  → HierarchicalConditionalKernelGP
 │    └─ Yes → HierarchicalConditionalKernelMultiTaskGP
 │
 └─ No
      ↓
      taskごとにfeature setが違う？
       ├─ Yes → HeterogeneousMTGP
       └─ No
            ↓
            context decompositionがある？
             ├─ No → 通常GP / MultiTaskGP等
             └─ Yes
                  ↓
                  context別rewardを個別観測できる？
                   ├─ No
                   │    ↓
                   │   context間relationを学習したい？
                   │    ├─ No  → SACGP
                   │    └─ Yes → LCEAGP
                   │
                   └─ Yes → LCEMGP
```

---

## 12.60 実務比較表

| 問題 | 第一候補 |
|---|---|
| 装置方式で有効変数が変わる | `HierarchicalConditionalKernelGP` |
| 階層空間 + 複数関連task | `HierarchicalConditionalKernelMultiTaskGP` |
| taskごとに入力featureが異なる | `HeterogeneousMTGP` |
| context別の入力block + 集約reward | `SACGP` |
| context間similarityも学習したい | `LCEAGP` |
| context別rewardを個別に観測できる | `LCEMGP` |
| 単純なcontinuous + categorical | `MixedSingleTaskGP` |
| 同じfeature spaceのmulti-task | `MultiTaskGP` |
| 独立output | `ModelListGP` |

---

## 12.61 Notebookとの対応

実行例は次の3つです。

```text
examples/notebooks/12_hierarchical_gp.ipynb
examples/notebooks/13_heterogeneous_multitask_gp.ipynb
examples/notebooks/14_contextual_gp.ipynb
```

それぞれ

```text
12 → conditional active dimensions
13 → task-specific feature spaces
14 → contextual decomposition / embedding
```

を扱います。

---

## 12.62 robotorchan APIまとめ

### Hierarchical single-task

```python
model = HierarchicalConditionalKernelGP(
    train_X=train_X,
    train_Y=train_Y,
    hierarchical_dependencies=hierarchical_dependencies,
)
```

### Hierarchical multi-task

```python
model = HierarchicalConditionalKernelMultiTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    task_feature=task_feature,
    hierarchical_dependencies=hierarchical_dependencies,
)
```

### Heterogeneous multi-task

```python
model = HeterogeneousMTGP(
    train_Xs=train_Xs,
    train_Ys=train_Ys,
    train_Yvars=None,
    feature_indices=feature_indices,
    full_feature_dim=full_feature_dim,
)
```

### Context additive

```python
model = SACGP(
    train_X=train_X,
    train_Y=train_Y,
    train_Yvar=None,
    decomposition=decomposition,
)
```

### Latent-context embedding additive

```python
model = LCEAGP(
    train_X=train_X,
    train_Y=train_Y,
    train_Yvar=None,
    decomposition=decomposition,
)
```

### Latent-context embedding multi-output

```python
model = LCEMGP(
    train_X=train_X,
    train_Y=train_Y,
    task_feature=task_feature,
)
```

---

## 12.63 共通学習API

これらのwrapperは基本的にExact GPとして

```python
mll = model.make_mll()
fit_gpytorch_mll(mll)
```

を利用できます。

robotorchanの共通APIにより、特殊kernel / task structureを持つモデルでも学習コードを揃えられます。

---

## 12.64 raw-data provenance

通常wrapperでは

```python
model.raw_train_X
model.raw_train_Y
model.raw_train_Yvar
```

を利用できます。

`HeterogeneousMTGP` ではtask別の入力形状が異なるため、

```python
model.raw_train_Xs
model.raw_train_Ys
model.raw_train_Yvars
```

です。

この違いはdata semanticsを反映したものです。

---

## 12.65 最も重要な判断

この章のモデル選択では、モデル名より先に次を確認します。

```text
1. なぜ入力構造が特殊なのか？
2. branchなのかtaskなのかcontextなのか？
3. どの情報を共有したいのか？
4. どの情報は共有すべきでないのか？
5. BO candidate generation側でも構造を守れているか？
```

特に

```text
hierarchical
heterogeneous
contextual
```

は「高性能な汎用GP」ではなく、**問題構造を明示的に仮定するGP**です。

仮定が正しければsample efficiencyの改善が期待できますが、誤った構造を与えると通常GPより悪化する可能性があります。

---

## 12.66 まとめ

本章では、入力・task・contextに構造があるGPを整理しました。

重要な対応は次の通りです。

1. parent choiceでactive dimensionsが変わる
   → `HierarchicalConditionalKernelGP`
2. 上記 + multi-task
   → `HierarchicalConditionalKernelMultiTaskGP`
3. taskごとにfeature setが異なる
   → `HeterogeneousMTGP`
4. context blockの加法構造 + aggregated reward
   → `SACGP`
5. context間similarityも学習
   → `LCEAGP`
6. context別rewardを個別観測
   → `LCEMGP`

これらを適切に使うには、kernel選択よりも先に

```text
problem structure
```

を正しく定義することが重要です。

次章では、ここまでのすべてのモデルを横断し、問題設定から候補モデルを選ぶための [Model Selection](13_model_selection.md) を整理します。

## 参考

- BoTorch `HierarchicalConditionalKernelGP`
- BoTorch `HierarchicalConditionalKernelMultiTaskGP`
- BoTorch `HeterogeneousMTGP`
- BoTorch `SACGP`
- BoTorch `LCEAGP`
- BoTorch `LCEMGP`
- [Mixed Variables](05_mixed_variables.md)
- [Multi-task / Multi-output](07_multitask_multioutput.md)
- [High-dimensional GP](08_high_dimensional_gp.md)
