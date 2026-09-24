# 7. Multi-task / Multi-output Gaussian Process

## 7.1 この章で扱うこと

実務では、1つの入力 `x` に対して複数の関連する応答や、条件・装置・材料ごとの関連タスクを同時に扱いたいことがあります。

```text
同じプロセス条件 x
  ├─ 装置Aでの品質
  ├─ 装置Bでの品質
  └─ 装置Cでの品質
```

あるいは

```text
同じ材料 x
  ├─ 強度
  ├─ 導電率
  └─ 熱伝導率
```

です。

このとき重要なのは、単に「出力が複数ある」ことと「出力間の相関を利用して学習する」ことを区別することです。

本章では、robotorchanで利用できる

- `MultiTaskGP`
- `KroneckerMultiTaskGP`
- `ModelListGP`

を中心に、Multi-task / Multi-output GPの考え方と使い分けを説明します。

---

## 7.2 Multi-outputとは

入力 `x` に対して複数の出力を返す問題を考えます。

\[
f(x)=
\begin{bmatrix}
f_1(x)\\
f_2(x)\\
\vdots\\
f_m(x)
\end{bmatrix}
\]

例えば

\[
f(x)=(\text{strength},\text{conductivity},\text{cost})
\]

です。

ただし、**multi-outputという言葉だけでは出力間の依存関係をモデル化するかどうかは決まりません**。

独立なGPを並べてもmulti-output modelになります。

---

## 7.3 Multi-taskとは

Multi-task GPでは、複数の関連タスクを同時にモデル化し、タスク間の相関を利用します。

タスクを `t` とすると

\[
f(x,t)
\]

を考えます。

例えば

```text
t = 0 : 装置A
t = 1 : 装置B
t = 2 : 装置C
```

です。

ある装置のデータから別装置の応答を推定できるなら、タスク間の情報共有が有効です。

```text
Task A data ─┐
Task B data ─┼→ shared multi-task GP → posterior for each task
Task C data ─┘
```

---

## 7.4 Single-task GPとの違い

タスクごとに完全に独立したGPを作るなら

\[
f_t(x)\sim\mathcal{GP}(m_t(x),k_t(x,x'))
\]

です。

この場合、Task Aの観測はTask Bのposteriorへ影響しません。

Multi-task GPでは

\[
f(x,t)\sim\mathcal{GP}
\left(
 m(x,t),
 k((x,t),(x',t'))
\right)
\]

とし、異なるタスク間にもcovarianceを持たせます。

そのため

\[
\operatorname{Cov}(f(x,t),f(x',t'))\neq 0
\]

となり得ます。

---

## 7.5 Task covariance

Multi-task GPの中心は **task covariance** です。

入力間のcovarianceを

\[
K_X
\]

タスク間のcovarianceを

\[
K_T
\]

とします。

例えば3タスクなら

\[
K_T=
\begin{bmatrix}
1.0 & 0.8 & 0.2\\
0.8 & 1.0 & 0.3\\
0.2 & 0.3 & 1.0
\end{bmatrix}
\]

のような構造を考えられます。

この例ではTask 1とTask 2の関係が強く、Task 1とTask 3の関係は弱いと解釈できます。

ただし、推定されたtask covarianceを単純な因果関係や物理的関係と解釈してはいけません。これはモデルが観測データから学習した共分散構造です。

---

## 7.6 Intrinsic Coregionalization Model (ICM)

代表的なMulti-task GPの構造が **ICM (Intrinsic Coregionalization Model)** です。

典型的には

\[
k((x,t),(x',t'))
=
k_X(x,x')k_T(t,t')
\]

とします。

つまり

```text
入力が似ているか
    ×
タスクが似ているか
    ↓
最終的な covariance
```

です。

行列としては、block designの場合に

\[
K=K_X\otimes K_T
\]

というKronecker積構造が現れます。

BoTorch 0.18.1の`MultiTaskGP`と`KroneckerMultiTaskGP`はいずれもICMの考え方を利用しますが、データ表現と計算構造が異なります。

---

## 7.7 MultiTaskGP

BoTorchの`MultiTaskGP`は、入力の1列をtask featureとして持つlong-format表現を使います。

例えば通常の入力が

```text
[x1, x2]
```

なら

```text
[x1, x2, task]
```

として与えます。

例:

```text
x1   x2   task   y
0.1  0.2   0    1.3
0.4  0.8   0    2.1
0.2  0.3   1    1.6
0.7  0.5   1    2.5
```

タスクごとに異なる `x` で観測されていても扱えることが重要です。

---

## 7.8 MultiTaskGPのcovariance

BoTorch 0.18.1の`MultiTaskGP`では、non-task featureに対するdata covarianceとtask covarianceを組み合わせます。

概念的には

\[
k((x,t),(x',t'))
=k_X(x,x')k_T(t,t')
\]

です。

したがってTask Aの観測は、

- `x` が近い
- task covarianceが大きい

ほどTask Bのposteriorにも強く影響します。

robotorchanはこのBoTorchの挙動を維持した薄いwrapperです。

---

## 7.9 task_feature

`MultiTaskGP`では、どの入力列がtaskを表すかを`task_feature`で指定します。

例えば

```python
model = MultiTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    task_feature=-1,
)
```

なら最後の列をtask featureとして使います。

タスク番号はモデルにとってtask identityを表すための値であり、通常の連続変数として「task 2はtask 1の2倍」という意味ではありません。

---

## 7.10 output_tasks

`MultiTaskGP`では、posteriorとして返したいtaskを`output_tasks`で指定できます。

例えば複数taskを学習に利用しつつ、BOで最終的に重要なtaskだけを出力対象にする構成も可能です。

これはMulti-Fidelityでtarget fidelityを意識したのと似た場面がありますが、Multi-taskには一般に精度やコストの順序は必要ありません。

---

## 7.11 rankとtask covariance

Task covarianceは低rank構造で表現できます。

概念的には

\[
K_T\approx BB^\top + D
\]

のように、少数の潜在因子でタスク関係を表現します。

`rank`を小さくするとtask covarianceの自由度を抑えられます。

```text
rank 小
  └─ 少数の共通潜在因子を仮定

rank 大
  └─ より柔軟なtask covariance
```

データが少ない状態で過度に柔軟なtask covarianceを推定すると不安定になるため、rankは表現力と推定安定性のトレードオフになります。

---

## 7.12 Positive task correlationについて

BoTorch 0.18.1の`MultiTaskGP`はデフォルトで`PositiveIndexKernel`をtask covarianceに利用し、task covarianceの要素を非負に制約する設計です。

これは限られたデータからtask correlationを安定して推定するための実務的な仮定です。

したがって、強い負相関を本質的に表現したい問題では、デフォルト設定の仮定が適切かを確認する必要があります。

---

## 7.13 KroneckerMultiTaskGP

`KroneckerMultiTaskGP`は、**すべてのtaskが同じ入力点で観測されるblock design**を前提にします。

入力が

\[
X\in\mathbb{R}^{n\times d}
\]

出力が

\[
Y\in\mathbb{R}^{n\times m}
\]

です。

例えば

```text
x1   x2   taskA  taskB  taskC
0.1  0.2   1.2    0.8    2.1
0.4  0.7   1.8    1.3    2.6
0.8  0.5   2.4    1.9    3.0
```

のように、各 `x` ですべてのtaskが測定されています。

---

## 7.14 Block design

Block designとは

```text
x_1 → task 1, task 2, ..., task m
x_2 → task 1, task 2, ..., task m
...
x_n → task 1, task 2, ..., task m
```

のように、同じ入力集合ですべてのtaskを観測するデータ構造です。

この構造があると、共分散を

\[
K=K_X\otimes K_T
\]

として扱えます。

これが`KroneckerMultiTaskGP`という名前の由来です。

---

## 7.15 Kronecker積の意味

\[
K_X\in\mathbb{R}^{n\times n}
\]

\[
K_T\in\mathbb{R}^{m\times m}
\]

なら

\[
K_X\otimes K_T
\]

は全 `n × m` 個の関数値間のcovarianceを表します。

直感的には

```text
入力間構造 K_X
      ×
タスク間構造 K_T
      ↓
入力 × タスク全体の covariance
```

です。

Kronecker構造を利用すると、巨大な `(nm) × (nm)` covariance matrixを常に密行列として直接扱うより、線形代数上の構造を利用できる場合があります。


### 7.15.1 Kronecker extensionで何を変更するか

robotorchanのKronecker extensionでは、block designとtask covarianceを固定したまま、
主にdata covariance factorを拡張します。基準となるlatent covarianceは

\[
K_f = K_{data} \otimes K_{task}
\]

です。

Spectral Mixture variantでは

\[
K_{data}=K_{SM}
\]

Infinite-width BNN variantでは

\[
K_{data}=K_{NNGP}
\]

とし、task covariance factorは別に維持します。Mixed variantでもcategorical covarianceは
data factor側の構造であり、task identityとは異なります。

この分離は重要です。task indexを入力特徴へ追加してlong-format modelへ変形すると、
block-design Kronecker modelとは異なる統計モデルになります。同様に、入力依存noiseや
heavy-tailed likelihoodは単なるdata kernel交換ではありません。latent covariance
`K_data ⊗ K_task` に加えてobservation modelを定義する必要があります。

したがって、robotorchanではKronecker extensionを次の3種類に区別します。

1. **data-factor extension**: Mixed、reduction、Nonstationary、Spectral Mixture、NNGP。
2. **observation-model extension**: heteroskedastic、replicate noise、RRP、Student-t、
   contaminated。専用のblock-design observation semanticsが必要です。
3. **inference-architecture extension**: SAAS fully BayesianやDeepGP。kernel交換だけでは
   成立せず、Pyro/variational inferenceのownershipまで含む設計が必要です。

この区別により、「別のMultiTask版が存在する」ことだけを理由にKronecker版を追加することを
避けます。

---

## 7.16 MultiTaskGPとKroneckerMultiTaskGPの違い

最も重要な違いは**学習データの形**です。

| 観点 | `MultiTaskGP` | `KroneckerMultiTaskGP` |
|---|---|---|
| データ形式 | long format | wide / block design |
| task指定 | `task_feature` | `train_Y`の出力軸 |
| taskごとに異なるX | 可能 | 基本不可 |
| 全taskが同一X | 必須ではない | 必須 |
| task correlation | 利用 | 利用 |
| ICM | Yes | Yes |
| Kronecker構造 | 一般のlong-formatとして扱う | 明示的に利用 |

したがって

```text
全taskが同じXで観測？
  ├─ Yes → KroneckerMultiTaskGPを検討
  └─ No  → MultiTaskGPを検討
```

が基本的な判断です。

---

## 7.17 ModelListGP

`ModelListGP`は複数のGPをまとめるcontainerです。

例えば

```python
model_1 = SingleTaskGP(X1, Y1)
model_2 = SingleTaskGP(X2, Y2)
model = ModelListGP(model_1, model_2)
```

とします。

重要なのは、**各sub-modelは独立**だということです。

つまり

\[
\operatorname{Cov}(f_1(x),f_2(x'))=0
\]

というモデル化に相当します。

---

## 7.18 ModelListGPを使う理由

独立なら別々のmodelを持てばよいように見えますが、BoTorchではmulti-output acquisitionへ共通interfaceで渡すために`ModelListGP`が便利です。

さらに各outputで

- 異なるtraining X
- 異なるkernel
- 異なるnoise model
- 異なるGP model type

を使えます。

そのため非常に柔軟です。

---

## 7.19 MultiTaskGPとModelListGPの本質的な違い

```text
MultiTaskGP
  └─ task correlationを学習して情報共有

ModelListGP
  └─ independent modelsを1つのmulti-output interfaceに束ねる
```

例えばTask Aのデータが少ないとします。

`MultiTaskGP`ではTask Bと強く関連していれば、Task Bの観測がTask Aの予測を改善できます。

`ModelListGP`ではTask BのデータはTask Aのposteriorを直接改善しません。

したがって「出力が複数だからMultiTaskGP」ではなく、**出力間で統計的情報を共有したいか**が判断基準です。

---

## 7.20 Multi-outputとMulti-objectiveの違い

Multi-outputはモデルの出力数に関する概念です。

Multi-objectiveは最適化したいobjectiveの数に関する概念です。

例えば

```text
output 1 = 強度
output 2 = コスト
output 3 = 測定時間
```

をモデル化していても、最適化で強度だけを使えばsingle-objectiveです。

逆に強度最大化とコスト最小化を同時に扱えばmulti-objective BOです。

```text
Multi-output
  └─ surrogateが複数outputを返す

Multi-objective
  └─ optimization objectiveが複数
```

両者は関連しますが同義ではありません。

---

## 7.21 Multi-taskとMulti-Fidelityの違い

前章のMulti-Fidelityも関連情報源を共有するため、Multi-taskと似ています。

しかしMulti-Fidelityには通常

- fidelityの順序
- 評価cost
- target fidelity

があります。

Multi-taskでは

```text
装置A
装置B
装置C
```

のように、どれかが「高精度版」である必要はありません。

```text
精度・cost階層 + targetがある
  → Multi-Fidelity

関連する複数task間で情報共有したい
  → Multi-task
```

と考えると整理しやすくなります。

---

## 7.22 Multi-taskとカテゴリ変数の違い

Task IDとcategorical variableは表面的には似ています。

例えば

```text
machine = A / B / C
```

を「設計変数として選びたい」のか、「関連する複数のデータ源として情報共有したい」のかで意味が変わります。

```text
machine自体を最適化対象として選択
  → categorical design variable

machineごとの応答を関連taskとして学習
  → task feature
```

同じ列でも問題設定によって役割が変わるため、モデルを選ぶ前に「何を最適化したいか」を明確にする必要があります。

---

## 7.23 Multi-task BO

Multi-task modelをBOで使うと、関連taskの観測を利用して目的taskの探索効率を上げられます。

例えば

```text
Task A : 過去装置
Task B : 新装置
```

で、新装置Bのデータが少ない場合を考えます。

Task Aとの相関が強ければ

```text
大量のTask A data
       +
少量のTask B data
       ↓
MultiTaskGP
       ↓
Task B posterior
       ↓
Acquisition Function
       ↓
次のTask B条件
```

というtransferが可能です。

ただしtask correlationが弱い場合、無理な情報共有は有利とは限りません。

---

## 7.24 Negative transfer

Multi-task learningでは、関連性の低いtaskを共有すると性能が悪化することがあります。

これを一般にnegative transferと考えられます。

```text
Task AとBが強く関連
  → sharingが有効

Task Cは全く異なる
  → sharingがbiasを持ち込む可能性
```

したがって

- task covariance
- taskごとの予測精度
- independent modelとの比較

を確認することが重要です。

Multi-task modelを使うこと自体が目的ではなく、**共有によって目的taskの予測・意思決定が改善するか**を検証します。

---

## 7.25 データ量が不均衡な場合

Multi-task GPの利点が出やすい典型例は、task間でデータ量が不均衡な場合です。

```text
Task A : 100 points
Task B : 100 points
Task C : 10 points
```

Task CがA/Bと関連していれば、共有により少量taskの推定を改善できる可能性があります。

一方、Task Cが独立なら`ModelListGP`などの方が自然です。

---

## 7.26 Missing task observations

`MultiTaskGP`のlong-format表現では、全taskが同じXで観測されている必要はありません。

```text
x1 → task Aのみ
x2 → task A, B
x3 → task Bのみ
```

のようなデータも表現できます。

一方、`KroneckerMultiTaskGP`はblock designを仮定するため、このような欠測をそのままwide-formatで扱う用途には向きません。

この違いはモデル選択で非常に重要です。

---

## 7.27 Noiseについて

Taskごとに観測noiseが異なる問題はよくあります。

例えば

```text
simulation task : noise小
experiment task : noise大
```

です。

BoTorch 0.18.1の`MultiTaskGP`では、既知noiseなら`train_Yvar`を渡せます。一方、noiseを推定するデフォルト構成にはtask間noiseの扱いに制約があるため、taskごとに異なるnoiseを明示的に表現したい場合はlikelihood設計を確認する必要があります。

モデル選択ではtask covarianceだけでなくnoise modelも確認します。

---

## 7.28 Outcome standardization

Multi-task modelではtaskごとの出力scaleが異なることがあります。

```text
Task A : 0 ～ 1
Task B : 100 ～ 500
```

この場合、全taskをまとめて一括標準化するとtask間のscale差を不適切に扱う可能性があります。

BoTorchの`MultiTaskGP`では、標準化をtask単位で考えることが重要です。

これはtask covarianceの推定安定性にも関係します。

---

## 7.29 robotorchan: MultiTaskGPの基本例

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import MultiTaskGP

torch.set_default_dtype(torch.double)

# [x1, x2, task]
train_X = torch.tensor(
    [
        [0.10, 0.20, 0.0],
        [0.40, 0.70, 0.0],
        [0.20, 0.30, 1.0],
        [0.60, 0.50, 1.0],
    ]
)
train_Y = torch.tensor([[0.20], [0.55], [0.30], [0.70]])

model = MultiTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    task_feature=-1,
)

mll = model.make_mll()
fit_gpytorch_mll(mll)
```

この例では最後の列がtask featureです。

---

## 7.30 robotorchan: KroneckerMultiTaskGPの基本例

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import KroneckerMultiTaskGP

torch.set_default_dtype(torch.double)

train_X = torch.tensor(
    [
        [0.10, 0.20],
        [0.40, 0.70],
        [0.80, 0.50],
    ]
)

# 同じXでtask 0, 1を両方観測
train_Y = torch.tensor(
    [
        [0.20, 0.35],
        [0.55, 0.60],
        [0.80, 0.75],
    ]
)

model = KroneckerMultiTaskGP(
    train_X=train_X,
    train_Y=train_Y,
)

mll = model.make_mll()
fit_gpytorch_mll(mll)
```

`train_Y.shape[-1]`がtask数に対応します。

---

## 7.31 robotorchan: ModelListGPの基本例

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import ModelListGP, SingleTaskGP

torch.set_default_dtype(torch.double)

X1 = torch.rand(10, 2)
Y1 = torch.sin(X1[:, :1])

X2 = torch.rand(15, 2)
Y2 = torch.cos(X2[:, :1])

model_1 = SingleTaskGP(X1, Y1)
model_2 = SingleTaskGP(X2, Y2)
model = ModelListGP(model_1, model_2)

mll = model.make_mll()
fit_gpytorch_mll(mll)
```

robotorchanの`ModelListGP.make_mll()`は`SumMarginalLogLikelihood`を返します。

---

## 7.32 Raw Data API

`MultiTaskGP`と`KroneckerMultiTaskGP`は他のsupervised wrapperと同様に、構築時のraw dataを保持します。

```python
model.raw_train_X
model.raw_train_Y
model.raw_train_Yvar
model.make_mll()
```

`ModelListGP`は複数のchild modelを持つcontainerなので、container levelでは

```python
model.raw_train_Xs
model.raw_train_Ys
model.raw_train_Yvars
```

として各child modelのraw dataを参照します。

`raw_*`は構築時provenanceであり、`condition_on_observations()`などで更新された現在の学習状態そのものではありません。

---

## 7.33 どのモデルを選ぶか

基本的には次の順で判断できます。

```text
複数のoutput / taskがある
        ↓
output間の情報共有が必要？
  ├─ No
  │   └─ ModelListGP / independent GP
  │
  └─ Yes
      ↓
全taskを同じXで観測？
      ├─ Yes
      │   └─ KroneckerMultiTaskGP
      │
      └─ No
          └─ MultiTaskGP
```

ただし、同じX・同じmodel structureでoutput correlationを使わない場合は、BoTorchのbatched model表現も候補になります。

---

## 7.34 実務上の選択表

| 状況 | 第一候補 |
|---|---|
| 1つのoutputだけ | `SingleTaskGP` |
| 複数outputだが独立 | `ModelListGP` |
| outputごとに異なるX・異なるmodel | `ModelListGP` |
| 関連task、taskごとにXが異なる | `MultiTaskGP` |
| 関連task、全taskが同一X | `KroneckerMultiTaskGP` |
| 少量taskへ他taskの情報をtransferしたい | `MultiTaskGP` / `KroneckerMultiTaskGP` |
| task間相関が疑わしい | independent modelとの比較を推奨 |
| 精度・cost階層とtargetがある | Multi-Fidelityも検討 |

---

## 7.35 よくある誤解

### 「出力が2つならMultiTaskGP」

必ずしもそうではありません。独立outputなら`ModelListGP`が自然です。

### 「MultiTaskGPとKroneckerMultiTaskGPは同じ入力形式」

異なります。`MultiTaskGP`はtask featureを含むlong format、`KroneckerMultiTaskGP`は同一Xですべてのtaskを観測するblock designです。

### 「KroneckerMultiTaskGPの方が常に上位」

違います。block designという強いデータ構造を利用するモデルであり、taskごとにXが異なる場合には適しません。

### 「Task IDは連続変数」

違います。Task IDはtask identityを表します。

### 「Task covarianceが高いから因果関係がある」

Task covarianceは統計的な共分散構造であり、因果関係を意味しません。

### 「Multi-taskなら必ず精度が上がる」

Taskが十分関連していなければnegative transferが起こり得ます。独立modelとの比較が重要です。

---

## 7.36 ベイズ最適化での使い分け

モデル選択とacquisition設計は分けて考えます。

```text
Training data
    ↓
MultiTaskGP / KroneckerMultiTaskGP / ModelListGP
    ↓
Multi-output posterior
    ↓
Objective / PosteriorTransform
    ↓
Acquisition Function
    ↓
Candidate
```

例えば複数taskをモデル化していても、特定taskだけをobjectiveにできます。

また複数outputをmulti-objective BOに使うこともできます。

したがって「surrogateが何をモデル化するか」と「BOで何を最適化するか」を分離して設計することが重要です。

---

## 7.37 まとめ

Multi-task / Multi-output GPで最も重要なのは、**複数の出力を持つことと、出力間の相関を利用することを区別すること**です。

```text
Independent outputs
  └─ ModelListGP

Related tasks
  ├─ different X allowed
  │    └─ MultiTaskGP
  │
  └─ same X for every task
       └─ KroneckerMultiTaskGP
```

理論的には

\[
k((x,t),(x',t'))
=
k_X(x,x')k_T(t,t')
\]

というICMの考え方が中心です。

特に覚えておきたい点は次の通りです。

1. `MultiTaskGP`はtask featureを使うlong-format model
2. `KroneckerMultiTaskGP`はblock designを前提とする
3. `ModelListGP`は独立GPのcontainer
4. task covarianceによって関連task間で情報共有できる
5. 関連性が弱いtaskではnegative transferに注意する
6. Multi-outputとMulti-objectiveは同義ではない
7. Multi-taskとMulti-Fidelityも目的が異なる

次章では、高次元入力で通常のGPが難しくなる理由と、SAASなどの考え方を扱う [High-dimensional GP](08_high_dimensional_gp.md) を説明します。

## 参考

- BoTorch v0.18.1 `MultiTaskGP`
- BoTorch v0.18.1 `KroneckerMultiTaskGP`
- BoTorch v0.18.1 `ModelListGP`
- Bonilla, Chai, Williams, *Multi-task Gaussian Process Prediction*, NeurIPS 2007
- Swersky, Snoek, Adams, *Multi-Task Bayesian Optimization*, NeurIPS 2013
- [Kernel](03_kernel.md)
- [Multi-Fidelity](06_multi_fidelity.md)
