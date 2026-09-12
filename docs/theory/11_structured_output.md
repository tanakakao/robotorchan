# 11. Structured Output GP

## 11.1 この章で扱うこと

通常の Gaussian Process（GP）回帰では、1つの入力 `x` に対して1つの scalar output `y` を予測する問題を考えます。

しかし実際の科学・工学データでは、1回の実験から

- スペクトル
- 時系列
- 温度分布
- 画像
- 2次元マップ
- 3次元場
- 波長 × 時間の応答

のような **構造を持つ出力** が得られることがあります。

このような問題を structured output regression と考えます。

robotorchan では主に

```python
HigherOrderGP
LatentKroneckerGP
```

を利用します。

本章では、

- structured output とは何か
- vector output と tensor output の違い
- separable covariance
- Kronecker product
- `HigherOrderGP`
- latent output coordinates
- `LatentKroneckerGP`
- missing outputs
- pathwise conditioning
- iterative inference
- structured output と Bayesian Optimization
- robotorchan での使い分け

を整理します。

---

## 11.2 Structured Output とは

通常の scalar regression は

\[
x \mapsto y
\]

です。

multi-output regression なら

\[
x \mapsto \mathbf y
\]

です。

Structured Output では、さらに出力の並び方・位置関係そのものに意味があります。

例えばスペクトルなら

\[
y(x, \lambda)
\]

画像なら

\[
y(x, u, v)
\]

時空間場なら

\[
y(x,t,r)
\]

のように、出力側にも座標や軸があります。

```text
入力 x
  ↓
実験・シミュレーション
  ↓
出力 tensor

例:
32波長スペクトル      → [32]
64×64画像             → [64, 64]
20時間×50位置         → [20, 50]
16×16×16 volume       → [16, 16, 16]
```

重要なのは、出力要素同士が独立ではないことです。

---

## 11.3 出力を単純にflattenすればよいのか

例えば `64 x 64` 画像なら、flattenして4096出力として扱うことは理論上可能です。

しかし、出力間の完全な covariance matrix を持つと

\[
4096 \times 4096
\]

の巨大な covariance を考える必要があります。

しかも flatten しただけでは

```text
隣接pixelは似ている
波長が近いほど相関しやすい
時間的に近い点は似ている
```

という構造を積極的には利用できません。

Structured Output GP はこの構造を covariance に組み込みます。

---

## 11.4 Kronecker 構造

Structured Output GP の重要な道具が Kronecker product です。

2つの covariance matrix

\[
K_X
\]

\[
K_T
\]

から

\[
K = K_X \otimes K_T
\]

を作ると、入力側と出力軸側の covariance を分離して表現できます。

例えば

```text
K_X
  → 実験条件どうしの似方

K_T
  → 波長位置どうしの似方

K_X ⊗ K_T
  → 条件 × 波長の全体 covariance
```

です。

---

## 11.5 separable covariance

最も基本的な仮定は

\[
k((x,t),(x',t'))
=
k_X(x,x')k_T(t,t')
\]

です。

これは **separable covariance** と呼ばれます。

意味としては、

```text
全体の相関
=
入力条件の相関
×
出力座標の相関
```

です。

この仮定は強い一方、巨大な covariance matrix を構造化して扱える大きな利点があります。

---

## 11.6 HOGP と LatentKroneckerGP の違い

robotorchan で最も重要な区別です。

```text
HigherOrderGP
  └─ 出力 tensor の shape / 各軸の構造を利用する

LatentKroneckerGP
  └─ 入力 X と明示的な出力座標 T の積空間をモデル化する
```

より具体的には、

```text
画像・多次元grid
    → HigherOrderGP

波長・時間・位置など
意味のある座標Tを明示したい
    → LatentKroneckerGP
```

と考えると分かりやすいです。

---

# HigherOrderGP

## 11.7 HigherOrderGP とは

BoTorch の `HigherOrderGP` は tensor-valued output のための GP です。

BoTorch の説明でいう “higher-order” は、予測対象が matrix / tensor であることを意味します。

例えば

```text
train_X: [n, d]
train_Y: [n, h, w]
```

なら、各入力に対して `h x w` の画像を予測します。

1次元スペクトルも利用できますが、特に

- 画像
- 空間grid
- 時間 × 空間
- 波長 × 時間

のような複数出力軸を持つ tensor で自然です。

---

## 11.8 HOGP の covariance

入力軸と各出力軸に covariance を持たせます。

例えば出力が2次元なら概念的に

\[
K
=
K_X
\otimes
K_1
\otimes
K_2
\]

です。

ここで

- \(K_X\): input covariance
- \(K_1\): output axis 1 covariance
- \(K_2\): output axis 2 covariance

です。

出力が3次元ならさらに

\[
K_3
\]

が加わります。

---

## 11.9 なぜ Kronecker が効くのか

例えば出力 shape が

```text
10 × 20 × 30
```

なら出力要素数は

\[
10\times20\times30=6000
\]

です。

直接6000次元の covariance を持てば

\[
6000^2=36,000,000
\]

要素です。

一方 Kronecker 構造なら

```text
10×10
20×20
30×30
```

の小さな covariance の組み合わせとして表現できます。

これが structured covariance の主要な利点です。

---

## 11.10 HOGP の latent representation

HOGP では各出力軸について latent coordinates を持ち、それらの上で kernel covariance を構成します。

robotorchan wrapper の constructor には

```python
num_latent_dims
learn_latent_pars
latent_init
```

があります。

つまり単に tensor index を固定座標として扱うだけではなく、出力軸の latent representation 自体を学習できます。

---

## 11.11 `num_latent_dims`

`num_latent_dims` は各 output dimension を表す latent coordinate の次元数を制御します。

概念的には

```text
output axis
  ↓
latent coordinate
  ↓
kernel
  ↓
axis covariance
```

です。

latent dimension を増やせば柔軟性は増しますが、parameter 数も増えます。

---

## 11.12 `learn_latent_pars`

```python
learn_latent_pars=True
```

では output latent parameters を学習します。

これは

```text
出力 index 間の関係を data から学ぶ
```

ことに相当します。

一方で物理的な波長や時間座標がすでに分かっている場合、その座標を直接モデルに渡せる `LatentKroneckerGP` の方が解釈しやすいことがあります。

---

## 11.13 HOGP の入力 shape

robotorchan wrapper では

```python
HigherOrderGP(
    train_X=train_X,
    train_Y=train_Y,
)
```

とします。

典型的に

```text
train_X
batch_shape × n × d

train_Y
batch_shape × n × output_shape
```

です。

例えば画像なら

```text
train_X: [30, 5]
train_Y: [30, 64, 64]
```

です。

---

## 11.14 robotorchan の HOGP wrapper

現在の wrapper は BoTorch の挙動をそのまま利用し、主に

- raw training data 保持
- `make_mll()`
- 共通 model API

を追加しています。

```python
from robotorchan.models import HigherOrderGP

model = HigherOrderGP(
    train_X=train_X,
    train_Y=train_Y,
)

model.raw_train_X
model.raw_train_Y
model.make_mll()
```

です。

`raw_train_Yvar` は `None` です。

---

## 11.15 HOGP の fitting

BoTorch では HOGP の fitting に注意が必要です。

specialized Kronecker solve を有効にし、通常の SciPy L-BFGS-B ベースの `fit_gpytorch_mll()` ではなく、PyTorch optimizer を使う `fit_gpytorch_mll_torch()` が推奨されます。

```python
from botorch.fit import fit_gpytorch_mll_torch
from linear_operator.settings import _fast_solves

mll = model.make_mll()

with _fast_solves(True):
    fit_gpytorch_mll_torch(mll)
```

これは approximate / structured computations により MLL が滑らかでない場合があり、L-BFGS-B と相性が悪いためです。

---

## 11.16 HOGP posterior

学習後は

```python
posterior = model.posterior(test_X)
```

で structured posterior を取得できます。

posterior mean の shape も output shape を保ちます。

例えば

```text
test_X: [4, 5]
output_shape: [32, 32]
```

なら、概念的に

```text
posterior.mean: [4, 32, 32]
```

です。

---

## 11.17 HOGP posterior sampling

HOGP の posterior sampling は structured covariance を利用します。

BoTorch は posterior sampling に Matheron's rule を利用します。

概念的には

```text
prior sample
   +
conditioning correction
   ↓
posterior sample
```

です。

巨大な full posterior covariance を単純に構築するより、構造を活かしてsampleを得られます。

---

## 11.18 HOGP の仮定

HOGP は便利ですが、出力軸 covariance が Kronecker 分解できるという構造を利用しています。

したがって

```text
axis 1 と axis 2 の相互作用が
単純な separable structure では表現しにくい
```

場合にはモデルミスマッチが生じます。

Kronecker は計算効率と引き換えに structural assumption を置いていると理解する必要があります。

---

# LatentKroneckerGP

## 11.19 LatentKroneckerGP とは

`LatentKroneckerGP` は

```text
X: 実験条件・設計変数
T: 時間・波長・位置などの出力座標
Y: X × T 上の観測
```

を明示的に扱います。

モデル化のイメージは

\[
y = f(x,t)
\]

です。

covariance は概念的に

\[
k((x,t),(x',t'))
=
k_X(x,x')k_T(t,t')
\]

です。

---

## 11.20 `train_T`

robotorchan の wrapper は

```python
LatentKroneckerGP(
    train_X=train_X,
    train_T=train_T,
    train_Y=train_Y,
)
```

と使います。

例えばスペクトルなら

```text
train_X: [n, d]
train_T: [m, 1]   # wavelength
train_Y: [n, m]
```

です。

`T` が出力軸そのものを表します。

---

## 11.21 HOGP と異なる点

HOGP では output tensor の index / latent coordinates が中心です。

LatentKroneckerGP では

```text
波長 = 450 nm
時間 = 12.5 s
位置 = 3.2 mm
```

のような意味のある output coordinate を直接与えられます。

そのため

```text
Tに物理的意味がある
```

場合は非常に自然です。

---

## 11.22 新しい T で予測できる

大きな利点の1つは、学習時の `train_T` と異なる `test_T` で posterior を評価できることです。

```python
posterior = model.posterior(
    test_X,
    test_T,
)
```

です。

例えば学習データが

```text
450, 460, 470, ..., 700 nm
```

でも、posterior を

```text
455, 465, 475, ...
```

で評価できます。

これは output coordinate 上での interpolation に相当します。

---

## 11.23 missing output への対応

LatentKroneckerGP の重要な特徴は、完全な block design でなくても Kronecker structure を活用できる点です。

つまり、すべての `X` で全 `T` が観測されていなくても扱える設計です。

```text
X1: T1 T2 T3 T4 観測
X2: T1 -- T3 T4 観測
X3: -- T2 T3 -- 観測
```

のような missing entries を想定できます。

これは完全gridを要求する単純な Kronecker model に対する重要な拡張です。

---

## 11.24 latent Kronecker structure

missing entries があると full Kronecker product を単純には使えません。

LatentKroneckerGP は underlying latent product-space structure を維持しながら、観測された部分だけに conditioning します。

```text
latent full grid
  X × T
   ↓
Kronecker prior
   ↓
observed entriesのみconditioning
```

という考え方です。

---

## 11.25 pathwise conditioning

LatentKroneckerGP は posterior sampling に pathwise conditioning を利用します。

概念的には

```text
1. prior function sample を生成
2. 観測とのずれを計算
3. conditioning correction を加える
4. posterior sample を得る
```

です。

posterior covariance を巨大な dense matrix として完全に構築する方法とは異なります。

---

## 11.26 iterative linear solvers

LatentKroneckerGP は効率的な inference のため iterative linear system solver を使います。

BoTorch では

```python
with model.use_iterative_methods():
    ...
```

という context manager が用意されています。

学習と posterior inference で、この context を使うのが重要です。

---

## 11.27 fitting example

```python
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import LatentKroneckerGP

model = LatentKroneckerGP(
    train_X=train_X,
    train_T=train_T,
    train_Y=train_Y,
)

mll = model.make_mll()

with model.use_iterative_methods():
    fit_gpytorch_mll(mll)
```

です。

HOGP と異なり、こちらは `use_iterative_methods()` が重要です。

---

## 11.28 posterior example

```python
with model.use_iterative_methods():
    posterior = model.posterior(
        test_X,
        test_T,
    )
```

とします。

posterior mean は `test_X × test_T` のstructured responseになります。

---

## 11.29 robotorchan の raw data API

LatentKroneckerGP wrapper では

```python
model.raw_train_X
model.raw_train_T
model.raw_train_Y
```

を保持します。

特に `raw_train_T` がこのモデル固有です。

これらは constructor input の provenance であり、内部で broadcasting / masking / transforms が適用された現在状態とは区別します。

---

## 11.30 covariance modules

LatentKroneckerGP では input space と output-coordinate space に別々の covariance module を指定できます。

```python
covar_module_X
covar_module_T
```

です。

これにより、

```text
process variables X
  → Matérn kernel

wavelength T
  → RBF kernel
```

のような使い分けも可能です。

---

## 11.31 mean modules

同様に

```python
mean_module_X
mean_module_T
```

を持ちます。

実務上は default から始めることが多いですが、明確な基調トレンドがある場合には mean structure も検討できます。

---

# モデル比較

## 11.32 HigherOrderGP vs LatentKroneckerGP

| 観点 | HigherOrderGP | LatentKroneckerGP |
|---|---|---|
| 主用途 | tensor-valued output | X × T product-space output |
| output coordinate | latent / tensor axis | 明示的 `T` |
| 画像 | 得意 | T設計次第 |
| スペクトル | 可能 | 特に自然 |
| 時系列 | 可能 | 特に自然 |
| 新しい出力座標 | 基本的に軸shape依存 | `test_T` で自然 |
| missing outputs | 構造上制約あり | missing entries を意識した設計 |
| posterior sampling | Matheron's rule | pathwise conditioning |
| fitting | `_fast_solves` + torch optimizer | iterative methods |
| raw固有API | `raw_train_X`, `raw_train_Y` | `raw_train_X`, `raw_train_T`, `raw_train_Y` |

---

## 11.33 選択フロー

```text
1回の実験からstructured responseが得られる
        ↓
出力軸に物理的な座標 T がある？
  ├─ Yes
  │    ↓
  │   新しいTで補間したい / 欠損Tがある？
  │    ├─ Yes → LatentKroneckerGP
  │    └─ No  → LatentKroneckerGP または HOGP
  │
  └─ No
       ↓
       tensor shape 自体が重要？
        ├─ Yes → HigherOrderGP
        └─ No  → 通常の multi-output GP も検討
```

---

## 11.34 Multi-output GP との違い

Multi-output GP は

```text
y1, y2, ..., ym
```

を複数outputとして扱います。

Structured Output GP はそれらの output が

```text
規則的なgrid
波長軸
時間軸
空間座標
```

を持つことを利用します。

したがって単に output 数が多いだけなら HOGP / LatentKroneckerGP が必ずしも必要ではありません。

---

## 11.35 KroneckerMultiTaskGP との違い

`KroneckerMultiTaskGP` も Kronecker structure を利用します。

ただし主な意味は

```text
X × task
```

です。

一方 Structured Output では

```text
X × output coordinate
```

または

```text
X × tensor axes
```

です。

時間や波長を単なる task label として扱うより、連続座標として扱う方が自然な場合があります。

---

## 11.36 ModelListGP との違い

出力位置ごとに独立 GP を作る方法なら

```python
ModelListGP(...)
```

も可能です。

しかし例えば100波長なら100個の独立GPになります。

これは

- 波長間相関を使わない
- model 数が増える
- smooth spectrum structure を利用できない

という欠点があります。

Structured Output GP は output間の構造を共有します。

---

# BOとの接続

## 11.37 structured response をそのまま最大化するわけではない

Bayesian Optimization では最終的に「何を最大化 / 最小化するのか」を決める必要があります。

structured output

\[
y(x,t)
\]

をそのまま scalar acquisition に渡すのではなく、通常は objective を定義します。

例えばスペクトルなら

- target wavelength の強度
- peak height
- peak position
- 積分面積
- target spectrum との距離
- 複数bandの比

などです。

---

## 11.38 objective function の例

スペクトル `Y` に対して

```python
def objective(Y):
    return Y[..., target_idx]
```

なら target wavelength 最大化です。

積分面積なら概念的に

```python
def objective(Y):
    return Y.mean(dim=-1)
```

です。

target spectrum に近づけるなら

```python
def objective(Y):
    return -((Y - target) ** 2).mean(dim=-1)
```

とできます。

---

## 11.39 posterior sample 上で objective を評価する

structured posterior から sample

\[
Y^{(s)}(x)
\]

を取り、各sampleに objective

\[
g(Y^{(s)}(x))
\]

を適用すれば、structured uncertainty を scalar objective uncertainty に伝播できます。

```text
structured posterior
      ↓ sample
spectrum / image sample
      ↓ objective
scalar score sample
      ↓
MC acquisition
```

です。

---

## 11.40 non-linear objective で特に有効

例えば

```text
maximum peak
peak position
threshold exceedance area
spectrum similarity
```

のような nonlinear functional は posterior mean にだけ適用するより、posterior sample ごとに評価する方が uncertainty を自然に反映できます。

---

## 11.41 Active Learning

目的が最適化ではなく structured response 全体の精度向上なら Active Learning として考えます。

例えば

```text
全波長の予測誤差を減らしたい
画像全体のuncertaintyを減らしたい
未知時間領域を学びたい
```

という目的です。

この場合は integrated variance や information gain のような基準が自然です。

---

# 実務上の注意

## 11.42 output normalization

structured output は位置によってscaleが異なることがあります。

例えばスペクトルで

```text
一部band: 0〜1
別band: 0〜1000
```

なら、scaleの大きい領域が fitting を支配する可能性があります。

BoTorch の HOGP は structured output 用の outcome transform を持ち、default transformも利用します。

物理的意味を壊さない範囲で scale を揃えることを検討します。

---

## 11.43 input normalization

通常のGPと同様、`X` は unit cube 近辺にnormalizeするのが基本です。

`LatentKroneckerGP`では `T` のscaleもkernel lengthscaleと関係するため、

```text
時間: 0〜100000
波長: 400〜800
```

のような場合は適切なスケーリングを検討します。

---

## 11.44 出力gridが大きすぎる場合

Kronecker structure を使っても、出力gridが極端に大きいと計算負荷は大きくなります。

例えば高解像度画像なら

- downsampling
- PCA / latent representation
- autoencoder
- DKL
- patch-based model

なども候補です。

Structured GP は「構造化すれば無限に大きい出力を扱える」わけではありません。

---

## 11.45 PCAとの違い

PCAで output を低次元latent vectorに圧縮してGPを作る方法もあります。

```text
Y
 ↓ PCA
z
 ↓ GP
z_hat
 ↓ inverse PCA
Y_hat
```

これは有力なbaselineです。

一方 HOGP / LatentKroneckerGP は output structure を GP covariance の中で直接扱います。

---

## 11.46 PCAが向く場合

- output dimension が非常に大きい
- 少数principal componentsで十分説明できる
- grid geometryよりglobal latent variationが重要

なら PCA + GP は有力です。

---

## 11.47 HOGPが向く場合

- tensor axes自体が重要
- axisごとのseparable correlationが期待できる
- 画像やgrid responseを直接扱いたい
- Kronecker structureが妥当

なら HOGP が候補です。

---

## 11.48 LatentKroneckerGPが向く場合

- output coordinate `T` に物理的意味がある
- `T` 上でinterpolationしたい
- missing output entries がある
- スペクトル・時系列・空間profile
- product-space covarianceが自然

なら LatentKroneckerGP が候補です。

---

## 11.49 separability の診断

Kronecker model の大きな仮定は separability です。

例えば

\[
k((x,t),(x',t'))=k_X(x,x')k_T(t,t')
\]

が妥当かを考えます。

もし

```text
あるX領域ではT方向が非常に滑らか
別X領域ではT方向が急激
```

なら単純なseparable kernelでは不足する可能性があります。

---

## 11.50 non-stationarity

スペクトルや時系列では局所的に異なるsmoothnessを持つ場合があります。

例えば

```text
baseline領域: smooth
phase transition周辺: sharp
```

です。

stationary RBF / Matérn kernel だけでは表現しづらい場合、kernel拡張やinput warpingを検討します。

---

## 11.51 noise structure

structured output では noise も位置依存する場合があります。

```text
特定波長だけS/Nが悪い
時間後半だけ測定誤差が大きい
画像端部でnoiseが増える
```

標準 Gaussian likelihood が仮定する単純noiseでは不足する可能性があります。

---

## 11.52 missingness mechanism

LatentKroneckerGP が missing entries を扱えるとしても、missingnessが informative なら注意が必要です。

例えば

```text
測定不能だった箇所だけmissing
```

なら missingness 自体が応答に依存している可能性があります。

単に欠損を無視するだけではbiasが生じます。

---

## 11.53 extrapolation in T

`LatentKroneckerGP` は新しい `T` でposteriorを評価できますが、interpolationとextrapolationは別です。

```text
train T: 400〜800 nm
predict T: 450 nm
  → interpolation

predict T: 1200 nm
  → extrapolation
```

後者はkernel priorへの依存が大きくなります。

---

## 11.54 diagnostic plots

structured output model では scalar RMSE だけでなく、構造ごとの診断が重要です。

例えば

- wavelength-wise RMSE
- time-wise RMSE
- image residual map
- peak position error
- integrated area error
- uncertainty calibration by coordinate

です。

---

## 11.55 holdout design

random element holdout だけでなく、目的に応じて holdout を変えます。

```text
new X prediction
  → Xごとholdout

new T interpolation
  → T座標をholdout

missing-grid robustness
  → X×T entryをmask
```

です。

---

## 11.56 robotorchan Notebook

実行例は

```text
examples/notebooks/11_structured_output_gp.ipynb
```

です。

Notebookでは合成スペクトルを使い、

1. `train_X`, `train_T`, `train_Y` を生成
2. `HigherOrderGP` を学習
3. `LatentKroneckerGP` を学習
4. posterior spectrum を比較
5. `LatentKroneckerGP` を新しい `test_T` で評価

します。

---

## 11.57 HigherOrderGP example

```python
from botorch.fit import fit_gpytorch_mll_torch
from linear_operator.settings import _fast_solves
from robotorchan.models import HigherOrderGP

model = HigherOrderGP(
    train_X=train_X,
    train_Y=train_Y,
)

mll = model.make_mll()

with _fast_solves(True):
    fit_gpytorch_mll_torch(mll)

posterior = model.posterior(test_X)
```

---

## 11.58 LatentKroneckerGP example

```python
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import LatentKroneckerGP

model = LatentKroneckerGP(
    train_X=train_X,
    train_T=train_T,
    train_Y=train_Y,
)

mll = model.make_mll()

with model.use_iterative_methods():
    fit_gpytorch_mll(mll)
    posterior = model.posterior(test_X, test_T)
```

---

## 11.59 raw data

HOGP:

```python
model.raw_train_X
model.raw_train_Y
model.raw_train_Yvar  # None
```

LatentKroneckerGP:

```python
model.raw_train_X
model.raw_train_T
model.raw_train_Y
model.raw_train_Yvar  # None
```

です。

raw data は constructor input provenance であり、内部 transformed state ではありません。

---

## 11.60 よくある誤解

### 「出力数が多ければ HOGP」

必ずしもそうではありません。出力に規則的なtensor structureがあるかが重要です。

### 「LatentKroneckerGP の T は task ID」

必ずしもtask labelではありません。時間・波長・位置など連続座標として使えます。

### 「Kronecker を使えばどんな巨大データでも軽い」

違います。構造による効率化はありますが、grid sizeやtraining sizeが増えれば計算負荷も増えます。

### 「structured outputをそのままEIに入れればよい」

通常はstructured responseからscalar / low-dimensional objectiveを定義します。

### 「新しいTで予測できるなら外挿も安全」

違います。training range外ではpriorへの依存が大きくなります。

---

## 11.61 モデル選択表

| 問題 | 推奨 |
|---|---|
| scalar response | `SingleTaskGP` |
| 少数の独立outputs | `ModelListGP` |
| 関連tasks | `MultiTaskGP` / `KroneckerMultiTaskGP` |
| 画像・tensor output | `HigherOrderGP` |
| 波長profile | `LatentKroneckerGP` |
| 時系列profile | `LatentKroneckerGP` |
| spatial profile with coordinates | `LatentKroneckerGP` |
| 新しいoutput coordinateで補間 | `LatentKroneckerGP` |
| structured outputだが非常に巨大 | PCA / latent model / DKLも検討 |

---

## 11.62 数学的な全体像

Structured Output GP の基本は

\[
f(x,t)\sim\mathcal{GP}(m,k)
\]

と

\[
k((x,t),(x',t'))
=
k_X(x,x')k_T(t,t')
\]

です。

grid全体では

\[
K
=
K_X\otimes K_T
\]

となります。

HOGP では複数 output axes に拡張して

\[
K
=
K_X
\otimes K_1
\otimes \cdots
\otimes K_p
\]

を利用します。

---

## 11.63 BOとの全体像

```text
X
 ↓
structured-output surrogate
 ↓
Y(x, T) posterior
 ↓ posterior sampling
structured response samples
 ↓ objective g(Y)
scalar utility samples
 ↓
acquisition function
 ↓
next X
```

です。

この分解を理解すると、model と objective と acquisition の責任範囲が明確になります。

---

## 11.64 まとめ

Structured Output GP は、出力を単なる多数のscalar valuesとしてではなく、**出力側の構造を利用してモデル化するGP**です。

robotorchan では主に

```python
HigherOrderGP
LatentKroneckerGP
```

を使います。

重要点は次の通りです。

1. HOGP は tensor-valued output の各軸に Kronecker structure を利用する
2. LatentKroneckerGP は `X × T` のproduct-space model
3. `T` は時間・波長・位置などの明示的なoutput coordinateとして使える
4. LatentKroneckerGP は新しい `T` でposteriorを評価できる
5. LatentKroneckerGP はmissing output entriesを意識した設計
6. HOGP fittingでは specialized fast solve と torch optimizer が重要
7. LatentKroneckerGP では iterative methods が重要
8. Kronecker efficiency は separable covariance assumption と引き換え
9. BOではstructured responseから何を最適化するかobjectiveを明示する
10. posterior sample上でobjectiveを評価するとstructured uncertaintyを伝播できる

次章では、条件付き探索空間・異種task・contextを扱う [Hierarchical / Contextual GP](12_hierarchical_contextual_gp.md) を説明します。

## 参考

- Zhe, Xing, Kirby, *Scalable High-Order Gaussian Process Regression*, AISTATS 2019
- Maddox et al., structured GP posterior sampling / Bayesian optimization work
- Lin et al., *Scaling Gaussian Processes for Learning Curve Prediction via Latent Kronecker Structure*, 2024
- Lin et al., *Scalable Gaussian Processes with Latent Kronecker Structure*, ICML 2025
- BoTorch `HigherOrderGP`
- BoTorch `LatentKroneckerGP`
- [Gaussian Process](02_gaussian_process.md)
- [Kernel](03_kernel.md)
- [Multi-task / Multi-output](07_multitask_multioutput.md)
