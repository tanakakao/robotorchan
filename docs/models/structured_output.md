# Structured output

## HigherOrderGP

`HigherOrderGP` と `MixedHigherOrderGP` は画像、スペクトルグリッド、空間分布など tensor-valued output を扱います。出力テンソルの軸構造そのものに意味がある場合に候補になります。

## LatentKroneckerGP

`LatentKroneckerGP` と `MixedLatentKroneckerGP` は、入力 `X` と時間・波長・位置などの出力座標 `T` を明示的に扱います。新しい `T` 位置での補間・予測が必要な問題で特に自然です。

`HigherOrderGP` と `LatentKroneckerGP` は「出力次元が多い」という理由だけで交換可能ではありません。出力軸を tensor structure として扱うか、明示的座標として扱うかで選択します。

Notebook: [Structured output](../../examples/notebooks/11_structured_output_gp.ipynb)  
Theory: [Structured Output](../theory/11_structured_output.md)


## API examples and fitting notes

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

specialized Kronecker solve を有効にし、通常の SciPy L-BFGS-B ベースの `fit_gpytorch_mll()` ではなく、PyTorch
optimizer を使う `fit_gpytorch_mll_torch()` が推奨されます。

```python
from botorch.fit import fit_gpytorch_mll_torch
from linear_operator.settings import _fast_solves

mll = model.make_mll()

with _fast_solves(True):
    fit_gpytorch_mll_torch(mll)
```

これはHOGP固有のstructured linear algebraとupstreamの推奨fitting pathを
維持するためです。optimizer選択を一般のExact GPと機械的に同一視しないようにします。

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

### Mixed LatentKroneckerGP

robotorchanには `MixedLatentKroneckerGP` もあります。

~~~text
X factor
  → continuous + categorical design variables

T factor
  → time / wavelength / task-like output coordinates
~~~

Mixed版が置き換えるのは **X側のcovariance** です。`cat_dims` は `train_X` の
categorical design columnsを表し、`train_T` の列をcategorical指定するものではありません。

この分離は `MixedHigherOrderGP` と同じく、「design-input structure」と
「structured-output structure」を別の責務として扱うためのものです。

# モデル比較

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


## Additional API examples and usage notes

## 11.10 HOGP の latent representation

HOGP では各出力軸について latent coordinates を持ち、それらの上で kernel covariance を構成します。

robotorchan wrapper の constructor には

```python
num_latent_dims
learn_latent_pars
latent_init
```

があります。

つまり出力axisを表すlatent coordinatesを用いてaxis covarianceを構成し、設定に応じて
そのrepresentation自体も学習できます。これは波長などの既知の物理座標を
そのまま入力することとは異なります。

---

## 11.12 `learn_latent_pars`

```python
learn_latent_pars = True
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

### Surrogateとobjectiveを分ける

Structured Output GPが答えるのは、

~~~text
このXで、response全体はどう分布するか？
~~~

です。一方BOが必要とするのは、

~~~text
そのresponseの何を良いと定義するか？
~~~

です。

したがってHOGPやLatentKroneckerGPを選んだだけでは最適化問題は完成しません。
peak、積分値、target spectrumとの距離などをObjective / PosteriorTransform側で
定義し、必要ならposterior sample上で非線形functionalを評価します。

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
