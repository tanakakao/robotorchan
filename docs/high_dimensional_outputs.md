# 高次元出力モデルの選び方

robotorchan では、高次元な目的変数をすべて同じ方法で扱うのではなく、**出力が高次元である理由**に応じてモデルを選びます。

## 1. まず区別すること

高次元出力には大きく4種類あります。

| 状況 | 第一候補 | 考え方 |
|---|---|---|
| 多数の連続出力を低次元潜在表現へ圧縮したい | `OutputPCAGP` / `OutputPLSGP` | 出力を latent space に圧縮して GP を学習し、posterior を元の Y 空間へ復元 |
| 出力が画像・2次元マップ・多軸スペクトルなど tensor 構造を持つ | `HigherOrderGP` | 出力テンソルの各軸の構造を直接モデル化 |
| 波長・時間・位置など出力軸そのものに座標がある | `LatentKroneckerGP` | 入力 X と出力座標 T の積空間をモデル化 |
| 同じ X で複数特性を毎回すべて測定する | `KroneckerMultiTaskGP` | block design の複数 task として相関を学習 |

これらは代替関係になる場合もありますが、前提としている出力構造が異なります。

## 2. OutputPCAGP / OutputPLSGP

高次元ベクトル出力を少数の latent component に圧縮する方法です。

```text
Y_original [n, m]
    ↓ output reducer
Y_latent   [n, k]
    ↓ GP
posterior latent
    ↓ inverse transform
posterior original [*, m]
```

典型例:

- 100波長のスペクトルを5〜10成分へ圧縮する
- 多数の相関した品質特性を少数成分へ圧縮する
- 多目的 BO で surrogate の出力次元を下げたい

```python
from robotorchan.models import OutputPCAGP

model = OutputPCAGP(
    train_X=train_X,
    train_Y=train_Y,
    n_components=5,
)

posterior = model.posterior(X)
# posterior.mean[..., -1] は元の出力次元 m
```

### PCA component を目的関数にしない

PCA component は元の目的変数ではありません。BO の objective は復元後の元 Y 空間で定義します。

```python
objective = GenericMCObjective(
    lambda samples, X=None: samples[..., 10]
)
```

多目的の場合も同様です。

```python
objective = IdentityMCMultiOutputObjective(
    outcomes=[2, 7, 10]
)
```

robotorchan の output-reduced GP は posterior sample を元の出力空間へ復元してから acquisition function に渡します。

## 3. HigherOrderGP

`HigherOrderGP` は出力を flatten して圧縮するモデルではありません。

例えば、

```text
train_X: [n, d]
train_Y: [n, height, width]
```

のような tensor-valued output を、そのテンソル構造を保ったままモデル化します。

典型例:

- 2次元分布
- 画像状の応答
- 複数軸を持つスペクトル
- 空間場

出力テンソルの軸そのものに意味がある場合は、単純な PCA 圧縮より `HigherOrderGP` の方が自然です。

## 4. LatentKroneckerGP

`LatentKroneckerGP` は出力軸に明示的な座標 `T` を持たせるモデルです。

```text
train_X: [n, d]
train_T: [m, d_t]
train_Y: [n, m]
```

例えばスペクトルなら、

```text
X = 材料組成・プロセス条件
T = 波長
Y = 各波長での強度
```

と解釈できます。

典型例:

- 波長
- 時間
- 温度軸
- 空間座標
- 周波数

`T` に意味があり、未観測の `T` での補間や出力軸方向の構造を学習したい場合に適しています。

BoTorch の `LatentKroneckerGP` が持つ missing-output 処理もそのまま利用します。

## 5. KroneckerMultiTaskGP

すべての入力条件で、すべての task / 特性が観測される block design 向けです。

```text
train_X: [n, d]
train_Y: [n, m]
```

例:

- 各実験で強度・延性・密度・熱伝導率をすべて測定
- 各工程条件で複数センサ指標を必ず取得

複数出力間の task covariance を学習したい場合に自然です。

一方、出力が100〜1000次元あるだけで task として扱う意味が薄い場合は、`OutputPCAGP` の方が軽量になることがあります。

## 6. 選択フロー

```text
高次元 Y
  |
  +-- Y に tensor 構造がある？
  |      |
  |      +-- Yes -> HigherOrderGP
  |
  +-- 出力軸に時間・波長・位置など明示座標 T がある？
  |      |
  |      +-- Yes -> LatentKroneckerGP
  |
  +-- 各出力を task と考える意味があり、全Xで全taskを観測？
  |      |
  |      +-- Yes -> KroneckerMultiTaskGP
  |
  +-- 多数の相関出力を圧縮したい？
         |
         +-- Yes -> OutputPCAGP / OutputPLSGP
```

## 7. 多目的 BO との関係

高次元出力と many-objective BO は別問題です。

`OutputPCAGP` で surrogate の latent output を5次元へ減らしても、復元後の100出力すべてを真の目的として qEHVI に渡せば、hypervolume 側の計算量は依然として高くなります。

多数の出力すべてが真の目的の場合は、次も検討します。

- scalarization
- ParEGO / qLogNParEGO
- preference / ROI
- objective subset selection
- reference direction

一方、スペクトルや画像などの高次元応答から最終的に数個のスカラー指標を計算する場合は、posterior sample を元空間へ復元した後にその指標を計算する構成が自然です。

## 8. robotorchan での位置づけ

`ReducedGP` と structured-output model は混在させません。

```text
ReducedGP family
- PCAGP
- PLSGP
- RandomProjectionGP
- OutputPCAGP
- OutputPLSGP

Structured / correlated output family
- HigherOrderGP
- LatentKroneckerGP
- KroneckerMultiTaskGP
- MultiTaskGP
```

`ReducedGP` は reducer composition を担当し、structured-output model はそれぞれ BoTorch 本来の covariance structure を保持します。
