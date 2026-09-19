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
objective = GenericMCObjective(lambda samples, X=None: samples[..., 10])
```

多目的の場合も同様です。

```python
objective = IdentityMCMultiOutputObjective(outcomes=[2, 7, 10])
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

## 9. ALEBO と ReducedGP の関係

ALEBO は `PCAGP` や AE 系のようなデータ駆動 reducer ではなく、BO の探索空間そのものを
固定された低次元線形 embedding へ制限する search strategy です。robotorchan では
`ALEBOStrategy` が ambient-space bounds と embedding / feasible polytope を管理し、
`ALEBOGP` は embedded coordinates 上で Mahalanobis geometry を学習します。

したがって、ALEBO を `ReducedGP` / `ReducedMultiTaskGP` の reducer として組み込みません。
両者を機械的に合成すると、候補生成に使う embedding と surrogate 内部の reducer という
二重の座標変換が生じ、どの空間で bounds・pending points・acquisition optimization を扱うかが
曖昧になります。

基本的な使い分けは次の通りです。

- ambient dimension は大きいが、目的関数が低次元の線形 subspace に依存すると仮定する:
  `ALEBOStrategy` + `ALEBOGP`
- 観測済み X から低次元表現を学習して通常の BO を行う: `ReducedGP` family
- 元特徴の少数だけが重要だと仮定する: SAAS family
- task 相関も同時に必要: `ReducedMultiTaskGP` family または SAAS MultiTask

現行 ALEBO は single-output の embedded surrogate を明示的なスコープとします。
`ALEBOMultiTaskGP`、`MixedALEBOGP`、`ReducedALEBOGP` のような名前だけの直積モデルは
追加しません。multi-task / mixed ALEBO は embedding、task/categorical structure、
feasible polytope、acquisition optimization を一体として定義できる場合に独立拡張として扱います。

## 9. SAAS と次元削減の関係

SAAS は PCA / PLS / AE のように入力を低次元座標へ射影するモデルではありません。
高次元の元入力空間を保ったまま、lengthscale に sparsity prior を置いて重要な入力次元を
選択的に使う方針です。そのため、次元削減モデルとは相補的ですが、同じ問題への別アプローチです。

robotorchan では次を基本方針とします。

- 少数の元特徴だけが効くと考えられる高次元 X: `SaasFullyBayesianSingleTaskGP`
- 同じ sparse high-dimensional X を複数 task で共有する long-format データ:
  `SaasFullyBayesianMultiTaskGP`
- 低次元 manifold / 線形部分空間を仮定したい X: `PCAGP`, `PLSGP`, AE/VAE 系
- 高次元 X と task covariance の両方を扱いつつ明示的に圧縮したい:
  `ReducedMultiTaskGP` / `ReducedKroneckerMultiTaskGP` 系

SAAS の前段へ reducer を機械的に追加した `ReducedSaas...` クラスは作りません。SAAS prior が
作用する座標が reducer 後の latent coordinate に変わるため、「元特徴の sparsity」という
SAAS の解釈が失われるからです。必要な場合は独立した統計モデルとして設計・検証します。

また、BoTorch の fully Bayesian multi-task SAAS は long-format task-feature 型です。
現時点では `SaasFullyBayesianKroneckerMultiTaskGP` に相当するモデルを新設しません。
block-design の `KroneckerMultiTaskGP` と SAAS を名前だけで合成せず、対応する inference / kernel
設計が明確になった場合に別機能として追加します。

## 9. MultiTask と output reduction の関係

Phase 9 の監査では、output reduction と MultiTask / KroneckerMultiTask を単純に合成する
専用モデルは追加しません。

理由は、両者が同じ出力軸に対して異なる意味を持つためです。

- `OutputPCAGP` / `OutputPLSGP`: 元の高次元 Y を latent component へ圧縮し、latent 出力を独立な GP 出力として扱う。
- `MultiTaskGP`: task identity を明示的な入力特徴として持ち、task covariance を学習する。
- `KroneckerMultiTaskGP`: Y の各列を task として保持し、列間の task covariance を学習する。

したがって、例えば `OutputPCAKroneckerMultiTaskGP` のようなモデルを機械的に追加すると、
「PCA component を task とみなす」のか「元の task を PCA で混合する」のかが曖昧になります。
後者では task covariance の意味も元の task 空間から latent component 空間へ変化します。

高次元 Y を圧縮すること自体が目的なら `OutputPCAGP` / `OutputPLSGP` を使い、
各出力を task として相関までモデル化したいなら `MultiTaskGP` または
`KroneckerMultiTaskGP` を使います。入力 X も高次元なら、Phase 2--8 で追加した
`ReducedMultiTaskGP` / `ReducedKroneckerMultiTaskGP` 系で **入力側だけ** を圧縮します。

つまり、robotorchan では次を別のモデリング選択として扱います。

```text
high-dimensional X + task-correlated Y
    -> ReducedMultiTaskGP / ReducedKroneckerMultiTaskGP family

ordinary X + compressible high-dimensional Y
    -> OutputPCAGP / OutputPLSGP

structured Y with coordinates / tensor axes
    -> LatentKroneckerGP / HigherOrderGP
```

output reduction と task covariance の同時利用は、明確な統計モデルと posterior 復元則を
定義できる場合にのみ将来の独立機能として追加します。単なるクラス直積としては追加しません。

## 9. Reducer の lifecycle

入力・出力 reducer はモデル構築時に学習し、その後は同じ基底を固定して使います。

```text
model construction
    ↓
reducer.fit(...)
    ↓
BO iteration 1, 2, 3, ...
    ↓
同じ reducer.transform(...) を利用
```

BO iteration ごとに PCA / PLS 基底を学習し直すと latent 座標系自体が変化し、posterior、pending point、fantasy、acquisition landscape の意味が変わります。そのため通常の逐次 BO では reducer を再 fit しません。

`condition_on_observations()` でも reducer は再学習せず、新しい `X` / `Y` を既存の基底へ投影します。

## 10. 変換順序

現行の `ReducedGP` では reducer と BoTorch transform の順序は次の通りです。

入力側:

```text
raw X
  ↓ input reducer
latent X
  ↓ input_transform
GP
```

出力側:

```text
raw Y
  ↓ output reducer
latent Y
  ↓ outcome_transform
GP
```

posterior では逆方向に戻し、公開される出力は元の `Y` 空間です。

したがって `Normalize` や `Standardize` を指定する場合、それらは reducer 後の latent 次元に対して設定します。元の物理量空間でのスケーリングを PCA 前に適用したい場合は、学習データを事前にスケーリングするか、将来の raw-space preprocessing 層を利用する設計とします。

## 11. 保存と再利用

Reducer の学習済み基底と fit metadata は PyTorch の `state_dict` に含まれます。

```python
state = model.state_dict()
torch.save(state, "model.pt")
```

Reducer 単体でも round-trip できます。

```python
from robotorchan.models.reduction import PCAInputReducer

reducer = PCAInputReducer(n_components=5)
reducer.fit(train_X)

torch.save(reducer.state_dict(), "pca_reducer.pt")

restored = PCAInputReducer(n_components=5)
restored.load_state_dict(torch.load("pca_reducer.pt", weights_only=True))

Z = restored.transform(X)
```

`is_fitted`、`input_dim`、`output_dim` も復元されるため、load 後に `fit()` を呼び直す必要はありません。

モデル全体を復元する場合は、同じモデル構成を生成してから `load_state_dict()` を使う PyTorch 標準の方式を前提とします。

## 12. 現時点の制約

Output reduction では次の機能を明示的に制限しています。

- explicit `train_Yvar`
- Tensor-valued `observation_noise`
- output reduction と `output_indices` の直接併用
- output reduction と `posterior_transform` の直接併用

理由は、これらを latent output 空間へ単純に渡すと元出力空間とは異なる意味になるためです。

多目的・制約付き BO では、MC posterior sample が元の `Y` 空間へ復元された後に `GenericMCObjective`、`MCMultiOutputObjective`、constraint を適用する使い方を基本とします。
