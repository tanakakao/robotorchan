# robotorchan モデル概要・使い所ガイド

> 対象: `robotorchan` main ブランチ（2026-09-12 時点）  
> 対応方針: BoTorch の既存モデルを薄くラップし、`raw_*` データ保持や `make_mll()` など robotorchan 共通 API を追加する。

---

## 1. このドキュメントの目的

robotorchan には、通常の Gaussian Process だけでなく、混合変数、マルチタスク、Multi-Fidelity、高次元、Preference、構造化出力、階層探索空間、Contextual BO などに対応するモデルが含まれる。

このドキュメントでは各モデルについて、

- 何を表現するモデルか
- どのようなデータに向いているか
- どのような場面で使うべきか
- 似たモデルとの使い分け
- 主な注意点

を整理する。

---

# 2. まずどのモデルを選ぶか

## 2.1 モデル選択早見表

| 状況 | 第一候補 | コメント |
|---|---|---|
| 通常の連続変数 + 単一目的 | `SingleTaskGP` | まずここから始める基本モデル |
| 連続変数 + カテゴリ変数 | `MixedSingleTaskGP` | 材料種、装置種、製法などカテゴリを含む場合 |
| 低精度/高精度データを併用 | `SingleTaskMultiFidelityGP` | シミュレーション精度や試験コストが複数段階ある場合 |
| 複数タスクで情報共有したい | `MultiTaskGP` | タスクごとの観測点が異なっていても扱いやすい |
| 全タスクを同じ X で測定 | `KroneckerMultiTaskGP` | block design のマルチタスク |
| 出力ごとに独立モデルでよい | `ModelListGP` | 相関をモデル化しない複数出力 |
| データ量が多く Exact GP が重い | `SingleTaskVariationalGP` | inducing point による近似 GP |
| 「A と B のどちらが良いか」の比較データ | `PairwiseGP` | Preference / ranking 学習 |
| 高次元・少数データ | `SaasFullyBayesianSingleTaskGP` | NUTS を使う本格 SAAS |
| 高次元・高速寄り | `AdditiveMapSaasSingleTaskGP` / `EnsembleMapSaasSingleTaskGP` | MAP-SAAS 系 |
| 外れ値や不要特徴の影響を抑えたい | `RobustRelevancePursuitSingleTaskGP` | relevance pursuit を利用 |
| 高次元だが低次の加法構造を期待 | `OrthogonalAdditiveGP` | additive structure を仮定 |
| 画像・曲線・テンソルなど構造化出力 | `HigherOrderGP` | tensor-valued output |
| 時間/位置など潜在的な軸を持つ出力 | `LatentKroneckerGP` | `train_T` を別に持つ構造化出力 |
| 条件によって有効変数が変わる | `HierarchicalConditionalKernelGP` | 階層・条件付き探索空間 |
| 階層探索 + マルチタスク | `HierarchicalConditionalKernelMultiTaskGP` | 上記の multitask 版 |
| タスクごとに X の構造が異なる | `HeterogeneousMTGP` | heterogeneous multitask |
| Context ごとに入力を分解できる | `SACGP` | Structural Additive Contextual GP |
| Context 間の潜在表現も学習したい | `LCEAGP` | latent context embedding + additive |
| Context/Task を埋め込みで扱う multi-output | `LCEMGP` | latent context embedding multi-output |

---

# 3. robotorchan 共通 API

robotorchan の多くの supervised model wrapper は、BoTorch の予測挙動をそのまま利用しつつ、以下の共通情報を保持する。

```python
model.raw_train_X
model.raw_train_Y
model.raw_train_Yvar
model.raw_data
model.raw_data_names

model.supports_mll
model.make_mll()
```

`raw_*` は **コンストラクタに渡した生データのスナップショット** である。

重要なのは、`condition_on_observations()` や `fantasize()` 後の最新学習状態を表すものではない点である。最新状態は BoTorch ネイティブの `train_inputs` や `train_targets` を参照する。

基本的な Exact GP では以下の形で学習できる。

```python
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import SingleTaskGP

model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
mll = model.make_mll()
fit_gpytorch_mll(mll)
```

---

# 4. 標準モデル

## 4.1 `SingleTaskGP`

### 概要

最も標準的な単一タスク GP。

### 向いているケース

- 単一目的の回帰
- 連続変数中心の Bayesian Optimization
- データ数が数十〜数百程度
- まずベースラインを作りたい場合

### 使い所

robotorchan で迷った場合の第一候補。

材料開発であれば、

- 温度
- 圧力
- 時間
- 組成比
- 製造条件

などを連続変数として性能を予測する通常の BO に適する。

### 注意点

高次元では ARD lengthscale の推定が難しくなりやすい。数十次元以上で有効変数が少ないと考えられる場合は SAAS 系を検討する。

---

## 4.2 `MixedSingleTaskGP`

### 概要

連続変数とカテゴリ変数が混在する single-task GP。

### 向いているケース

例えば次のような探索空間。

```text
temperature : continuous
pressure    : continuous
material    : categorical
machine     : categorical
```

### 使い所

- 装置番号
- 原料種
- 製法
- 触媒種
- 処理方式

のようなカテゴリ条件を含む BO。

### `SingleTaskGP` との違い

カテゴリを単純な連続値として扱うのではなく、カテゴリ次元としてモデル化する。

### 注意点

カテゴリ値を「0, 1, 2」という連続変数として扱う SingleTaskGP の代替ではない。カテゴリ変数は `cat_dims` で明示する。

---

# 5. Multi-Fidelity

## 5.1 `SingleTaskMultiFidelityGP`

### 概要

異なる fidelity のデータをまとめて利用する Gaussian Process。

fidelity は例えば、

- シミュレーションの粗さ
- 計算反復回数
- メッシュ解像度
- 実験の簡易測定 / 精密測定
- 小型試験 / 実機試験

などを表す。

### 使い所

高 fidelity の測定が高コストで、低 fidelity のデータを安く大量に取得できる場合。

```text
low fidelity simulation
        ↓
medium fidelity simulation
        ↓
high fidelity experiment
```

の情報を共有しながら最適化する。

### 主な指定

- `iteration_fidelity`
- `data_fidelities`

### 通常 GP との違い

通常 GP は fidelity を単なる入力変数として扱うが、Multi-Fidelity GP は fidelity の構造をカーネル側で明示的に扱う。

### 注意点

「測定条件の一つとして fidelity 値がある」だけでは利用メリットが出ない。低 fidelity と高 fidelity の間に相関があることが前提。

---

# 6. Multi-task / Multi-output

## 6.1 `MultiTaskGP`

### 概要

複数タスク間の相関を学習する GP。

BoTorch の long-format representation を使う。

```text
X1, task=0 -> y
X2, task=0 -> y
X3, task=1 -> y
X4, task=2 -> y
```

### 向いているケース

- 材料 A / B / C を別タスクと考える
- 装置ごとのモデルを共有したい
- 測定方法ごとの出力を関連付けたい
- 低コスト評価と高コスト評価を「タスク」として扱いたい

### 強み

タスク間に相関があれば、データの少ないタスクが他タスクから情報を借りられる。

### 注意点

単純に複数の目的変数があるから MultiTaskGP、というわけではない。

「同一現象を異なる task として共有できる」と考えられる場合に使う。

---

## 6.2 `KroneckerMultiTaskGP`

### 概要

すべての X で全タスクが観測される **block design** 向け MultiTask GP。

```python
train_X.shape == [n, d]
train_Y.shape == [n, m]
```

### 向いているケース

各実験条件 X について、

```text
property_A
property_B
property_C
...
```

を毎回すべて測定している場合。

### `MultiTaskGP` との使い分け

| | `MultiTaskGP` | `KroneckerMultiTaskGP` |
|---|---|---|
| 入力形式 | long format | block design |
| 全タスク同じ X 必須 | No | Yes |
| 欠測タスク | 扱いやすい | 基本的に block design 前提 |
| タスク数が多い | 可 | Kronecker 構造が有効な場合あり |

**全タスクが同じ X で観測されているなら `KroneckerMultiTaskGP` を検討する。**

---

## 6.3 `ModelListGP`

### 概要

複数の独立 GP を一つの BoTorch Model としてまとめるコンテナ。

### 向いているケース

複数出力があるが、出力間相関をモデル化する必要がない場合。

例えば、

```text
strength -> GP1
cost     -> GP2
density  -> GP3
```

### MultiTaskGP との違い

`ModelListGP` は原則として各モデルが独立。

`MultiTaskGP` は task covariance を通じて情報共有する。

### Multi-objective BO では

まず `ModelListGP` を使う構成は非常に自然。

複数目的だから必ず multitask にする必要はない。

---

# 7. 大規模データ

## 7.1 `SingleTaskVariationalGP`

### 概要

inducing points を用いる variational GP。

Exact GP の計算量が問題になるデータ量で利用する。

### 向いているケース

- 数千点以上のデータ
- Exact GP の学習が重い
- minibatch 学習を行いたい

### 学習

Exact GP の `ExactMarginalLogLikelihood` ではなく `VariationalELBO` を使用する。

```python
model = SingleTaskVariationalGP(
    train_X=train_X,
    train_Y=train_Y,
    inducing_points=40,
)

mll = model.make_mll()
```

minibatch の場合は、`num_data` に全データ件数を指定する。

### 注意点

データが数十点程度なら Exact GP の方が単純かつ有力なことが多い。

---

# 8. Preference Learning

## 8.1 `PairwiseGP`

### 概要

絶対値の目的値ではなく「どちらが好ましいか」という pairwise comparison を学習する。

```text
A > B
C > D
A > D
```

### 向いているケース

- 人間の官能評価
- デザイン評価
- 見た目の好み
- 定量スコア化しにくい品質
- A/B comparison

### データ

通常の `train_Y` ではなく、

- `datapoints`
- `comparisons`

を使う。

### 注意点

比較結果を 0/1 の通常回帰に変換して `SingleTaskGP` に入れる問題ではない。

---

# 9. 高次元モデル

## 9.1 `SaasFullyBayesianSingleTaskGP`

### 概要

SAAS prior を使う fully Bayesian GP。

高次元空間の中で、本当に効いている変数が少数という sparse な構造を仮定する。

### 向いているケース

例えば、

```text
入力 50 次元
実際に重要なのは 3〜8 変数程度
データ 30〜100 点
```

のような場合。

### 強み

高次元少数データの BO で非常に有力。

### 注意点

NUTS / NumPyro による fully Bayesian fitting のため、通常 GP よりかなり重い。

`make_mll()` は使わず、

```python
from botorch.fit import fit_fully_bayesian_model_nuts
fit_fully_bayesian_model_nuts(model)
```

を使う。

---

## 9.2 `SaasFullyBayesianMultiTaskGP`

### 概要

上記 SAAS の multitask 版。

### 使い所

- 高次元
- 少数データ
- 複数タスク
- タスク間情報共有

を同時に扱いたい場合。

### 注意点

モデル・推論ともにかなり重い。必要性が明確な場合に選ぶ。

---

## 9.3 `AdditiveMapSaasSingleTaskGP`

### 概要

SAAS 的な shrinkage を MAP 推定で扱う高速寄りのモデル。

### 向いているケース

Fully Bayesian SAAS を使いたいが NUTS が重すぎる場合。

### 特徴

- SAAS の高次元適応性を狙う
- Fully Bayesian より計算を抑えやすい
- `num_taus` で複数の shrinkage scale を利用

---

## 9.4 `EnsembleMapSaasSingleTaskGP`

### 概要

複数の MAP-SAAS モデルを ensemble 的に扱うモデル。

### 使い所

単一 MAP 解だけでは不安定な場合に、複数の shrinkage scale を組み合わせたい場合。

### ざっくりした選択

```text
高速性優先
    ↓
AdditiveMapSaas
    ↓
EnsembleMapSaas
    ↓
Fully Bayesian SAAS
精密な不確実性優先
```

---

## 9.5 `OrthogonalAdditiveGP`

### 概要

目的関数が各変数や低次相互作用の「和」で近似できるという additive structure を利用する GP。

概念的には、

```text
f(x) ≈ f1(x1) + f2(x2) + ... + fij(xi, xj)
```

のような構造を仮定する。

### 向いているケース

- 入力次元が多い
- ただし複雑な高次相互作用は少ない
- 各因子の寄与を足し合わせる構造が期待できる

### `second_order=True`

二変数相互作用まで含めたい場合。

### 注意点

目的関数に強い高次相互作用がある場合は additive assumption が合わない。

---

## 9.6 `RobustRelevancePursuitSingleTaskGP`

### 概要

Relevance Pursuit を利用する robust GP。

### 向いているケース

- 一部観測が異常
- 外れ値的な挙動がある
- 通常 GP が少数点に引きずられやすい

### 注意点

単なる「高次元用モデル」として選ぶのではなく、robustness / relevance selection が必要なデータで使う。

---

# 10. 構造化出力

## 10.1 `HigherOrderGP`

### 概要

出力が scalar ではなくテンソル構造を持つ場合の GP。

例えば、

- 画像
- 2D map
- スペクトル
- 時系列曲線
- 空間分布

など。

### 例

```python
train_X.shape == [n, d]
train_Y.shape == [n, h, w]
```

### 使い所

各 X に対して、単一の値ではなく「構造全体」を予測したい場合。

### 注意点

単純に出力数が数個あるだけなら `ModelListGP` や multitask の方が扱いやすい場合が多い。

---

## 10.2 `LatentKroneckerGP`

### 概要

入力 `X` に加えて、出力側の座標 `T` を明示的に持つ構造化 GP。

```text
X : 実験条件
T : 時間 / 波長 / 空間座標
Y : X × T 上の応答
```

### 向いているケース

- スペクトル
- 時系列
- 温度プロファイル
- 波長依存特性
- 位置依存応答

### `HigherOrderGP` との違い

`LatentKroneckerGP` では、出力軸に意味のある座標 `train_T` が存在する。

単なるテンソル形状だけでなく、時間・波長・位置などの連続構造を利用したい場合に有力。

---

# 11. 階層・条件付き探索空間

## 11.1 `HierarchicalConditionalKernelGP`

### 概要

ある変数の値によって、別の変数が「有効 / 無効」になる hierarchical search space 用 GP。

### 例

```text
process = "heat"
    ├─ temperature
    └─ heating_time

process = "press"
    ├─ pressure
    └─ holding_time
```

`process=heat` のとき pressure は意味を持たない。

### 向いているケース

- アルゴリズム選択 + ハイパーパラメータ
- 製法選択 + 製法固有条件
- 材料種選択 + 材料固有条件

### 強み

無効な変数を通常の GP に無理やり入力するより、探索空間の条件構造を明示できる。

---

## 11.2 `HierarchicalConditionalKernelMultiTaskGP`

### 概要

Hierarchical Conditional Kernel と MultiTask GP を組み合わせたもの。

### 向いているケース

- 条件付き探索空間
- 複数タスク
- タスク間で情報共有

が同時に必要な場合。

---

# 12. Heterogeneous Multi-task

## 12.1 `HeterogeneousMTGP`

### 概要

タスクごとに異なる入力データ集合を持つ heterogeneous multitask GP。

通常の multitask では共通 feature space を前提とすることが多いが、このモデルは task-specific input structure を扱う。

### 向いているケース

例えば、

```text
Task A: process conditions
Task B: composition descriptors
Task C: measurement descriptors
```

のようにタスクごとのデータ構造が均一でない場合。

### 注意点

通常の `MultiTaskGP` で表現できる問題なら、まずそちらを使う方が単純。

---

# 13. Contextual GP

## 13.1 `SACGP`

### 概要

Structural Additive Contextual GP。

入力を context ごとの部分構造に分解し、加法的にモデル化する。

### 向いているケース

複数 context があり、それぞれの context に対応する入力部分が明確な場合。

例えば、

```text
炉1 context -> 炉1の条件
炉2 context -> 炉2の条件
炉3 context -> 炉3の条件
```

### 必要情報

`decomposition` で context と入力次元の対応を指定する。

---

## 13.2 `LCEAGP`

### 概要

Latent Context Embedding Additive GP。

SACGP の context structure に加えて、context 自体の latent embedding を学習する。

### 向いているケース

context 間に似た構造があり、それを低次元 embedding で共有したい場合。

例:

- 複数装置
- 複数ライン
- 複数材料系
- 複数プロセス

### SACGP との違い

SACGP:
- context の構造を明示

LCEAGP:
- context 間の類似性を embedding としてさらに学習

---

## 13.3 `LCEMGP`

### 概要

Latent Context Embedding Multi-output GP。

task / output の context 情報を埋め込みとして利用する multi-output GP。

### 向いているケース

複数出力・複数タスクに対して、

- categorical context feature
- continuous embedding feature

などを持たせたい場合。

### 使い所

単純な task index だけでなく、「タスク間のメタ情報」がある場合に特に有効。

---

# 14. 特に重要な使い分け

## 14.1 複数出力だから何を使うか

```text
複数出力
│
├─ 出力間の相関を利用しない
│      └─ ModelListGP
│
├─ タスクとして相関を利用する
│      │
│      ├─ X がタスクごとに異なる
│      │      └─ MultiTaskGP
│      │
│      └─ 全タスクを同じ X で観測
│             └─ KroneckerMultiTaskGP
│
├─ 出力が画像・曲線・tensor
│      └─ HigherOrderGP
│
└─ 出力軸に時間/波長/位置の座標がある
       └─ LatentKroneckerGP
```

---

## 14.2 高次元なら何を使うか

```text
高次元
│
├─ 有効変数が少数と考えられる
│      ├─ 精度・Bayesian inference 重視
│      │      └─ SaasFullyBayesianSingleTaskGP
│      │
│      └─ 計算速度重視
│             └─ MAP-SAAS
│
├─ 加法構造が期待できる
│      └─ OrthogonalAdditiveGP
│
└─ 特別な仮定がない
       └─ SingleTaskGP を baseline にして比較
```

「高次元だから SAAS」と決め打ちするのではなく、まず `SingleTaskGP` を baseline として比較するのが望ましい。

---

## 14.3 Multi-Fidelity と MultiTask の違い

### Multi-Fidelity

```text
同じ物理量を
安い近似 → 高精度評価
として測る
```

→ `SingleTaskMultiFidelityGP`

### MultiTask

```text
異なる task 間の相関を利用する
```

→ `MultiTaskGP`

低 fidelity / 高 fidelity を task として表現することも可能だが、fidelity 固有の構造を利用したいなら Multi-Fidelity GP が第一候補。

---

# 15. robotorchan での推奨モデル選択フロー

```text
START
  │
  ├─ Preference data?
  │      └─ Yes → PairwiseGP
  │
  ├─ Structured output?
  │      ├─ tensor → HigherOrderGP
  │      └─ time/wavelength/location axis → LatentKroneckerGP
  │
  ├─ Hierarchical search space?
  │      └─ Yes → HierarchicalConditionalKernelGP
  │
  ├─ Multi-fidelity?
  │      └─ Yes → SingleTaskMultiFidelityGP
  │
  ├─ Multiple tasks / outputs?
  │      ├─ independent → ModelListGP
  │      ├─ same X for every task → KroneckerMultiTaskGP
  │      └─ otherwise → MultiTaskGP
  │
  ├─ Categorical inputs?
  │      └─ Yes → MixedSingleTaskGP
  │
  ├─ Very high-dimensional?
  │      ├─ sparse relevance → SAAS / MAP-SAAS
  │      └─ additive structure → OrthogonalAdditiveGP
  │
  ├─ Large dataset?
  │      └─ Yes → SingleTaskVariationalGP
  │
  └─ Otherwise
         └─ SingleTaskGP
```

---

# 16. 実務上の推奨優先順位

特殊モデルを最初から使うより、次の順序で比較する方がよい。

1. `SingleTaskGP` を baseline にする
2. 入力構造に明確な理由がある場合だけ specialized model を使う
3. cross-validation / predictive likelihood / BO performance で比較する
4. モデルが複雑になることで得られる改善が実際にあるか確認する

特に、

- SAAS
- hierarchical GP
- contextual GP
- heterogeneous multitask
- structured output GP

は強力だが、問題設定そのものが対応する構造を持つ場合に使うモデルである。

---

# 17. 現在の public model 一覧

```python
from robotorchan.models import (
    SingleTaskGP,
    MixedSingleTaskGP,
    SingleTaskMultiFidelityGP,
    MultiTaskGP,
    KroneckerMultiTaskGP,
    ModelListGP,
    SingleTaskVariationalGP,
    PairwiseGP,
    SaasFullyBayesianSingleTaskGP,
    SaasFullyBayesianMultiTaskGP,
    HigherOrderGP,
    LatentKroneckerGP,
    OrthogonalAdditiveGP,
    AdditiveMapSaasSingleTaskGP,
    EnsembleMapSaasSingleTaskGP,
    RobustRelevancePursuitSingleTaskGP,
    HierarchicalConditionalKernelGP,
    HierarchicalConditionalKernelMultiTaskGP,
    HeterogeneousMTGP,
    SACGP,
    LCEAGP,
    LCEMGP,
)
```

---

# 18. モデルカテゴリまとめ

| カテゴリ | モデル |
|---|---|
| Standard GP | `SingleTaskGP` |
| Mixed variable | `MixedSingleTaskGP` |
| Multi-Fidelity | `SingleTaskMultiFidelityGP` |
| Multi-task | `MultiTaskGP`, `KroneckerMultiTaskGP` |
| Independent multi-output | `ModelListGP` |
| Approximate / scalable GP | `SingleTaskVariationalGP` |
| Preference | `PairwiseGP` |
| Fully Bayesian / high-dimensional | `SaasFullyBayesianSingleTaskGP`, `SaasFullyBayesianMultiTaskGP` |
| MAP-SAAS | `AdditiveMapSaasSingleTaskGP`, `EnsembleMapSaasSingleTaskGP` |
| Additive | `OrthogonalAdditiveGP` |
| Robust | `RobustRelevancePursuitSingleTaskGP` |
| Structured output | `HigherOrderGP`, `LatentKroneckerGP` |
| Hierarchical | `HierarchicalConditionalKernelGP`, `HierarchicalConditionalKernelMultiTaskGP` |
| Heterogeneous multitask | `HeterogeneousMTGP` |
| Contextual | `SACGP`, `LCEAGP`, `LCEMGP` |

---

# 19. 設計上のポイント

robotorchan の model layer は「BoTorch とは別の GP ライブラリ」を作ることを目的としていない。

基本方針は、

```text
BoTorch model
    +
robotorchan 共通 API
```

である。

そのため、

- posterior
- covariance
- input / outcome transform
- conditioning
- fantasize
- model-specific numerical behavior

などは可能な限り BoTorch に委譲する。

robotorchan 側では主に、

- raw data retention
- `supports_mll`
- `make_mll()`
- 一貫した public API

を提供する。

この構造により、BoTorch の acquisition function や optimizer と自然に組み合わせられることを重視している。
