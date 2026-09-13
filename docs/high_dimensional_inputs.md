# 高次元入力に対する次元削減モデル

robotorchan では、高次元入力を低次元潜在空間へ射影してから Gaussian Process を構築するための共通基盤として `ReducedGP` を使用します。

## 基本方針

入力次元削減モデルでは、ユーザーは常に元の入力空間 `X` を扱います。

```text
original X
   ↓
input reducer
   ↓
latent Z
   ↓
Gaussian Process
```

`posterior(X)`、`condition_on_observations(X, Y)`、獲得関数評価では、元の入力空間のテンソルをそのまま渡します。`ReducedGP` が内部で fitted reducer を使って潜在空間へ変換します。

## Reducer lifecycle

入力 reducer のライフサイクルは次の2通りです。

### 1. 未学習 reducer を渡す場合

通常の `PCAGP`、`PLSGP`、`RandomProjectionGP` はこの経路を使います。

```text
model construction
    ↓
reducer is not fitted
    ↓
reducer.fit(train_X, train_Y)
    ↓
reducer parameters fixed
    ↓
posterior / acquisition / conditioning / fantasize
    ↓
reducer.transform(X) only
```

モデル構築時に一度だけ学習し、その後は固定します。BO ループ中に自動的な reducer 再学習は行いません。これにより、探索途中で潜在座標系が変化することを防ぎます。

### 2. すでに学習済み reducer を渡す場合

`ReducedGP` に `is_fitted == True` の reducer を渡した場合、再学習せず、そのまま再利用します。

```python
from robotorchan.models import ReducedGP
from robotorchan.models.reduction import PCAInputReducer

reducer = PCAInputReducer(n_components=5).fit(reference_X)

model = ReducedGP(
    train_X=train_X,
    train_Y=train_Y,
    input_reducer=reducer,
)
```

この場合は次の経路になります。

```text
pre-fitted reducer
    ↓
ReducedGP construction
    ↓
fit is skipped
    ↓
reducer.transform(train_X)
    ↓
latent GP
```

この仕様により、別データで学習済みの PCA / PLS や、pretrained AutoEncoder / VAE encoder を、その潜在座標系を壊さず GP に接続できます。

この方針は output reducer にも適用されます。すでに学習済みの `OutputReducer` を渡した場合も再fitしません。

### 再学習について

モデルに接続済みの reducer を BO ループの途中で直接再fitすると、GP が保持している潜在 training inputs と reducer の座標系が不整合になります。そのため、`ReducedGP` は reducer の自動再学習を行いません。

reducer を更新したい場合は、更新後の reducer と全 raw training data から新しい GP を構築するのが基本方針です。明示的な model-level rebuild API は将来必要になった段階で追加します。

## 現在の入力 reducer

### PCAInputReducer / PCAGP

PCA により入力を低次元空間へ射影します。

```python
from robotorchan.models import PCAGP

model = PCAGP(
    train_X=train_X,
    train_Y=train_Y,
    n_components=5,
)
```

PCA の平均ベクトルと主成分行列はモデル構築時に学習され、その後の予測や fantasy model 生成では更新されません。

### PLSInputReducer / PLSGP

目的変数との共分散を利用した教師あり次元削減です。

```python
from robotorchan.models import PLSGP

model = PLSGP(
    train_X=train_X,
    train_Y=train_Y,
    n_components=5,
)
```

PLS は `train_Y` を利用して射影を学習しますが、射影行列はモデル構築後に固定されます。

### RandomProjectionInputReducer / RandomProjectionGP

Gaussian random projection を利用します。

```python
from robotorchan.models import RandomProjectionGP

model = RandomProjectionGP(
    train_X=train_X,
    train_Y=train_Y,
    n_components=5,
    random_state=0,
)
```

射影行列は初期 fit 時に一度だけ生成され、その後固定されます。

### AutoEncoderInputReducer

`AutoEncoderInputReducer` は非線形な encoder / decoder を reconstruction loss で学習し、encoder 出力を潜在入力として使用します。

```python
from robotorchan.models.neural_reduction import AutoEncoderInputReducer

reducer = AutoEncoderInputReducer(
    latent_dim=5,
    hidden_dims=(64, 32),
    activation="gelu",
    epochs=200,
    learning_rate=1e-3,
)

latent_X = reducer.fit_transform(train_X)
```

標準では入力特徴を標準化してから AutoEncoder を学習します。学習後は encoder / decoder のパラメータを freeze し、`transform(X)` では encoder の再学習を行いません。

```text
original X
   ↓ standardize with fitted mean / scale
frozen encoder
   ↓
latent Z
```

ネットワークパラメータは固定されますが、`transform(X)` は入力 `X` に関して differentiable です。そのため、今後 `ReducedGP` に統合した際も acquisition function の勾配を元の入力空間へ伝播できます。

また `reconstruct(X)` により decoder を通した元空間での再構成を確認できます。AutoEncoder の学習状態、標準化統計量、ネットワークパラメータは `state_dict` に保存され、device / dtype 変換にも追従します。

Phase 3 では reducer 本体までを実装しています。`AutoEncoderGP` convenience wrapper と acquisition integration は次の Phase で追加します。

## BoTorch API との関係

`ReducedGP` は BoTorch の `SingleTaskGP` と互換な公開インターフェースを維持します。

```python
posterior = model.posterior(test_X)
mll = model.make_mll()
```

`train_X` と `test_X` は元の入力次元のままで構いません。内部 GP は reducer が生成した潜在入力で学習します。

この構造により、獲得関数の最適化も元空間で実行できます。

```text
optimize acquisition in original X
          ↓
model.posterior(X)
          ↓
reducer.transform(X)
          ↓
latent GP posterior
```

したがって、PCA などの逆変換を使って潜在候補を元空間へ復元する処理は通常必要ありません。

## Phase 1-3 の保証範囲

現在は次の契約を基盤仕様として固定しています。

- raw training data は元空間のまま保持する。
- GP 本体は潜在入力で学習する。
- `posterior` は元空間入力を受け付ける。
- q-batch / leading batch dimension を保持して変換する。
- acquisition から元空間入力へ勾配を伝播できる。
- `condition_on_observations` で reducer を再学習しない。
- `fantasize` で reducer を再学習しない。
- reducer の buffer は model の device / dtype 変換に追従する。
- reducer の学習状態は `state_dict` round trip で保持する。
- `make_mll()` は既存 Exact GP wrapper と同じ契約を維持する。
- 未学習 reducer はモデル構築時に一度だけ fit する。
- 学習済み reducer は再fitせず、そのまま再利用する。
- model に接続した reducer の自動再学習は行わない。
- AutoEncoder は学習完了後にネットワークパラメータを freeze する。
- AutoEncoder の `transform` は元入力に関する勾配を保持する。

## 今後の拡張

この共通基盤上に以下を追加します。

1. AutoEncoderGP convenience wrapper / acquisition integration
2. VAEInputReducer / VAEGP

SAAS / MAP-SAAS は明示的な reducer を使わないため、`ReducedGP` 系とは分離した高次元モデルとして扱います。
