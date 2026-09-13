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

### reducer lifecycle

入力 reducer はモデル構築時に一度だけ学習し、その後は固定します。

```text
model construction
    ↓
reducer.fit(train_X, train_Y)
    ↓
reducer parameters fixed
    ↓
posterior / acquisition / conditioning / fantasize
    ↓
reducer.transform(X) only
```

BO ループ中に自動的な reducer 再学習は行いません。これにより、探索途中で潜在座標系が変化することを防ぎます。

この方針は PCA、PLS、Random Projection に共通です。将来 AutoEncoder / VAE reducer を追加する場合も、まずは同じ frozen-reducer contract に従います。

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

## Phase 1 の保証範囲

Phase 1 では次の契約を基盤仕様として固定します。

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

## 今後の拡張

この共通基盤上に以下を追加します。

1. AutoEncoderInputReducer / AutoEncoderGP
2. VAEInputReducer / VAEGP

SAAS / MAP-SAAS は明示的な reducer を使わないため、`ReducedGP` 系とは分離した高次元モデルとして扱います。
