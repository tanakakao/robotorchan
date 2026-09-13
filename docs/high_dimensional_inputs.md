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

通常の `PCAGP`、`PLSGP`、`RandomProjectionGP`、`AutoEncoderGP` はこの経路を使います。

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

この仕様により、別データで学習済みの PCA / PLS や pretrained neural encoder を、その潜在座標系を壊さず GP に接続できます。

この方針は output reducer にも適用されます。すでに学習済みの `OutputReducer` を渡した場合も再fitしません。

### 再学習について

モデルに接続済みの reducer を BO ループの途中で直接再fitすると、GP が保持している潜在 training inputs と reducer の座標系が不整合になります。そのため、`ReducedGP` は reducer の自動再学習を行いません。

reducer を更新したい場合は、更新後の reducer と全 raw training data から新しい GP を構築するのが基本方針です。明示的な model-level rebuild API は将来必要になった段階で追加します。

## 現在の入力 reducer / model

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

### AutoEncoderInputReducer / AutoEncoderGP

`AutoEncoderInputReducer` は非線形な encoder / decoder を reconstruction loss で学習し、encoder 出力を潜在入力として使用します。目的変数 `Y` は潜在空間の学習には使用しません。

```python
from robotorchan.models import AutoEncoderGP

model = AutoEncoderGP(
    train_X=train_X,
    train_Y=train_Y,
    latent_dim=5,
    hidden_dims=(64, 32),
    activation="gelu",
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
```

内部では次の流れになります。

```text
train_X
   ↓ standardize
AutoEncoder training by reconstruction loss
   ↓
freeze encoder / decoder
   ↓
latent train_Z
   ↓
SingleTaskGP
```

予測・獲得関数評価時は、元の入力空間をそのまま使います。

```text
candidate X
   ↓
frozen encoder
   ↓
latent Z
   ↓
GP posterior
```

encoder のパラメータは freeze されていますが、`transform(X)` は入力 `X` に関して differentiable です。そのため、qLogEI などの acquisition function の勾配を元の入力空間へ伝播できます。

`AutoEncoderGP` は `ReducedGP` と同じ公開契約を持ちます。

```python
posterior = model.posterior(test_X)
mll = model.make_mll()
conditioned = model.condition_on_observations(new_X, new_Y)
```

raw training data は元空間のまま保持し、GP の `train_inputs` のみ潜在空間になります。`condition_on_observations` や `fantasize` でも AutoEncoder を再学習しません。モデルと reducer の状態は `state_dict` に含まれます。

## BoTorch API との関係

`ReducedGP` 系は BoTorch の `SingleTaskGP` と互換な公開インターフェースを維持します。

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

したがって、PCA / AutoEncoder などの逆変換を使って潜在候補を元空間へ復元する処理は通常必要ありません。一方で、GP のモデル入力次元は削減されますが、acquisition optimizer 自体は元の `D` 次元空間を探索します。潜在空間そのものを最適化するモードは別拡張として扱います。

## Phase 1-4 の保証範囲

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
- `AutoEncoderGP` は公開 model API と acquisition integration を提供する。

## 今後の実装計画

高次元入力の neural reduction 系は、教師なし・教師あり・GPとの共同学習を分けて実装します。

1. **Phase 5: VAEInputReducer / VAEGP**
   - VAE を入力表現として学習する。
   - 初版では encoder posterior mean `mu(X)` を決定論的な GP 入力として使用する。
   - VAE 学習後は encoder を freeze する。
2. **Phase 6: SupervisedAutoEncoderInputReducer**
   - reconstruction loss と `Y` 予測 loss を併用して潜在表現を事前学習する。
   - 学習後は encoder を freeze し、通常の `ReducedGP` lifecycle に接続する。
3. **Phase 7: SupervisedAutoEncoderGP**
   - Supervised AutoEncoder と GP を接続する。
   - PLS-GP の非線形版に近い位置づけとする。
4. **Phase 8: Joint GP / Encoder Learning**
   - GP marginal log likelihood を encoder まで逆伝播する DKL 型を扱う。
   - frozen `InputReducer` とは分離した lifecycle とする。
5. **Phase 9: Hybrid AutoEncoder-GP**
   - GP likelihood と reconstruction loss を同時最適化する。
   - `-log p(Y | Z) + lambda_rec * L_reconstruction` を基本形とする。
6. **Phase 10: Supervised / Joint VAE extensions**
   - Supervised VAE-GP: reconstruction + KL + supervised loss で事前学習する。
   - Joint VAE-GP: GP likelihood を VAE encoder まで伝播する。
   - uncertainty-aware VAE-GP: `q(Z | X)` のサンプルを GP posterior に伝播し、潜在表現の不確実性を扱う。
7. **Phase 11: benchmark / notebook / model selection guide**
   - PCA-GP、PLS-GP、RandomProjection-GP、AE-GP、VAE-GP、Supervised AE/VAE、Joint/Hybrid 系、SAAS 系を共通ベンチマークで比較する。
   - RMSE、NLL、学習時間、acquisition 実行、BO regret を比較する。

### 系統の違い

```text
Unsupervised AE / VAE
X → representation pretraining → freeze → GP

Supervised AE / VAE
X,Y → supervised representation pretraining → freeze → GP

Joint / DKL
X → Encoder → GP → Y
         ↑       │
         └───────┘
         GP objective

Hybrid AE / VAE-GP
X → Encoder → GP → Y
    ↓        ↑
 Decoder     │
    ↓        │
 reconstruction / KL + GP objective
```

Supervised 系は GP 学習前に目的変数を使って潜在表現を作り、その後 freeze する安定性重視の方式です。Joint 系は予測性能を直接改善するよう encoder と GP を同時最適化します。Hybrid 系では reconstruction / KL を正則化として残します。

SAAS / MAP-SAAS は明示的な reducer を使わないため、`ReducedGP` 系とは分離した高次元モデルとして扱います。
