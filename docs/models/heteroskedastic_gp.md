# Heteroskedastic GP

## 実装方式

応答の GP と対数残差分散の GP を交互に学習する two-GP モデルです。
```text
y = f(x) + epsilon(x)
epsilon(x) ~ N(0, sigma(x)^2)
log sigma(x)^2 ~ GP
```
`HeteroskedasticSingleTaskGP.fit_heteroskedastic()` は、応答 GP の学習、二乗残差からの log-noise GP の学習、予測した入力依存分散による応答尤度の更新を繰り返します。

既知の `train_Yvar` を与える fixed-noise モデルとは異なり、未観測点でも問い合わせ可能なノイズの曲面を学習します。

## 使用例
```python
model = HeteroskedasticSingleTaskGP(train_X, train_Y, noise_floor=1e-6)
model.fit(iterations=3)

response = model.posterior(test_X)
noise_variance = model.predicted_noise(test_X)
log_noise_posterior = model.noise_posterior(test_X)
```
これは実用的な反復 two-process モデルです。`JointHeteroskedasticSingleTaskGP` は別の joint variational モデルとして実装されています。選択方法は [Robust / Noise](robust_noise.md) を参照してください。

## BoTorch との接続

単体の回帰だけでなく、BoTorch surrogate として MC 獲得関数、バッチ単位のノイズ予測、元入力空間での scenario 生成と risk 集約、登録済み noise GP の dtype 移行を検証対象にしています。

学習後の noise GP は PyTorch submodule として登録されます。`.to(...)`、`.float()`、`.double()` は両方の process を移行します。応答 posterior は獲得関数が使う標準の BoTorch posterior であり、`predicted_noise` は追加の診断・robust design 用の出力です。

### 統一学習 API とノイズ推定

`fit(iterations=3, optimizer_kwargs=None)` は応答GPと対数ノイズGPを学習します。従来の `fit_heteroskedastic()` も利用できます。全入力条件に反復測定がある場合は各条件の不偏標本分散をノイズGPの教師信号に用います（集約平均の分散ではなく単発測定の分散）。それ以外は学習点での単純な残差ではなく、近似leave-one-out残差と潜在分散を利用します。初期分散を下限に固定せず、最後に推定ノイズで応答GPを再学習します。標準化を使用する場合は内部尤度へ渡す分散を変換します。単一観測からのノイズ推定は識別が難しく、推定値の精度を保証するものではありません。Kronecker prototypeは対象外です。
