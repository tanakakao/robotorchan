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
model.fit_heteroskedastic(iterations=3)

response = model.posterior(test_X)
noise_variance = model.predicted_noise(test_X)
log_noise_posterior = model.noise_posterior(test_X)
```
これは実用的な反復 two-process モデルです。`JointHeteroskedasticSingleTaskGP` は別の joint variational モデルとして実装されています。選択方法は [Robust / Noise](robust_noise.md) を参照してください。

## BoTorch との接続

単体の回帰だけでなく、BoTorch surrogate として MC 獲得関数、バッチ単位のノイズ予測、元入力空間での scenario 生成と risk 集約、登録済み noise GP の dtype 移行を検証対象にしています。

学習後の noise GP は PyTorch submodule として登録されます。`.to(...)`、`.float()`、`.double()` は両方の process を移行します。応答 posterior は獲得関数が使う標準の BoTorch posterior であり、`predicted_noise` は追加の診断・robust design 用の出力です。
