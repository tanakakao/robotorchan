# Spectral Mixture GP

`SpectralMixtureGP` は、定常カーネルのスペクトル密度をガウス混合で表現する
Spectral Mixture (SM) kernel を利用するExact GPです。

## 位置づけ

RBFやMatérnは滑らかさや距離減衰を比較的単純な形で仮定します。一方SM kernelは、
Bochnerの定理に基づきスペクトル密度を混合分布として表現することで、周期・準周期・
複数スケールを含む定常構造を柔軟に表現します。

`num_mixtures` を増やすほど表現力は上がりますが、学習パラメータと局所解も増えるため、
常に大きくすればよいわけではありません。

## 初期化

SM kernelは初期値への依存が比較的大きいため、robotorchanでは明示的に初期化します。

- `initialization="data"`: GPyTorchの `initialize_from_data` を利用
- `initialization="empspect"`: empirical spectrumを使う初期化

通常は `data` から開始し、1次元の規則的な系列などでempirical spectrum初期化を
比較する使い方を想定します。

## 利用場面

周期性、準周期性、複数の周波数成分、複数スケールの定常変動が想定される問題に向きます。
非定常性そのものを扱うモデルではないため、入力位置によって周波数特性が大きく変化する
場合はNonstationary GPなど別のモデルも比較対象にします。

`SingleTaskGP` の共通Exact GP contractを継承するため、raw training data、
`make_mll()`、posterior、BoTorch acquisition、`optimize_acqf` を通常のGPと同様に
扱えます。
