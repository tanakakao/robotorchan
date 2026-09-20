# Infinite-width BNN GP

`InfiniteWidthBNNGP` は、全結合ReLUニューラルネットワークを無限幅極限へ
取ったときのNNGP（Neural Network Gaussian Process）カーネルをExact GPとして
利用するモデルです。

## 位置づけ

DKLは有限幅のニューラルネットワーク特徴抽出器を学習し、その潜在表現上でGPを
構築します。一方、Infinite-width BNN GPではニューラルネットワークそのものを
無限幅極限の解析的カーネルとして表現します。そのためニューラル特徴抽出器を
明示的に学習せず、通常のExact GPと同じposterior / MLL / BoTorch acquisition
interfaceを利用できます。

## カーネル

`InfiniteWidthReLUKernel` は線形共分散から開始し、ReLUの解析的共分散写像を
`depth` 回再帰適用します。主な設定は次です。

- `depth`: 隠れReLU層に対応する再帰回数
- `weight_variance`: 重み事前分散
- `bias_variance`: バイアス事前分散
- `ard`: 入力次元ごとのlengthscaleを利用するか

`depth` は有限幅NNの学習層数と完全に同じ意味ではなく、無限幅NNGPカーネルの
再帰深さです。

## 利用場面

RBF / Matérnよりニューラルネットワーク由来の非線形事前を使いたい一方、
DKLの特徴抽出器学習やDeepGPの変分推論を導入したくない場合に候補になります。

モデルは `SingleTaskGP` の共通Exact GP contractを利用するため、
`make_mll()`、posterior、qLogEI、`optimize_acqf` を通常のGPと同様に扱えます。
