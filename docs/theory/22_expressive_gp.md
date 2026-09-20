# Expressive Gaussian Process models

標準的なGaussian Process (GP) は、kernelによって関数の滑らかさや相関構造を表現します。
データが少ない領域でも不確実性を自然に扱える一方、単純なstationary kernelだけでは、
階層的な非線形性、学習された表現、複数周波数などを十分に表現できない場合があります。

robotorchanでは、この問題に対して異なる仮定を持つ4系統を扱います。

## Deep Kernel Learning

`JointEncoderGP` は入力を決定論的なneural feature extractorで変換し、その潜在表現上に
Exact GPを構築します。

```text
x -> neural feature extractor -> z(x) -> Exact GP
```

kernelを `k(z(x), z(x'))` と考えると、GPのkernel geometry自体をデータから学習している
と解釈できます。feature extractorとGP hyperparameterを同じMLL objectiveで学習するため、
固定済みencoderを使う単純な二段階法とは異なります。

表現学習能力は高い一方、小標本ではnetwork自由度が過学習につながる可能性があります。
カテゴリ変数を含む場合は `MixedJointEncoderGP` が連続特徴だけをencoderへ渡し、
カテゴリ構造をnative mixed kernel側に残します。

## Deep Gaussian Process

Deep GPはGPを階層化します。

```text
x -> GP_1 -> h_1 -> GP_2 -> ... -> y
```

中間表現そのものが確率変数なので、DKLの決定論的なfeature extractorとは異なります。
この階層を周辺化すると一般に単純なGaussian posteriorにはならないため、
`SingleTaskDeepGP` はvariational inferenceとMonte Carlo posteriorを使用します。

Exact GPより推論コストと最適化難度が高くなります。現在のrobotorchanでは、
数学的意味を保てるsingle-output continuous-input版を提供し、API対称性だけを目的とした
Mixed / multi-output wrapperは作りません。

## Infinite-width neural network GP

有限幅ニューラルネットワークの幅を無限大へ取ると、適切な事前の下で関数分布がGPへ
収束する場合があります。`InfiniteWidthBNNGP` は全結合ReLU networkに対応する
NNGP kernelを解析的に計算します。

DKLのようにfeature extractorを学習するのではなく、neural-network由来のinductive biasを
kernel priorとして組み込みます。そのため推論はExact GPのままで、
`make_mll()`、posterior、BoTorch acquisitionを標準GPと同様に利用できます。

## Spectral Mixture kernel

定常kernelはBochnerの定理によりspectral densityと対応します。Spectral Mixture kernelは
そのspectral densityをGaussian mixtureで近似し、周期、準周期、複数周波数を柔軟に
表現します。

`SpectralMixtureGP` の `num_mixtures` は周波数成分を表現する自由度を制御します。
自由度を増やすと局所解や識別性の問題も増えるため、初期化と複数条件での比較が重要です。
非定常性そのものを表すkernelではない点にも注意します。

## 4モデルの違い

| モデル | 表現力の源 | 推論 | 主な適用候補 |
|---|---|---|---|
| `JointEncoderGP` | 学習された決定論的特徴 | Exact GP | 非線形な潜在表現 |
| `SingleTaskDeepGP` | 確率的なGP階層 | Variational / MC | 階層的な非線形性 |
| `InfiniteWidthBNNGP` | neural-network由来kernel prior | Exact GP | neural inductive bias |
| `SpectralMixtureGP` | spectral density mixture | Exact GP | 周期・準周期・多周波 |

表現力が高いモデルが常に予測やBOで優れるわけではありません。標本数、objectiveの構造、
学習安定性、posterior計算コストを含めて `SingleTaskGP` と比較します。

## Bayesian Optimizationとの関係

DKL、Infinite-width BNN GP、Spectral Mixture GPはExact GP posteriorを提供し、
元入力空間から通常のBoTorch acquisitionへ直接接続できます。DeepGPはMonte Carlo
posteriorを介してMC acquisitionへ接続します。

候補探索時に利用者が潜在空間へ手動変換する必要はありません。モデル内部の表現方法が
異なっていても、探索候補は元の設計変数空間で解釈されます。

## robotorchanとの対応

- 利用ガイド: [Expressive models](../models/expressive.md)
- DKL詳細: [Joint neural GP](../models/joint_neural_training.md)
- DeepGP詳細: [DeepGP](../models/deep_gp.md)
- Infinite-width BNN: [Infinite-width BNN GP](../models/infinite_width_bnn.md)
- Spectral Mixture: [Spectral Mixture GP](../models/spectral_mixture.md)
- BoTorch統合: [BoTorch integration](../models/expressive_botorch_integration.md)
- Benchmark: [Predictive benchmark](../benchmarks/expressive_predictive.md)
- Notebook: [Expressive surrogate GP](../../examples/notebooks/24_expressive_surrogate_gp.ipynb)
