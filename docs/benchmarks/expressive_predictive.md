# 表現力モデル Predictive benchmark

`benchmarks/expressive_predictive.py` は、表現力を高めたsingle-output surrogateを
同一データ上で比較するための再現可能なpredictive benchmarkです。

比較対象は `SingleTaskGP`、`JointEncoderGP` (DKL)、`SingleTaskDeepGP`、
`InfiniteWidthBNNGP`、`SpectralMixtureGP` です。標準GPをbaselineとして残すことで、
複雑なsurrogateを導入した効果と計算コストを分離して確認します。

## 評価指標

各seedで同じtraining/test dataを使い、次を記録します。

- RMSE: posterior meanの予測誤差
- Gaussian NLL: posterior mean/varianceを用いた予測分布の対数損失
- 95% coverage: `mean ± 1.96 * std` に真値が含まれる割合
- fit time: surrogateの学習時間
- posterior time: test set全体のposterior計算時間

RMSEだけでは不確実性推定の品質を評価できないため、NLLとcoverageを併記します。
一方、DeepGP posteriorは有限個のpredictive sampleからmean/varianceを構成するため、
Exact GPと同じ解析posteriorではありません。この違いを保持したまま、BoTorchから見える
posterior contractに対して同じ評価指標を計算します。

## 合成objective

入力は3次元の `[0, 1]^3` とし、周期成分、非線形interaction、trendを組み合わせます。
単純なRBF GPだけでなく、DKL、DeepGP、NNGP kernel、Spectral Mixture kernelが持つ
異なる表現力を比較できるようにしています。

このbenchmarkは「どのモデルが常に優れているか」を決めるものではありません。
seed、training size、最適化条件を揃えた上で、関数構造ごとの適合性と計算コストを
確認するための基盤です。

## 実行

```bash
python benchmarks/expressive_predictive.py
```

軽量確認では、たとえば次のようにtraining/test sizeとDeepGPのstep数を減らせます。

```bash
python benchmarks/expressive_predictive.py \
  --seeds 0 \
  --n-train 16 \
  --n-test 32 \
  --deep-gp-steps 2
```

既定では `benchmark_results/expressive_predictive.csv` にraw resultを書き出します。
benchmark結果そのものをCIの合否判定には使用しません。CIでは構造・軽量実行だけを確認し、
モデル性能の大小を固定値としてテストしない方針です。
