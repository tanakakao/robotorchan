# Batch / Noisy / Pending：理論から実装へ

## この章で解決する問い

q点のjoint utilityは通常、1点のacquisitionの単純和ではない。noise、q、pending、fantasizationは独立の設計軸である。

数学的な定義・仮定・導出は[理論の対応章](../../theory/acquisition/04_batch_noisy.md)が正本であり、
この章は**数式の計算対象 → クラス構造 → 内部処理 → 近似と制限**をつなぐ日本語実装ガイドである。

## 実装の対応と委譲

`qLogExpectedImprovement`・`qLogNoisyExpectedImprovement`はBoTorch-native。
`make_qnei_acquisition`はX_baselineとsemantic objectiveをつなぐ。

## 採用時の確認と注意事項

X_baselineの非空2D契約、posteriorのjoint sample shape、baseline pruningとpending points、MC seedを区別。

## qLogNEIでjoint候補を評価する

観測ノイズを考慮する`qLogNoisyExpectedImprovement`では、評価済み点`X_baseline`を渡し、
候補`X`のjoint posteriorを使ってbaselineに対する改善量をMonte Carlo推定する。

## 関連する実装資料

baseline の選び方は単なる API detail ではなく、acquisition の計算量や意思決定対象に影響します。
大規模 baseline では pruningなどの計算上の工夫が重要になる場合があります。

## sample-wise utilityからjoint q-batchへ

`qLogExpectedImprovement`は`LogImprovementMCAcquisitionFunction`を通して
`SampleReducingMCAcquisitionFunction`を継承します。`_sample_forward()`が各応答sampleの改善量をlog-spaceで滑らかに近似し、
基底の`forward()`が候補軸を集約してからsample軸を集約します。Theoryの「同じsample内でq候補を比較し、そのutilityを期待する」順序に対応します。
数値的な平滑化を含むため、有限MC計算を厳密期待値と同一視しません。

`qLogNoisyExpectedImprovement`は`CachedCholeskyMCSamplerMixin`も継承します。
`_get_samples_and_objectives()`がbaselineと新しい候補のjoint sampleを扱い、
`compute_best_f()`がbaseline sampleごとの比較基準を用意します。観測済みYの単一最大値だけを全sampleの基準に固定する計算とは異なります。
`set_X_pending()`は未完了候補を扱う状態であり、実観測を追加する操作ではありません。

これらの推定・キャッシュ処理は[BoTorchのlogei.py][logei]が担当します。
robotorchanの[composition.py](../../../src/robotorchan/acquisition/composition.py)の
`make_qnei_acquisition()`は、baselineとsemantic objectiveをこの上流処理へ接続します。

[logei]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/acquisition/logei.py

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/32_batch_async_fantasization.ipynb)を参照してください。
