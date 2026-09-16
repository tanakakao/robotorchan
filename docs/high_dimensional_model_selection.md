# 高次元入力モデル選択ガイド

robotorchan の高次元入力モデルは、**次元削減の目的**と**データ数**で選ぶのが基本です。単純にモデル名だけで優劣を決めず、同じデータ分割・同じ評価指標で比較してください。

## まず試すモデル

| 状況 | 主な候補 | 特徴 |
| --- | --- | --- |
| 線形な低次元構造がありそう | `PCAGP` | Yを使わず安定。最初の基準モデルに向く |
| Yに関係する方向を抽出したい | `PLSGP` | 教師あり線形削減。小標本では成分数を小さくする |
| 非線形構造を圧縮したい | `AutoEncoderGP` / `VAEGP` | 表現学習はY非依存。事前学習後freeze |
| 非線形かつYに関連する表現が欲しい | `SupervisedAutoEncoderGP` / `SupervisedVAEGP` | Yを使って潜在表現を事前学習 |
| GP予測性能に合わせて表現を学習したい | `JointEncoderGP` / `JointVAEGP` | GP objectiveをencoderまで伝播 |
| 高次元だが少数の元変数だけが効く | SAAS / MAP-SAAS系 | 明示的な次元削減をせず関連次元を疎に選択 |

`RandomProjectionGP` は非常に安価な比較基準として有用です。複雑な表現学習モデルが Random Projection を明確に上回らない場合、表現学習コストに見合っているか再検討できます。

## 推奨比較順

小標本のBOでは、まず `SingleTaskGP`、`PCAGP`、`PLSGP`、`RandomProjectionGP` を比較します。その後、非線形性が必要なら AE/VAE 系を追加します。Joint 系は表現自由度が高い分、データが少ない場合の過学習と学習不安定性を確認してください。

### 評価指標

Phase 11 の共通ベンチマークでは、最低限以下を記録します。

- RMSE: posterior mean の予測誤差
- Gaussian NLL: posterior uncertainty を含めた予測品質
- training time: reducer学習とGP fittingを含む時間
- posterior time: test setに対するposterior計算時間

BO用途ではこれに加えて acquisition の評価時間と、反復ごとの simple regret / best observed value を評価します。予測RMSEが最良でもBO性能が最良とは限らないため、最終判断ではBO loop benchmarkを優先します。

## 共通ベンチマーク

`benchmarks/high_dimensional_inputs.py` は、低次元active subspaceを埋め込んだ高次元synthetic regressionでモデルを比較します。

```bash
python benchmarks/high_dimensional_inputs.py \
  --n-train 64 \
  --n-test 128 \
  --input-dim 40 \
  --latent-dim 5 \
  --neural-epochs 50
```

結果はデフォルトで `benchmark_results/high_dimensional_inputs.csv` に保存されます。

初期ベンチマークには `SingleTaskGP`、PCA、PLS、Random Projection、AE、VAE、Supervised AE、Supervised VAEを含めます。Joint/Hybrid系とSAAS系は学習手順が異なるため、次段階で専用runnerを追加します。

## 潜在次元の扱い

`latent_dim` / `n_components` は固定値だけで評価しないでください。例えば `2, 5, 10` のように複数候補を比較します。PCAの説明分散だけで決めると、目的変数に重要だが入力分散が小さい方向を落とす可能性があります。PLSやSupervised系ではYを利用するため、この挙動は異なります。

## BOでの注意点

現在のReducedGP系は、GP内部の入力次元を削減しますが、獲得関数の最適化自体は元のD次元空間で行います。そのため、高次元化によるGP modeling負荷は軽減できても、`optimize_acqf` の探索難易度そのものは残ります。

非常に高いDでacquisition optimizationがボトルネックになる場合は、REMBO / BAxUS / TuRBOのような**探索戦略側の高次元対応**を別レイヤーとして検討します。これはPCA-GPやVAE-GPなどのsurrogate reductionとは別問題です。
