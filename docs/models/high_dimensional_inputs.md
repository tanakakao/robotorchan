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

`posterior(X)`、`condition_on_observations(X, Y)`、獲得関数評価では元の入力空間をそのまま渡し、`ReducedGP` が内部で fitted reducer を使って潜在空間へ変換します。

## Reducer lifecycle

未学習 reducer はモデル構築時に一度だけ fit し、その後固定します。学習済み reducer を渡した場合は再fitせず再利用します。接続済み reducer をBO途中で直接再fitするとGPの潜在training inputsと座標系が不整合になるため、更新したい場合は全raw training dataからモデルを再構築します。

この契約は PCA / PLS / Random Projection / AE / VAE / supervised reducer、および output reducer に共通です。

## 実装済みモデル

### Frozen reduction

- `PCAGP`: 教師なし線形PCA。
- `PLSGP`: Yを利用する教師あり線形削減。
- `RandomProjectionGP`: 安価なGaussian random projection。
- `AutoEncoderGP`: reconstructionで事前学習したAEをfreezeしてGPへ接続。
- `VAEGP`: VAE posterior meanを決定論的GP入力として使用。
- `SupervisedAutoEncoderGP`: reconstruction + supervised lossで表現を事前学習。
- `SupervisedVAEGP`: reconstruction + KL + supervised lossで表現を事前学習。

Frozen reducerでも `transform(X)` は元入力Xに関してdifferentiableであり、acquisitionの勾配を元空間へ伝播できます。

### Joint representation learning

- `JointEncoderGP`: GP marginal log likelihoodでencoderとGPを共同学習。
- `HybridAutoEncoderGP`: GP objectiveにreconstruction regularizationを追加。
- `JointVAEGP`: GP objectiveにVAE reconstruction / KL regularizationを追加。

Joint系は `training_loss()` を共通の最適化契約とします。

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(100):
    optimizer.zero_grad()
    loss = model.training_loss()
    loss.backward()
    optimizer.step()
```

`JointEncoderGP.training_loss()` は negative exact GP MLL、`HybridAutoEncoderGP` はそこへ weighted reconstruction loss、`JointVAEGP` はさらに reconstruction / KL を加えます。旧APIの互換aliasは設けず、Joint系の学習APIは `training_loss()` に統一します。

`make_mll()` は引き続きBoTorch/GPyTorchのMLLオブジェクトを返す低レベルAPIです。Hybrid / JointVAEで正則化まで含めて学習する場合は `fit_gpytorch_mll(model.make_mll())` ではなく `training_loss()` を使用してください。

### VAE latent uncertainty

`JointVAEGP.uncertainty_aware_posterior()` は `q(z | X)` からlatent sampleを生成し、GP posterior mixtureの一次・二次モーメントをMonte Carloで近似します。通常の `posterior(X)` はposterior mean latent code `mu(X)` を使う決定論的予測です。

### SAAS

SAAS / MAP-SAASは明示的なreducerを使わず、元の高次元入力空間で有効次元を疎に扱う別系統です。ReducedGP系とは分離して実装しています。

## BoTorch APIとの関係

ReducedGP系は元空間の `train_X` / `test_X` を公開APIで受け付け、内部GPのみ潜在空間を使います。

```text
optimize acquisition in original X
          ↓
model.posterior(X)
          ↓
reducer.transform(X)
          ↓
latent GP posterior
```

このためPCA/AEなどの逆変換で潜在候補を復元する処理は通常不要です。ただしGP入力次元が削減されてもacquisition optimizer自体は元のD次元空間を探索します。潜在空間最適化やREMBO / BAxUS / TuRBOなど探索戦略側の高次元対応は別レイヤーとして扱います。

## 現在の保証範囲

- raw training dataを元空間のまま保持する。
- ReducedGP本体は潜在入力で学習する。
- `posterior` は元空間入力を受け付ける。
- q-batch / leading batch dimensionを保持する。
- acquisitionから元入力へ勾配を伝播できる。
- conditioning / fantasizeでfrozen reducerを再学習しない。
- reducerのbufferはdevice / dtype変換に追従する。
- reducer状態はstate_dict round tripで保持する。
- 未学習reducerは構築時にfitし、学習済みreducerは再fitしない。
- neural frozen reducerは学習後freezeする。
- Joint系は `training_loss()` でencoderとGPを共同最適化できる。
- JointVAEGPはlatent uncertainty-aware posteriorを提供する。

## Mixed / MultiTask との組合せ

public API は「すべての reducer × すべての task/mixed 形式」の機械的な直積ではありません。

- SingleTask の frozen / neural reduction は named wrapper を提供します。
- MultiTask / KroneckerMultiTask では PCA / PLS / Random Projection / AE / VAE / supervised / joint 系を task identity を保持する形で提供します。
- Mixed SingleTask では continuous feature だけを reducer へ通し、categorical feature を bypass します。
- Mixed + MultiTask の共通基盤は `MixedReducedMultiTaskGP` /
  `MixedReducedKroneckerMultiTaskGP` です。存在しない named cross-product をドキュメント上で仮定しません。

詳細は [高次元 MultiTask GP](high_dimensional_multitask.md) を参照してください。

## 高次元入力・MultiTask 拡張の実装状況

高次元入力モデルに加え、入力削減とMultiTask GPを組み合わせる基盤まで実装済みです。

| Phase | 内容 | 状態 |
| --- | --- | --- |
| 1-4 | ReducedGP lifecycle、PCA/PLS/RP、AutoEncoderGP | 完了 |
| 5 | VAEInputReducer / VAEGP | 完了 |
| 6-7 | Supervised AutoEncoder reducer / GP | 完了 |
| 8 | JointEncoderGP | 完了 |
| 9 | HybridAutoEncoderGP | 完了 |
| 10 | Supervised VAE、JointVAE、latent uncertainty propagation | 完了 |
| 11 | predictive/acquisition/SAAS/sequential/repeated-seed benchmark、model-selection guide、Notebook | 完了 |
| MT拡張 | PCA/PLS/RP/AE/VAE/教師あり/Joint系 × MultiTask/Kronecker | 完了 |
| Mixed MT | continuous reduction + categorical bypass + task保持 | 完了 |
| lifecycle | state_dict復元後のreducer依存training input再同期 | 完了 |
| benchmark | 高次元long-format MultiTaskの比較benchmark | 完了 |

モデル選択とbenchmarkの詳細は `docs/high_dimensional_model_selection.md`、実行例は `examples/notebooks/23_high_dimensional_bo_benchmark.ipynb` を参照してください。

## 次の拡張候補

現在の拡張候補は、モデル名の直積追加ではなく以下を独立テーマとして扱います。

- 連続高次元空間での acquisition optimization benchmark。
- REMBO / BAxUS / TuRBOなど探索戦略との統合。
- latent dimension / reducer hyperparameter選択支援。
- Joint系のconditioning / fantasize契約の強化。
- VAEの再現性、log-variance安定化、sampling APIの改善。
- 実材料・製造データを想定したbenchmark suite。
