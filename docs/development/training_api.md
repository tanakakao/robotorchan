# Model training API note

robotorchanでは、モデルの統計的な学習方法に合わせて学習契約を分けます。

- 通常のExact GP / frozen reducer model:
  `fit_gpytorch_mll(model.make_mll())`
- Joint neural model:
  optimizerで `model.training_loss()` を最小化
- Non-GP supervised surrogate:
  モデル固有の `model.fit()` を実行

`ModelTrainingMixin` は能力を明示します。

- `supports_mll=True`: `make_mll()` を利用できる
- `supports_fit=True`: 明示的な `fit()` 手順を持つ
- どちらも偽装しない。MLLを持たないモデルのために疑似MLLを作らない

Non-GP surrogateでは `NonGPModelMixin` を利用し、constructorで受け取った
`raw_train_X`, `raw_train_Y`, `raw_train_Yvar` を既存のraw-data契約に従って保持します。
raw-data snapshotはnon-persistent bufferなのでdevice / dtype変換には追従しますが、
`state_dict()` には含まれません。

`supports_input_gradients` はposterior sampleからcandidate inputまでautogradを
維持できるかを示します。Random Forestなどの外部tree ensembleでは通常False、
PyTorch-nativeなfinite BNNではTrueを目標にします。

BoTorchのMC acquisitionとの互換性は `posterior(X)` が返す `Posterior` の
sampling contractで決まり、GPであること自体は要件ではありません。

Joint neural modelの詳細は `../models/joint_neural_training.md`、
Non-GPの設計監査は `non_gp_surrogate_audit.md` を参照してください。
