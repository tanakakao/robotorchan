# Model training API note

robotorchanではExact GP wrapperの `make_mll()` と、Joint neural modelの `training_loss()` を役割で分けます。

- 通常のExact GP / frozen reducer model: `fit_gpytorch_mll(model.make_mll())`
- Joint neural model: optimizerで `model.training_loss()` を最小化

この区別により `make_mll()` の既存互換性を維持したまま、GP MLL以外のreconstruction / KL regularizationを欠落させず学習できます。

詳細は `docs/joint_neural_training.md` を参照してください。
