# Model training API note

robotorchanではExact GP wrapperの `make_mll()` と、Joint neural modelの `training_loss()` を役割で分けます。

- 通常のExact GP / frozen reducer model: `fit_gpytorch_mll(model.make_mll())`
- Joint neural model: optimizerで `model.training_loss()` を最小化

この区別により `make_mll()` の共通Exact GP契約を保ったまま、GP MLL以外のreconstruction / KL regularizationを欠落させず学習できます。

詳細は `../models/joint_neural_training.md` を参照してください。
