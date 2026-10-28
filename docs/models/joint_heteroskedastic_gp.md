# Joint Heteroskedastic GP

応答と対数ノイズ分散の 2 つの潜在 GP を、Monte Carlo 近似による joint ELBO で同時学習します。残差からノイズ GP を交互に推定する反復モデルとは推論目的が異なります。

## 利用方法と制約

`JointHeteroskedasticSingleTaskGP` の `training_loss()` を PyTorch optimizer で最小化し、`posterior(test_X)` で潜在応答、`predicted_noise(test_X)` で観測ノイズを取得します。`num_inducing` と正の `noise_floor` を指定できます。`supports_mll = False` であり、Exact GP の `make_mll()` を使う経路ではありません。

## 理論・実装・検証を辿る

[理論](../theory/15_heteroskedastic_noise.md)と[実装ガイド](../guide/15_heteroskedastic_noise.md)で仮定・推論方法・近似を確認できます。[ソース](../../src/robotorchan/models/robust/joint_heteroskedastic.py)から API と処理を確認してください。

過去の設計条件・数式・検証要件は[設計記録](../../.github/project-records/model-design/joint_heteroskedastic_gp.md)に原文を保存しています。そこでの将来計画や Phase の完了条件は現在の対応保証ではありません。現在のモデル選択は[モデル索引](README.md)を参照してください。
