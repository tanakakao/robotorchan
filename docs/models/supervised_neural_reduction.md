# 教師ありニューラル次元削減

`SupervisedAutoEncoderInputReducer` は入力再構成と補助的な出力予測の損失を合わせ、教師ありの非線形表現を学習します。出力を使わない AE と異なり、GP の周辺尤度で encoder を同時学習する joint DKL とも異なります。

## 利用方法と制約

`latent_dim`、`hidden_dims`、`supervised_weight`、`epochs` を設定し、`fit(train_X, train_Y)` の後に `transform(train_X)` を呼びます。学習後の encoder・decoder・補助 head は固定されますが、入力に対する変換の微分は保ちます。補助 head は最終予測モデルではなく診断用です。公開 wrapper は `SupervisedAutoEncoderGP` です。

## 理論・実装・検証を辿る

[理論](../theory/19_neural_reduction_gp.md)と[実装ガイド](../guide/19_neural_reduction_gp.md)で仮定・推論方法・近似を確認できます。[ソース](../../src/robotorchan/models/high_dimensional/reduced/supervised_neural.py)から API と処理を確認してください。

過去の設計条件・数式・検証要件は[設計記録](../../.github/project-records/model-design/supervised_neural_reduction.md)に原文を保存しています。そこでの将来計画や Phase の完了条件は現在の対応保証ではありません。現在のモデル選択は[モデル索引](README.md)を参照してください。
