# Mixed 不確実入力 GP

連続座標の Gaussian 不確実性と、既知のカテゴリに対する共分散を組み合わせます。カテゴリ符号を連続量のように揺らして不確実性を表現するモデルではありません。

## 利用方法と制約

`MixedUncertainInputSingleTaskGP` の `cat_dims` はカテゴリ列を表します。Gaussian 不確実性は連続列へ適用し、カテゴリ共分散と合成します。カテゴリ自身が不確かな場合は、確率ベクトルを使う別のモデルを選んでください。

## 理論・実装・検証を辿る

[理論](../theory/16_uncertain_input_gp.md)と[実装ガイド](../guide/16_uncertain_input_gp.md)で仮定・推論方法・近似を確認できます。[ソース](../../src/robotorchan/models/uncertain/uncertain_input.py)から API と処理を確認してください。

過去の設計条件・数式・検証要件は[設計記録](../../.github/project-records/model-design/mixed_uncertain_input_gp.md)に原文を保存しています。そこでの将来計画や Phase の完了条件は現在の対応保証ではありません。現在のモデル選択は[モデル索引](README.md)を参照してください。
