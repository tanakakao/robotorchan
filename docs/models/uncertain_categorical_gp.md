# 不確実なカテゴリ入力

真のカテゴリが分からない観測を、有限カテゴリ上の確率ベクトルで表現します。確率は非負で和が 1、既知のカテゴリは one-hot ベクトルで表します。整数符号への jitter は使いません。

## 利用方法と制約

`UncertainCategoricalSingleTaskGP` はカテゴリ共分散行列の下で期待共分散を計算します。2 つのカテゴリ分布を p、q とすると期待値は pᵀ K_cat q です。連続入力の共分散と積で組み合わせます。複数カテゴリ列や任意の構造への対応は、具体的な公開 API とテストで確認してください。

## 理論・実装・検証を辿る

[理論](../theory/16_uncertain_input_gp.md)と[実装ガイド](../guide/16_uncertain_input_gp.md)で仮定・推論方法・近似を確認できます。[ソース](../../src/robotorchan/models/uncertain/uncertain_categorical.py)から API と処理を確認してください。

過去の設計条件・数式・検証要件は[設計記録](../../.github/project-records/model-design/uncertain_categorical_gp.md)に原文を保存しています。そこでの将来計画や Phase の完了条件は現在の対応保証ではありません。現在のモデル選択は[モデル索引](README.md)を参照してください。
