# 反復測定による観測ノイズ

同一の設計条件で反復測定がある場合、条件内の変動から観測ノイズを推定できます。モデルは元入力の完全一致した行をまとめ、群平均を応答とし、不偏標本分散を反復数で割った平均値の分散を GP に渡します。

## 利用方法と制約

`ReplicateNoiseSingleTaskGP`、`MixedReplicateNoiseSingleTaskGP`、`ReplicateNoiseMultiFidelityGP` を入力構造に応じて選びます。`train_Yvar` の意味は個々の測定値の分散ではなく群平均の観測分散です。近似一致によるグループ化は仮定せず、丸め誤差のある入力や fidelity の異なる行を同じ群とみなさないでください。

## 理論・実装・検証を辿る

[理論](../theory/15_heteroskedastic_noise.md)と[実装ガイド](../guide/15_heteroskedastic_noise.md)で仮定・推論方法・近似を確認できます。[ソース](../../src/robotorchan/models/robust/replicate_noise.py)から API と処理を確認してください。

過去の設計条件・数式・検証要件は[設計記録](../../.github/project-records/model-design/replicate_noise_gp.md)に原文を保存しています。そこでの将来計画や Phase の完了条件は現在の対応保証ではありません。現在のモデル選択は[モデル索引](README.md)を参照してください。
