# 横断実装：モデル・獲得関数・候補最適化の境界

[BOのTheory](../../theory/01_bayesian_optimization.md)で説明する反復を、
別々の所有者を持つ実装へ対応付けます。各部品の内部計算は[BOの主章](../01_bayesian_optimization.md)と対応する章へ分け、
ここでは部品を越えてデータが渡る境界を説明します。

## 観測・予測・utilityは異なるデータ

[SingleTaskGP](../../../src/robotorchan/models/standard/single_task.py)は
観測を上流モデルへ渡し、[共通mixin](../../../src/robotorchan/models/base.py)が入力時点のraw snapshotと学習目的への接続を提供します。
学習後の`posterior()`はモデルに保持されたデータとパラメータを使います。

獲得関数は候補Xに対する分布からutilityを計算し、候補optimizerはこの計算を繰り返してXを選びます。その値は新しい実験の観測ではありません。
観測取得とデータの追加を含む反復は呼び出し側が所有し、モデルや公開dispatcherが暗黙に実験を実行する構造ではありません。

## 異種出力を獲得関数へつなぐ

[HeterogeneousModel](../../../src/robotorchan/models/heterogeneous.py)は
出力位置と子モデルの所有関係を保持します。回帰のlatent分布と分類確率を、単一のGaussian出力として連結するcontainerではありません。

[composition.py](../../../src/robotorchan/acquisition/composition.py)は
目的の出力位置・方向と、分類可行性を解決してBoTorch acquisitionへ接続します。回帰sampleに適用する連続制約と、外側から重み付けする分類PoFは別の経路です。
計算の詳細は[分類](../24_classification.md)と[制約付き獲得関数](../acquisition/08_constraints.md)へ分けます。

## 探索変数と公開座標を分ける

[dispatch.py](../../../src/robotorchan/optim/dispatch.py)は変数型・候補制約を確認し、
数値backendへ分岐します。モデルの予測分布や獲得関数の数式は変更しません。埋め込みstrategyは内部座標で探索しても最終候補を元空間へ戻します。
ただしacquisition自体が評価される空間はstrategyごとの設計に依存します。

one-shotの補助decision、未完了のpending点、実験する候補も別の表現です。
[Batch / one-shot](../optimization/08_batch_one_shot.md)は補助行の抽出、
[高次元探索](../20_high_dimensional_search.md)は射影、
[TuRBO](../optimization/10_trust_region.md)は完了観測による状態遷移を担当します。

## 使用手順と実装を読む資料

実際の観測追加・学習・候補生成・可視化は、既存の
[End-to-End Notebook](../../../examples/notebooks/36_end_to_end_workflows.ipynb)にあります。
このページでは同じ実行手順を再掲しません。クラスと呼び出し経路は上記のソース、公開引数の詳細は
[モデル資料](../../models/README.md)と[候補最適化資料](../../optimization/README.md)を参照してください。
