# DeepGP モデルと入力の条件

`SingleTaskDeepGP`、`MixedSingleTaskDeepGP`、`MultiTaskDeepGP`、`MixedMultiTaskDeepGP` は変分 Deep Gaussian Process モデルです。Exact GP wrapper とは推論方法が異なります。入力とタスクの構造に応じて選びます。

## 単一タスク・連続入力

`SingleTaskDeepGP` は確率的な GP 層の階層で単一タスク回帰を行います。BoTorch の獲得関数には Monte Carlo posterior を公開します。確率的な DeepGP の軌跡を厳密な Gaussian posterior と解釈しないでください。

## 連続・カテゴリ入力

`MixedSingleTaskDeepGP` は元の混合入力を受け取り、`cat_dims` を要求します。連続特徴を標準化し、カテゴリ特徴を**学習可能な embedding** に変換してから DeepGP 層へ渡します。整数のカテゴリ ID を連続距離として扱いません。

カテゴリ列の位置には負のインデックスも使えます。訓練データのカテゴリ値は、非負・整数・0 始まり・欠番なしである必要があります。連続特徴は少なくとも 1 列必要です。この embedding はモデル固有の表現であり、Exact GP の `MixedSingleTaskGP` が使うカテゴリカーネルとは異なります。

## 複数タスク・long-format 入力

`MultiTaskDeepGP` は long-format のタスク列を受け取り、標準化したデータ特徴とは別にタスク ID を embedding へ変換します。`posterior()` と `training_loss()` には元の long-format 入力を渡します。`train_Y[n, m]` を使う block-design モデルとは区別してください。

## 混合入力の複数タスク

`MixedMultiTaskDeepGP` は long-format の `task_feature` と、カテゴリのデータ列を表す `cat_dims` を組み合わせます。タスク ID とカテゴリ値には別々の embedding を学習し、連続データ特徴は標準化します。タスク列を `cat_dims` に含めることはできず、連続データ特徴が少なくとも 1 列必要です。訓練データのタスク ID とカテゴリ ID は、0 始まりで欠番のない整数にしてください。モデルには元の long-format 入力を渡します。

## 学習と獲得関数への接続

学習は変分目的を使い、Exact GP の周辺尤度を最適化する経路とは異なります。posterior sampling は確率的な層を保持し、テスト対象の形状条件に従ってバッチ候補を評価します。特定の候補探索が使えると仮定する前に、獲得関数と最適化 backend の互換性を確認してください。

ファミリーの選択と接続条件は [表現力の高いモデル](expressive.md)、[BoTorch との接続](expressive_botorch_integration.md) を参照してください。
