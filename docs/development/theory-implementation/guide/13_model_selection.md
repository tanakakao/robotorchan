# Model Selection：問題設定から実装を選ぶ

## 1. 選択の出発点

モデル名より先に観測形式・入力構造・推論方式・意思決定目的を決める。[理論第13章](../../../theory/13_model_selection.md)と[利用者向けモデル選択](../../../getting_started/model_selection.md)が基礎となる。通常の連続scalar回帰は `SingleTaskGP` を比較基準にするが、比較観測・分類・構造化出力を無理に通常回帰へ変換しない。

次の順に確認する：観測はnumericかpairwiseかbinaryか／outputはscalarかvectorかtensorか／categorical・hierarchy・contextがあるか／taskとfidelityを区別できるか／異分散・外れ値・入力不確実性のどれが問題か／大きいのは入力次元dか標本数nか／獲得関数が必要とするposterior演算は何か。

## 2. モデルの数学的仮定と公開API

| 問題設定 | 主なモデル | 実装・推論で重要な違い |
| --- | --- | --- |
| 通常の連続回帰 | `SingleTaskGP` | BoTorch Exact GPを継承、robotorchanはraw data保持・fitを拡張 |
| Mixed入力 | `MixedSingleTaskGP` | categorical kernelと連続kernelを区別。カテゴリの数字は通常距離ではない |
| Fidelity構造 | `SingleTaskMultiFidelityGP` | fidelity featureとtarget fidelity、評価費用を分離 |
| 関連task | `MultiTaskGP` / `KroneckerMultiTaskGP` | long formatとblock designは別の観測契約 |
| 独立複数output | `ModelListGP` | 出力間の統計的共分散が必要か検討 |
| 比較観測 | `PairwiseGP` | Gaussian numeric likelihoodではない |
| 構造化tensor | `HigherOrderGP` | posteriorのtensor軸とsamplingを確認 |
| 大標本数n | `SingleTaskVariationalGP` | inducing pointsとVariationalELBO |
| 高次元・軸sparsity | SAAS / MAP-SAAS family | Fully Bayesian NUTSとMAP推定は異なる |
| 入力低次元表現 | PCA / PLS / Neural reduced GP | frozen reducerとjoint encoderの学習契約が異なる |
| 位置依存noise | Heteroskedastic / Joint heteroskedastic GP | latent不確実性とobservation noiseを区別 |
| 外れ値 | Student-t / Relevance Pursuit | heavy-tailedとsparse outlierは異なる生成機構 |
| 二値分類 | `BinarySingleTaskGPClassifier` | latent posterior ≠ class probability |
| Non-GP tabular | Random Forest / Boosting | ensemble disagreementはGP posteriorではなく入力微分にも制約 |

各公開モデルのconstructorや未対応事項は[モデルガイド](../../../models/README.md)と[旧モデルMapping](../models.md)で確認する。表中の掲載は特定のモデル・獲得関数・optimizerの組合せが検証済みという意味ではない。

## 3. Model → Acquisition → Optimizerの接続

標準の単目的BOは `SingleTaskGP` のposteriorからBoTorch `LogExpectedImprovement`などのutilityを構成し、`robotorchan.optim.optimize_acqf`またはBoTorch optimizerで候補を生成する。`best_f`は最大化する目的の尺度を使う。[獲得関数のImprovement](acquisition/02_improvement.md)と[Gradient optimizer](optimization/02_gradient_based.md)に接続する。

観測ノイズがある場合はX_baselineのlatent uncertaintyを扱うqLogNEI系、複数目的ではobjective尺度・reference point・Pareto partitioningを定義するEHVI/NEHVI系を検討する。複数outputがあるだけで多目的とは限らず、outcome constraintsと既知candidate constraintsも層が違う。[Batch/Noisy](acquisition/04_batch_noisy.md)、[多目的](acquisition/07_multiobjective.md)、[制約](acquisition/08_constraints.md)を参照する。

回帰+分類の意思決定では分類 `posterior(X)`のlatent値を可行確率と混同せず `predict_proba(X)`を使う。`ProblemSemantics`とacquisition compositionの責務は[Classificationガイド](24_classification.md)と[End-to-Endガイド](walkthrough/README.md)で確認する。

高次元ではモデル側のSAASやReducedGPと、探索側のTuRBO / REMBO / HeSBO / ALEBO / BAxUSを別々に選ぶ。[高次元モデル](08_high_dimensional_gp.md)と[高次元探索](20_high_dimensional_search.md)は異なる座標系を扱う。入力次元dが大きい問題と標本数nが大きい問題は同じではない。

## 4. 実行例から辿る

[統合APIスクリプト](../../../../examples/unified_api_workflow.py)では二出力Exact GPのfit、BoTorch sampler/objective、qSimpleRegret、Sobol candidateをつなぐ。[End-to-End Notebook](../../../../examples/notebooks/36_end_to_end_workflows.ipynb)と[横断ウォークスルー](walkthrough/README.md)を併読する。前者のスクリプトはcandidate proposalまでであり、真の評価を追加する閉ループ全体とは区別する。

## 5. 検証の境界

モデルを作れることと、posterior sample、fantasize、q-batch、one-shot、mixed constraints、dtype/deviceまで互換であることは別。実装の分類はImplemented / Extended / Delegated / Approximated / Unsupported / Unverifiedを使い分ける。[検証索引](verification/README.md)では数式の独立数値検証、契約テスト、CI成功を分ける。旧Mappingは移行台帳の原文区間照合が済むまで保全する。