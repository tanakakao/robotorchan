# Robust Gaussian Process：理論と実装

## 背景と理論の入口

外れ値に頑健であること、ノイズが入力依存であること、入力自体が不確かなことは別問題である。観測の大きな残差にはStudent-t heavy-tail、contamination mixture、sparse relevance pursuitなど異なる確率機構を使う。データを無条件に削除する処理とは異なり、尤度・異常観測の重み・潜在応答への影響をモデル化する。

数式と確率的仮定の正本は[対応する理論章](../../../theory/14_robust_gaussian_process.md)を参照。本章ではrobotorchanの実装責務・推論・制約に重点を置く。

## 実装アーキテクチャと責務

- **Delegated/Extended:** `src/robotorchan/models/robust/robust.py`の`RobustRelevancePursuitMultiTaskGP`はBoTorchの`RobustRelevancePursuitMixin`を利用し、multi-taskのtask列とローカルなraw観測契約を維持する。
- **Implemented:** Student-t系列は非Gaussian尤度とvariational objectiveを使う。標準exact Gaussian MLLを流用しない。
- **Implemented/Extended:** contaminated系列は名目観測と混入観測の混合をモデル化する。明示的な外れ値ラベルの正解を返すとは限らない。
- RRPではsparse outlier relevanceの学習契約を個別に確認し、通常の`SingleTaskGP.fit()`と同じ計算路を仮定しない。

## APIと実装を読むための最小例

```python
from robotorchan.models import StudentTSingleTaskGP

print(StudentTSingleTaskGP.__name__)
```

コード例は公開APIを示す静的な例であり、このセッションで実行して成功したとの主張ではない。

## 適用条件、テスト、限界

Student-tでは有限分散の自由度条件、観測ノイズとlatent uncertaintyの分離、variational fittingの収束を確認する。relevance pursuitは支持集合の復元精度を別途測る。モデル名やshape testのみから異常検知性能、校正、頑健性を結論付けない。

旧[モデルMapping](../models.md)・[数学的検証Mapping](../verification.md)を残し、学習ガイド本文に過去のPR/CI履歴を混入させない。[モデル選択ガイド](../../../models/robust_noise.md)には利用者向けの比較を記す。

## Robust relevance pursuitの実装と推論の切り分け

外れ値に頑健なモデルを比較する際は、尤度のheavy-tail、混合汚染、観測ごとのrelevance追跡を区別する。どれもGaussian GP posteriorの標準分散を増やすだけの同一アルゴリズムではない。

src/robotorchan/models/robust/robust.pyのRobustRelevancePursuitSingleTaskGPはBoTorchRobustRelevancePursuitSingleTaskGPを継承する。一方、RobustRelevancePursuitMultiTaskGPはMultiTaskGPとBoTorchのRobustRelevancePursuitMixinを組み合わせ、base likelihoodにoutlier-related noise構造を付加する。MultiTask版はlong-formatのtask_featureを明示的に受け取る。

MultiTask版のto_standard_model()は現在の学習／評価モードを保持しつつ、対応するMultiTaskGPを再構築する補助経路である。raw observations・task列・選択したcovar_moduleなどは継承するが、robust推論の全操作が通常のSingleTaskGP.fit()と同一になるわけではない。

MixedRobustRelevancePursuitMultiTaskGPはtask_featureを構造列として扱うため、cat_dimsと重ねるとnormalize_feature_dimsが拒否する。カテゴリ共分散の構築にはmake_mixed_covar_moduleを利用する。外れ値重みの推論やsupport選択は通常の最大周辺尤度の一段階最適化と混同しない。

**検証境界：** robust.py内のclass継承、mixin初期化、task列とカテゴリ列の検証を確認した。外れ値支持集合の復元精度、likelihood同定、実測ノイズ下の較正や数値的頑健性の比較は今回実行していない。
