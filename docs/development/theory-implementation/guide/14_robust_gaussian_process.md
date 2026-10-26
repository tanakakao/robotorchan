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
