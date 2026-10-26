# Optimizer：探索解と数学的検証の違い

対応ガイド：[最適化基礎](../optimization/01_foundations.md)、[Candidate constraints](../optimization/05_constraints.md)、[高次元探索](../20_high_dimensional_search.md)、[TuRBO](../optimization/10_trust_region.md)。実装の正本：[旧Optimization Mapping](../../optimization.md)、独立数値参照の正本：[Verification Mapping](../../verification.md)。

## 検証項目の分離

| Property | 独立に検査する内容 | shape / smoke testだけで足りない理由 |
| --- | --- | --- |
| Candidate feasibility | bounds、fixed features、linear/nonlinear inequality/equality、許容tol | 候補が返っても制約違反の可能性がある |
| Joint q-batch | candidate shape `[q,d]`、acquisition valueがjoint utilityを表す | pointwise top-qはjoint optimumと異なる |
| Integer / categorical | legal integersとcategorical valueの保存 | dtypeだけではカテゴリの離散性を表さない |
| Embedding / reconstruction | 元座標のbounds、写像・inverse/latent契約、seed | clippingやhashingが最適値を変えうる |
| TuRBO state | success/failure、length、restart、batch単位の更新 | stateが存在するだけではtransitionの正しさは言えない |
| Hybrid handoff | global `[q,d]`をlocal initial conditionsへ引き渡す | 一点ずつの上位候補とは意味が違う |

Phase 9相当の旧検証記録にはoptimizerの複数の独立参照・契約テストが記述されている。現段階ではパス・数値tol・assertionを再照合していない項目を、verifiedの網羅表として掲載しない。対応節は[旧Verification Mapping](../../verification.md)へ残す。

各探索手法が数値的に候補を生成することと、その候補が大域最適解であることは同義ではない。NSGA-IIがvector Pareto目的を扱う一方、EHVI等はscalar acquisitionである点も区別する。
