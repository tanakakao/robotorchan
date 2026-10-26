# 日本語理論・実装ガイド：最終監査と条件付き完了判定

> 監査対象：固定18 Phase・48章の技術文書整備。対象PRの先頭は #1466（最終HEAD `e5b16e05d63300d8925df0d31e2d528f807a3afe`）。この文書は**現時点の証拠範囲を明記する判定**であり、確認していない数値実験を成功と記載しない。

## 目的と判定

**判定：構造・CIレベルは達成。技術内容の完全な移行と実行の最終完了判定は保留。**

固定48章の理論対応と旧Mapping5ファイルの存在は、`scripts/check_theory_guide_quality.py`と`tests/docs/test_theory_guide_quality.py`が検査する。PR #1466最終HEADの**CIとDocumentation linksがともにsuccess**であることをGitHubワークフローで確認した。元Mappingの完全な置換、数式の全property検証、全コード例の実行を保証するものではない。

## 最終ゲートごとの証拠

| 判定対象 | 観察した根拠 | 判定 |
| --- | --- | --- |
| 理論に対応する48章 | Main24 + acquisition12 + optimization12。チェックscriptの固定basename照合 | 達成（章の存在・見出し・theoryリンクの構造） |
| 旧Mapping保全 | `README.md`、`models.md`、`acquisition.md`、`optimization.md`、`verification.md`の存在をCIで確認 | 達成（存在チェック）；全区間の内容比較ではない |
| 追加利用導線 | `guide/walkthrough/README.md`、`guide/verification/README.md`、`guide/13_model_selection.md`などを作成 | 配置済み。実例の実行は個別検証が必要 |
| Markdownリンクとアンカー | `.github/workflows/documentation-links.yml`で既存docsリンク・Notebook・referencesと新規48章checkerを実行 | PR #1466 HEADで成功 |
| パッケージと全体CI | GitHub CI workflow | PR #1466 HEADで成功。CI範囲外の実験・環境を含まない |
| 数学的検証 | `guide/verification/`を作成し旧`verification.md`のproperty別参照を保持 | 検証索引の配置は達成。全理論の独立参照完了ではない |
| 旧Mappingの区間移行 | [Phase1保全監査](phase1-preservation-audit.md)では174見出し候補、H4/表/数式の区間照合の未完了を報告 | **未確認・要照合** |
| Source/Test/Notebook対応の全件 | 旧MappingおよびPhase1の移行記録に部分的な静的監査 | **未確認**（全行の実在とassertion照合未証明） |
| 学習本文からPhase/PR/CI履歴を除去 | 主な新規ガイド本文では分離方針を採用 | **未完了**：旧Mappingに履歴が残る。原文保護のため未削除 |
| コード例の実行 | 一部の既存Notebookや統合scriptはCIの対象。新規ガイドの全コード例単体実行を確認した記録なし | **未確認** |
| 重複・不要ファイル | 旧Mappingを移行元として保持する設計 | **未完了**：全区間の照合まで整理・削除不可 |

## 学習用ガイドとして必要な残検証

- 原文の全見出し・表・数式・negative contractを移行台帳の区間へ対応付け、各行に`primary_destination`、必要なら`secondary_destinations`、具体的なassertion／参照先、`review_status`を記録する。現存するのは構造的対応であり、内容の完全移行と同義ではない。
- Model / Acquisition / Optimizerの各章にある具体的なAPI名・引数・posterior semanticsを実際のsrcおよびテストから再照合する。BoTorchへの`Delegated`とrobotorchan側の`Implemented/Extended`は推測で埋めない。
- 実行例は新規章ごとにparse/import/最小実行を検査し、CPU/float64/seed、依存環境、成功・失敗記録を保存する。CIが緑でも非実行のコードブロックまで実証済みとは扱わない。
- 履歴を学習本文から取り除くのは**技術内容を再配置した後**とする。旧原文を単純に短縮・削除するのは情報喪失になるため禁止する。
- `guide/README.md`の「予定」表現は48章作成の実態と齟齬があったため、索引表現を今回修正した。章内に残る他の時間依存表現は別途監査する。

## 今回の完了条件の扱い

固定計画には新しいPhaseを追加しない。上記の保留事項は**Phase18の最終監査ゲートの未達**として記録する。構造・リンク・CIの達成を維持しながら、意味論的移行検査・コード例実行・文書重複の解消を最終ゲートとして再判定する。

関連：[48章ガイド索引](../../../docs/development/theory-implementation/guide/README.md)、[元Mapping](../../../docs/development/theory-implementation/README.md)、[検証索引](../../../docs/development/theory-implementation/guide/verification/README.md)、[移行先設計](phase1-migration-routing.md)。

## PR #1467 の実行ゲート確認（2026-10-26）

Phase18の文書修正HEAD `07b459f98555ea5d147040cf46950934c4a0f1e1` に対し、次をGitHub Actionsで確認した。

| Workflow | Run ID | 結果 | 証拠の意味 |
| --- | --- | --- | --- |
| Documentation links | `40296208635` | **success** | Markdownリンク・参照・48章構造など、ワークフロー内検査の成功 |
| CI | `40296208646` | **success** | 現在のCIで対象としたlint / tests / notebook jobs等の成功 |

PR #1467はこの記録時点で**Open（未マージ）**である。したがって、完了をmainに反映済みとは扱わない。CI緑は変更内容をマージ可能とする必要条件の一つであり、174見出しの意味論的移行確認、H4以下の数式・negative contract保全、すべての新規コード例の実行完了を示さない。

[既存の見出し移行台帳](phase1-heading-preservation-ledger.md)は**174見出しの仮割当**で、各レコードが`planned (review required)`とされている。これを目視・機械照合したという記録なしに`verified`へ更新しない。

### Phase18で追加のゲートを閉じる際の順序

1. **台帳の内容検収：** 見出し区間の本文、H4・数式・表・参照・exception・未対応情報を原文と照合する。primaryとsecondary移行先を明示し、単にリンクを置いただけでは`migrated`にしない。
2. **APIとテスト：** 各ガイドの使用例について、コード上の公開export・constructor・実行を検査し、別途numerical propertyのassertion根拠を確認する。
3. **監査履歴の整理：** 学習用本文に歴史的なPhase/PR/CIが混入していないかを確認し、必要な技術説明を残したままauditに分離する。
4. **最終再判定：** CIとリンクが緑であるだけでなく、上記の台帳・実行証拠・情報非喪失が説明できた時点でのみPhase18全体を`complete`とする。

これらは固定Phase18の**未解決ゲート**であり、新しいPhaseの追加ではない。

## 原文区間の実照合：Models Registry（部分達成）

[旧Model Mapping](../../../docs/development/theory-implementation/models.md)の行18–91を直接取得し、以下を照合した。

| 原文区間 | 調査した事実 | 対応先 | 移行判定 |
| --- | --- | --- | --- |
| 18–31「Registry boundaries」 | 回帰registry監査時点142件／二値分類29 ID、異種modelはcoverage外、分類のhelper exportとmodel IDの区別 | [Model registry appendix](../../../docs/development/theory-implementation/guide/appendix/model-registry-and-public-api.md) | **partial, review required**：意味論は対応したが現在mainの独立カウントは未実施 |
| 32–51「Regression model inventory by existing guide」 | 11 familyの集計（合計142件）と各モデル利用ガイドへの参照 | 同Appendix | **partial, review required**：全142 model名と個別notebook/theoryの全行照合は未実施 |
| 52–91「Binary classification registry inventory」 | 分類IDとpublic classの名前空間の差。binary.kronecker_multitaskはExact Kroneckerを意味しない | 同Appendix・[Classification](../../../docs/development/theory-implementation/guide/24_classification.md) | **partial, review required**：全29 IDのassertion証拠の移管・全件検収は未実施 |

確認した本文は新しいAppendixに概念別にまとめたが、**原文のinventory全件をコピー・移行したものではない**。既存の174見出し台帳上の`planned (review required)`を`verified`へ変更せず、重複情報があるという理由で旧Mappingを削除しない。原文のsource blob SHA、詳細なテスト証拠、各レコードの完全な再配置は引き続き未達である。

## Acquisition Active Learningの実装照合（部分達成）

旧`acquisition.md`の「Distinct implementation paths」「Phase 5 — regression active-learning acquisition mappings」「Phase 5 — classification AL and Gaussian EPIG」を読んで、現行の`src/robotorchan/acquisition/active_learning/variance.py`、`classification.py`および`registry.py`のソースと照合した。[ALガイド](../../../docs/development/theory-implementation/guide/acquisition/09_active_learning.md)へ具体的な数式・クラス・posterior/predict_probaの違い・q=1・拒否条件を反映した。

**確認した範囲：** `PosteriorVariance`、`PosteriorStd`、`PredictiveEntropy`、`MarginUncertainty`、`ProbabilityVariance`、`BALD`、`LatentStraddle`の責務、ならびにregression ALのstructured/ensemble guardとclassification model interface。数値テストをこの作業で再実行したわけではなく、旧原文の全行・全entryの移行を検収したわけでもない。原文見出し台帳は依然`review required`のまま保全する。

## Constrained AcquisitionとOptimizer接続の部分照合

旧`acquisition.md`の「Distinct implementation paths」「Acquisition/optimizer interaction」の技術的主張に対し、`src/robotorchan/acquisition/composition.py`および`optimizer_compatibility.py`のソースを直接照合した。[制約付きAcquisitionガイド](../../../docs/development/theory-implementation/guide/acquisition/08_constraints.md)にsemantic binding、sample shape、分類可行性の独立性近似、candidate/outcome制約の区別、one-shot/mixed静的互換性の限界を追加した。

**状態：partial / review required。** 本作業では全constrained helperの数値的同値性、非線形候補制約の各backendでの実行、joint feasibility相関モデルは独立テストしていない。旧Mappingの原区間の数式・例外・テストパスを全件照合して`verified`にする作業は保留する。

## Optimizer Candidate Constraints：原文・ソース・ガイドの部分照合

旧`optimization.md`の「Baseline and separation of responsibilities」「Named optimization dispatcher」「Candidate-space constraints versus outcome constraints」の区間を、`src/robotorchan/optim/constraints/contracts.py`、`validation.py`および`dispatch.py`と比較した。確認対象は11 named backends、`CandidateConstraints`の線形・非線形tupleと符号、intrapoint/interpointの入力shape、`sequential=True`の拒否条件、非線形初期点・公開座標/埋め込み座標の違いである。技術ガイド[Candidate constraints](../../../docs/development/theory-implementation/guide/optimization/05_constraints.md)へ具体的な実装契約を反映した。

**移行判定：partial / review required。** この調査は制約関連のソース照合であり、全backendのconstraint enforcementの数値実行や、旧`optimization.md`の残りのstrategy/one-shot/NSGA-II区間の意味論的照合を完了したものではない。旧Mappingと174見出し台帳は変更しない。

## TuRBOの状態更新：実装と旧Mappingの部分照合

旧`optimization.md`の「Stateful and coordinate-changing algorithms」「Inspected optimizer and strategy assertion evidence」を読み、`src/robotorchan/optim/trust_region/turbo.py`の`TuRBOState`、`update_turbo_state`、`restart_turbo_state`、`generate_turbo_restart_center`の実装本文を直接照合した。実装で定義されたbatch observations/state_values区別、relative improvement閾値、success/failure counters、region lengthの拡縮、restart、ARD重みの幾何平均正規化を[Trust Regionガイド](../../../docs/development/theory-implementation/guide/optimization/10_trust_region.md)へ詳記した。

**判定：partial / review required。** 数式・sourceは照合したが、今回新たなテスト実行、全TuRBO optimizer/Thompson/pending/mixed pathのassertion-level完全照合、regret評価は行っていない。旧原文と174見出し台帳を保全する。

## Mixed one-shot：augmented batch契約のソース照合（部分達成）

旧`optimization.md`の「Specialized optimization paths: inspected source and assertions」と、`src/robotorchan/optim/one_shot/{initialization,mixed}.py`を直接比較した。[Batch / One-shotガイド](../../../docs/development/theory-implementation/guide/optimization/08_batch_one_shot.md)へ`get_augmented_q_batch_size(q)`と実候補qの区別、`extract_candidates`、one-shot初期点のshape、mixed one-shotの行別カテゴリ割当ての列挙（組合せ数`m ** augmented_q`）、`q=1`制限、`max_assignments`上限、BoTorchへの連続部分問題委譲を追記した。旧Mapping記載の`tests/optim/one_shot/test_mixed.py`はassertion追跡先として維持した。

**判定：partial / review required。** 本作業はsrc記述を確認した文書照合であり、独立の混合KG数値検証や全候補制約の組合せ、全one-shotアルゴリズムの性能検証ではない。固定174見出し台帳は保全する。

## NSGA-II：vector objectiveとscalar acquisitionの境界を照合

旧`optimization.md`の「Specialized optimization paths: inspected source and assertions」、`src/robotorchan/optim/backends/nsga2.py`および`tests/optim/backends/test_nsga2_backend.py`の実装・テスト本文を確認した。[Evolutionaryガイド](../../../docs/development/theory-implementation/guide/optimization/07_evolutionary.md)に、NSGA-IIが11種のnamed acquisition dispatcherに含まれないこと、`[population,d] → [population,m]`（`m>=2`、全目的最大化）の契約、nondominated ranking／crowding／選択・交叉・突然変異、返却される近似Pareto集合とテストの限界を明記した。

**判定：partial / review required。** この照合はソースとテストの静的確認であり、今回独立にNSGA-IIを実行した結果やhypervolume収束の証明ではない。旧Mappingは引き続き保全する。
