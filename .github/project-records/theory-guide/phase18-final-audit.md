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
