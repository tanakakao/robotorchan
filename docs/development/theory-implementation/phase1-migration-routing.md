# Phase 1 — 既存Mappingの移行先設計とAppendix保全方針

> この文書は移行設計の初版です。元の5ファイルは変更せず、章別移行の前に見出し単位・行範囲単位の対応を確定します。

## 基本方針

理論章に自然に配置できない情報を、無理に本文へ押し込んだり削除したりしません。横断的な契約・レジストリ全件照合・静的監査・検証限界などは **Appendix** として保全します。

- **章別ガイド**: 理論に対応する実装の仕組み、数学的意味、利用例、制約。
- **Appendix**: 複数章にまたがる情報、レジストリ全件表、公開API照合、共通契約、例外一覧、上流委譲の横断的分析、章別本文では冗長になる詳細な技術資料。
- **verification/**: 独立数値参照、実際のassertion、条件、許容誤差、未検証事項。
- **audit/**: Phase/PR/CI/過去の完了判定、移行証跡。
- **旧Mapping**: すべての情報が照合されるまで不変の移行元。

## 提案ディレクトリ（Phase 2で確定）

```text
docs/development/theory-implementation/
  README.md
  01_...md ～ 24_...md
  acquisition/
  optimization/
  verification/
  audit/
  appendix/
    README.md
    model-registry-and-public-api.md
    model-composition-and-delegation.md
    acquisition-contracts-and-compatibility.md
    optimizer-dispatch-and-constraints.md
    cross-cutting-implementation-contracts.md
    limitations-and-unverified.md
```

ファイル名は暫定であり、既存文書との責務重複をPhase 2で調査してから確定します。

## 初期の見出し・行範囲別移行設計

| 移行元 | 見出し・行範囲 | 第一候補 | 保全上の注意 |
| --- | --- | --- | --- |
| `README.md` | 1–107 | README / appendix | Source of truth、Mapping契約、ナビゲーションを残す |
| `README.md` | 108–278 | audit / verification | 12 Phaseの歴史的な完了判定と条件付き判断を保全 |
| `models.md` | 1–798 | appendix / audit / 各章 | レジストリ全件・公開export・分類の個別契約を保全 |
| `models.md` | 799–1168 | 各モデル章 / appendix | 代表的な数学・実装・テストの対応を保全 |
| `models.md` | 1169–1796 | 各モデル章 / appendix / audit | family別全件照合、52件/27件などの集計、Kronecker参照を保全 |
| `acquisition.md` | 1–78 | acquisition/ / appendix | registry・獲得関数とoptimizerの境界 |
| `acquisition.md` | 79–405 | acquisition/ / appendix | AL、分類AL、MC、MO、制約、lookaheadの各方式 |
| `acquisition.md` | 406–465 | appendix / audit | coverageと未検証・引継ぎ事項 |
| `optimization.md` | 1–181 | optimization/ / appendix | dispatch、制約意味論、埋め込みと座標復元 |
| `optimization.md` | 182–355 | optimization/ / appendix | named backend、specialist、stateful strategy、制約不変条件 |
| `verification.md` | 1–291 | verification/ / audit | modelのproperty-specific検証と数値許容誤差 |
| `verification.md` | 292–407 | verification/ / audit | acquisition reference、sample-level検証 |
| `verification.md` | 408–510 | verification/ / audit | optimizer/TuRBO/制約の数値参照 |

この表は**初期ルーティング**であり、移行済み判定ではありません。各区間をさらに全H2/H3見出しごとに照合し、複数移行先がある場合はprimary/secondaryを明示します。

## 消してはいけない重要情報

1. `models.md` のregistry-wide reconciliation、非公開/公開APIの差、分類モデル・回帰モデルの各variantの差。
2. Mixed modelごとのカテゴリ処理の違い。native kernel / one-hot / embedding / continuous-only passthrough / validation-onlyは同一視しない。
3. `acquisition.md` のAL、classification AL、outcome constraints、one-shot、information-theoretic acquisitionの意味論と適用限界。
4. `optimization.md` のcandidate constraintとoutcome constraintの区別、backendsごとのfeasibility enforcement、探索座標と公開座標の変換。
5. `verification.md` の検証対象property、reference formula、dtype、seed、tolerance、テスト名、結果と検証対象外の範囲。
6. 歴史的なCI/PR/Phaseの記録。本文からは分離しても、監査記録には保持する。

## 保全台帳の必須列

`source_file`, `source_heading`, `source_line_start`, `source_line_end`, `source_blob_sha`, `content_type`, `primary_destination`, `secondary_destinations`, `evidence_links`, `migration_status`, `review_status`, `notes`。

`migration_status` は `unmapped` / `planned` / `migrated` / `verified` とし、**verifiedにならない限り元情報を削除しない**。

## 次の監査ゲート

- 全H2/H3見出しに台帳行を割り当て、無所属区間をゼロにする。
- Appendixへ置く理由を明示し、章別ガイドから逆引きリンクを設ける。
- 既存参照リンクとCIの依存先を検査する。
- 元文書と移行先の技術的assertionを照合する。

## Phase・PR・CI履歴の仕分け規則（原文監査・2026-10）

この節は移行時の**情報保存ルール**であり、既存文書の削除や改名を指示しない。見出しに`Phase`や`audit`が含まれていても、本文が数学・実装・テスト条件を説明する場合は**技術内容を章に移し、履歴情報のみauditに分ける**。見出し名の正規表現だけで全文を履歴扱いしない。

| 原文と位置 | 本文として残すもの | Appendix/auditに分離するもの | 注意 |
| --- | --- | --- | --- |
| `README.md` 1–23,108–136 | `docs/theory`、`docs/models`、`docs/optimization`、tests、notebooks、coverage artifactの正本責任。staticと数値検証の証拠水準 | 旧**12 Phase**のベースラインSHA、PR #1361/#1366/#1400/#1408/#1411等、完了判定表 | 現在の**18 Phase**計画と旧12 Phase開発サイクルは別。数字を置換しない |
| `README.md` 175–231 | 読者導線、5つのcanonical mappingファイル、双方向リンク契約、構造的チェックと数値検証の差 | PR #1444–#1448、当時のCI成功、Phase 10/11 exit decision | 移行前に既存anchor/リンク依存を確認。現時点では移動しない |
| `README.md` 232–278 | 未検証のheteroskedastic posterior、HV/制約/情報獲得、optimizer収束などの限界 | 旧Phase 1–12のdeliverableと条件付き完了判定 | 過去の`Phase 12 in progress`を現在の進捗と誤認しない |
| `models.md` 5–31,107–119,120–185 | Registryとpublic exportの差、分類model IDと回帰class名の違い、生成coverageの正本 | 旧Phase 2 inventory/audit履歴、監査実行時の件数 | `142+29=171`は監査時点の値 |
| `models.md` 236–393,589–606,774–798 | evidence level、source/test/static scanの証拠能力と限界 | コマンド実行記録、当時の検出件数、Phase sign-off | 静的検出をruntimeや数値同値性へ格上げしない |
| `models.md` 799–1168,1225以降 | 個別モデルの数理・API・ownership・検証限界は**モデル章**に保持 | 見出し中の旧Phase番号、PR番号、CI履歴 | `Phase 4`を含むだけで数学的MappingをAppendix送りにしない |
| `models.md` 1169–1224 | variant別未検証事項、family→variantの追跡要件 | 旧Phase 4 exit gate・当時の作業待ちリスト | 旧時点の`in progress`と後続READMEの`Complete (static scope)`は時点差として両方保全 |
| `acquisition.md` 1–78,406–465 | registry metadata≠runtime、local AL/native BoTorch/compositionの区別、数値未検証事項 | 旧Phase 3/5 exit criteria・21 entryの時点集計・PR/CI | 79–405の`Phase 5`見出し配下は**獲得関数の技術本文** |
| `optimization.md` 1–19,86–181 | optimizerの11名dispatch契約、capability/test/feasibilityの証拠水準 | 旧Phase 3 audit queue、exit gateとPR/CI履歴 | 182–355の`Phase 6`見出し配下は**optimizer/strategyの技術本文** |
| `verification.md` 1–64,149–183,223–244,274–291,376–407,476–510 | property別の数式・数値条件・未検証範囲、負のAPI契約 | 旧Phase 7–9のexecution gate、PRごとのCI/merged状態、exit decision | 65–148、184–222、245–273、292–375、408–475も検証の技術本文。見出しにPhaseがあっても移す |

### 時点差と矛盾を消さない

- `models.md`には旧Phase 4が未完了とする**中間時点の記録**があり、`README.md`には後日の`Complete (static mapping scope)`がある。移行時は前者を誤記として黙って削除せず、監査時点と証拠水準を明記して履歴に残す。
- `README.md`の旧Phase 12 `in progress`は元の開発サイクルの条件付き記録であり、今回のPhase 1進捗ではない。旧Phase 7–9の`Complete`も代表的propertyの範囲内での判断。
- `verification.md`のPR #1426 final-head CI pendingのような記述は**その時点の事実**として残し、後続のPRや実装状態から現在のCI成功を推測しない。

### 移行実行前の保全ゲート

1. 原文5ファイルの内容とリンクを固定し、移動先ごとに`source file + heading + line range`を残す。
2. 数式・API・limitations・test evidenceを章側へ移したか確認し、旧PR/Phase/CIはAppendix/audit側で検索可能にする。
3. 既存`README.md#reader-navigation-phase-10`、`verification.md#phase-7-final-exit-reconciliation`等のanchorと、`scripts/check_theory_implementation_mapping.py`の5ファイル契約を壊さない。
4. リンクとmapping checker、docsテストを実行して初めて移行完了と判定する。**この監査更新ではCI未実行**。

**残作業:** 見出しごとの完全な行単位仕分け、全参照の存在/anchor確認、章の確定設計。Phase 1完了判定は保留。
