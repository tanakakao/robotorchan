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
