# Phase 1 — Documentation baseline and preservation inventory

> Status: **in progress**. This is an initial, non-destructive inventory, not an approval to migrate or remove existing content.
>
> Baseline: `main` commit `9de071c899dcce5fa2c7c3c0083281d529a3b077` (2026-10-25 23:57 UTC).

## Non-loss migration policy

1. Keep all five original `docs/development/theory-implementation/*.md` files unchanged during inventory and drafting.
2. Treat **every** mapping row, exception, upstream delegation, mathematical assertion, test path, numerical tolerance, limitation, and unsupported/unverified statement as preservation-critical.
3. Build a source-section-to-destination ledger before moving material. Record original heading/line range, new destination, preservation disposition, and review status. No unreviewed content may be discarded as redundant.
4. Separate technical explanation, verification evidence, and historical PR/CI/Phase records without conflating test existence with mathematical correctness.
5. Do not remove or replace legacy files until all entries in the ledger are reconciled and inbound links are checked. Preserve compatibility links or redirects where needed.
6. Record uncertainties explicitly; never infer implementation support from a class name or a mapping assertion alone.

## Baseline inventory (direct file reads)

| File | Characters | Lines | H1–H3 headings | `Phase N` mentions | PR-number mentions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `README.md` | 16,748 | 278 | 13 | 38 | 29 |
| `models.md` | 196,924 | 1,796 | 91 | 103 | 0 |
| `acquisition.md` | 32,345 | 465 | 18 | 29 | 0 |
| `optimization.md` | 26,725 | 355 | 21 | 22 | 0 |
| `verification.md` | 29,283 | 510 | 31 | 46 | 29 |

These counts are descriptive only: mention counts are not proof of historical-only content. All numbers are from direct fetched content, including blank lines.

## Theory-side navigation observed

- `docs/theory/README.md` defines **Part I–VIII** and chapter numbers **01–24**, with chapter 13 (model selection) grouped in Part VIII rather than numeric order.
- `docs/theory/acquisition/` search inventory: `README.md`, `01_foundations.md` through `12_selection_guide.md`, plus `classification-active-learning.md`.
- `docs/theory/optimization/` search inventory: `README.md`, `01_foundations.md` through `12_selection_guide.md`.
- Chapter 24 (classification) is present; the theory README's earlier “23 chapters” prose requires a consistency check before any editing.

## Early migration risk findings

- `models.md` is the largest preservation risk (~197k characters); dividing it by class names alone risks losing cross-family caveats and source/test traceability.
- Existing `verification.md` includes **property-scoped** mathematical references, numerical tolerances and explicit exclusions. Preserve those verbatim or with traceable equivalents; do not reduce them to “tests passed”.
- Existing `README.md` embeds prior 12-phase audit scope and historical baseline; archive the historical claim rather than treating it as the current 18-phase plan.
- Mixed categorical semantics differ by model family (native covariance, internal one-hot, embeddings, passthrough, validation only). Preserve these distinctions.
- Existing docs/theory README distinguishes model latent coordinates, search coordinates, and posterior semantics across surrogate families. Preserve those conceptual boundaries in Japanese guides.

## Phase 1 outstanding gates

- [ ] Obtain authoritative full recursive repository tree at the baseline SHA; verify all theory chapter filenames and docs/models, docs/optimization, notebooks, tests, and CI paths.
- [ ] Build full heading-level inventory and section-level preservation ledger for all five source files.
- [ ] Extract all inbound relative links, anchor references and CI dependencies; assess migration breakage.
- [ ] Inspect duplicate content and missing evidence without guessing.
- [ ] Record source-to-test-to-notebook traceability and explicit verification gaps.
- [ ] Review the completed audit and decide whether Phase 1 exit criteria are satisfied.

**Exit status: NOT YET COMPLETE.** No original file has been modified or deleted.

## Phase 1 統合完了判定（2026-10-26）

**判定：未完了（Phase 2への正式移行は保留）。** 理由は、原文の保全と移行設計は進んだが、原文の全区間・全参照の網羅性とリンクの実行検証が未確認であるため。18 Phase計画と元の5ファイルは変更しない。

| 当初の監査ゲート | 現在の証拠 | 判定 |
| --- | --- | --- |
| main baseline / 旧Mapping 5ファイルの保全 | `main` SHA `9de071c899dcce5fa2c7c3c0083281d529a3b077`、5ファイル・3,404行・292,025文字の記録。元ファイルは編集していない | 達成（静的） |
| H1–H3の見出し単位移行台帳 | 174見出しの候補行が台帳にあり、正式理論24章およびacquisition/optimization各12章のREADMEと照合した | 部分達成：174行すべて`review required`、H4/数式/表の移行先を未確定 |
| 技術内容とPhase/PR/CIの分離 | 5ファイルの履歴候補を仕分け、モデル・獲得関数・optimizerの意味論をレビュー。履歴見出し配下の技術内容を本文へ残すルール | 方針達成、行単位完全仕分けは未達 |
| Source/Test/Notebook evidence | 静的抽出178パス候補、ユニーク35パス直接取得、12テストのassertion本文をレビュー | 部分達成：残りの存在・参照解決・実行は未確認 |
| 旧アンカー/双方向リンク/CI依存 | 高リスクアンカー2件と6つのdocs CIコマンドを特定、旧5ファイル保持方針 | 依存調査達成、リンクチェッカー/CI実行は未達 |
| 完了判定の再現性 | 本節に根拠・未完了ゲート・次の作業を記録 | 本判定のみ達成 |

### 完了までの最短の残作業（追加Phaseは作らない）

1. **全174見出しの台帳を機械的に再照合**：集計行と見出し行を区別し、ファイル・行番号・見出し・移行先の一意性、欠落、重複を確認する。特に本文中のH4・表・数式・negative contractは区間単位で確認する。
2. **178候補参照の存在確認と内部リンク解決**：`src/`、`tests/`、`examples/`、`docs/`の候補を実際のrepo treeと照合する。存在しない参照は削除せず`stale/unverified`として残す。旧アンカーと相対リンクを検査する。
3. **docsの検証を実行できる環境で走らせる**：`scripts/check_docs_links.py`、`scripts/check_notebook_links.py`、`scripts/check_docs_references.py`、`scripts/check_theory_implementation_mapping.py`、関連unittestを実行し結果を記録する。実行できない場合は`not run`のままとし、完了を主張しない。
4. 上記の証拠をこの監査記録へ集約し、**Phase 1のexit gateを再判定**する。日本語技術ガイドの正式章構成確定は予定どおりPhase 2で行う。

### 計数上の注意

`phase1-heading-preservation-ledger.md`には174見出しの移行候補以外にファイル集計・説明表も含まれる。ファイル名から始まるMarkdown表行を単純に数えると183行となり、**183見出しではない**。見出し台帳は表の役割・列を区別して集計する。過去の`Phase 12`記録は旧開発サイクルの履歴であり、現在の18 Phaseの進捗に読み替えない。

**今回の判定はGitHub上の文書内容を確認した静的監査**であり、CI成功、全参照の実在、数学的正当性を確認したものではない。
