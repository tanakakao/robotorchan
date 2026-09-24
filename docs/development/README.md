# Development documentation

このディレクトリでは、robotorchan の現在も有効な開発契約を管理します。

## Current contracts

現在仕様の source of truth として扱う恒久文書は次のとおりです。

- `acquisition-architecture.md`: acquisition の責務と custom 実装境界
- `architecture.md`: BoTorch extension としての package / API 設計
- `model_design_guidelines.md`: model 追加・変更時の設計原則
- `kronecker_research_gates.md`: 未実装 Kronecker 拡張の統計・実装ゲート
- `classification_ordinal_research_gates.md`: 分類・ordinal model / AL の統計・実装ゲート
- `input_perturbation_contract.md`: decision-time input perturbation の認定契約
- `external_non_gp_surrogates.md`: 未実装 external non-GP adapter の統合契約
- `training_api.md`: `make_mll()` と training API の契約
- `releasing.md`: version、GitHub Release、PyPI 公開手順

## Temporary design records

audit、prototype、feasibility、gap、個別 model design の文書は恒久仕様ではありません。
現在も必要な判断は、対応する model guide、theory、または上記の恒久文書へ統合します。
統合後の一時文書は削除し、実装経緯は Git 履歴と Pull Request に残します。

次の情報は恒久文書として新規追加しません。

- Phase ごとの進行記録や closeout
- 特定時点の実装件数や gap のスナップショット
- 解消済み feasibility audit
- 実装前 prototype の結果だけを記録した文書
- 現行コードと重複する public seam / runtime contract の確認記録

## Documentation ownership

公開モデルの利用方法は `docs/models/`、理論的背景は `docs/theory/`、
最適化手順は `docs/optimization/` を正とします。

public model の documentation coverage は
`docs/model_coverage.json` と対応する coverage test を正とします。

利用者に必要な現行仕様を development 文書だけに置かないことを原則とします。
このディレクトリには、開発者が継続的に守る必要がある契約だけを残します。

## Change policy

API 変更時に旧 alias、deprecated wrapper、互換関数、互換 Markdown を残しません。
実装、呼び出し側、テスト、Notebook、ドキュメントを同じ現行仕様へ完全移行します。

過去の実装経緯が必要な場合は Git 履歴と Pull Request を参照し、
現在仕様の文書へ履歴を混在させません。
