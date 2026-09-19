# Development documentation

このディレクトリは robotorchan の**現在も有効な開発契約**をまとめます。Phase進行記録、closeout、解消済み feasibility audit は恒久ドキュメントとして保存しません。

- [Architecture](architecture.md): BoTorch extension としての責務、wrapper policy、package/API設計
- [Model design guidelines](model_design_guidelines.md): model追加・変更時の設計原則、Mixed/robust semantics、CI preflight
- [Training API](training_api.md): `make_mll()` と joint neural `training_loss()` の契約
- [Releasing](releasing.md): version、GitHub Release、PyPI Trusted Publishing

## 保存基準

開発文書として残すのは、今後の変更でも判断基準になる設計原則・公開契約・運用手順です。特定Phaseの進捗、実装前のgap分析、すでに解消されたfeasibility判断、closeout時点の件数スナップショットは削除します。

過去の実装経緯が必要な場合はGit履歴とPull Requestを参照し、現在仕様の文書へ履歴を混在させません。

## 変更時の原則

API変更時に旧alias、deprecated wrapper、互換関数、互換Markdownを追加して残しません。実装、呼び出し側、テスト、Notebook、ドキュメントを同じ現行仕様へ完全移行します。

public model のドキュメントcoverageは [model_coverage.json](../model_coverage.json) と対応テストを正とします。
