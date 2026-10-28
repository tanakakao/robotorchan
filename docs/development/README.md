# 開発ガイド

ライブラリを拡張・保守するための学習用ガイドです。まず[貢献手順](../../CONTRIBUTING.md)で開発環境と品質確認の流れを確認してください。

- [アーキテクチャの読み方](architecture.md)：モデル・予測・獲得関数・探索の責務を理解する。
- [学習 API](training_api.md)：Exact GP の周辺尤度と Joint neural model の学習目的を区別する。
- [リリース手順](releasing.md)：バージョン・配布物・公開の手順を確認する。
- [理論と実装のガイド](../guide/README.md)：数理から実装とテストへ辿る。

実装が守るべき詳細な条件は[内部契約](../../docs-internal/README.md)、過去時点の監査・検証結果は[プロジェクト記録](../../.github/project-records/README.md)で管理します。内部資料も公開リポジトリの一部であり、アクセス制限を意味しません。
