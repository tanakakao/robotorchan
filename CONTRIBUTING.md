# Contributing to robotorchan

robotorchanへの貢献を歓迎します。

## 開発環境

Python 3.11以上を使用してください。

```bash
git clone https://github.com/tanakakao/robotorchan.git
cd robotorchan
pip install -e ".[dev,examples]"
```

Fully Bayesianモデルを変更する場合は、必要に応じて次も導入してください。

```bash
pip install -e ".[dev,examples,fully-bayesian]"
```

## 変更方針

- BoTorch-nativeな挙動とacquisition function互換性を維持してください。
- 共通wrapperではraw training dataと共通インターフェースの一貫性を維持してください。
- API変更時に旧名alias、deprecated wrapper、互換関数を追加して残さないでください。
  呼び出し側、テスト、example、Notebook、ドキュメントを新仕様へ完全に移行してください。
- 公開モデルを追加・変更した場合は、モデルガイド、理論ドキュメント、Notebookの対応範囲も確認してください。
- 新しいCI jobは、既存CIでは検証できない明確な理由がある場合だけ追加してください。
- GitHub APIや自動編集で複数行コードを書き換えた場合は、更新後の実ファイルを再取得し、
  エスケープ文字（特にリテラルの \\n）がコードへ混入していないことを確認してください。
- Pythonコードを変更した場合は、PR作成・更新前に `ruff check .` と
  `ruff format --check .` の両方を確認してください。行長だけでなくRuff formatterの
  自動整形結果を正とし、手動整形との差分を残さないでください。
- 同種の失敗や修正が複数回発生した場合は、その場の修正だけで終わらせず、
  再発防止策を本ファイルの開発ルールへ追記してください。頻発する失敗をルール化すること
  自体を必須の開発手順とします。

## 品質確認

通常の変更では、少なくとも対象範囲のテストを実行してください。

```bash
ruff check .
ruff format --check .
pytest \
  --ignore=tests/models/test_fully_bayesian_single_task_gp.py \
  --ignore=tests/models/test_fully_bayesian_multi_task_gp.py \
  --ignore=tests/models/test_mixed_fully_bayesian.py
```

Fully Bayesian関連は専用依存関係を導入した環境で対象テストを実行してください。
Notebookを変更した場合は、リポジトリのNotebook runnerも確認してください。

## Pull Request

Pull Requestには次を簡潔に記載してください。

- 変更内容と目的
- 公開APIへの影響
- 実行したテスト
- ドキュメントやNotebookへの影響

変更は可能な限り1つの目的に絞ってください。
