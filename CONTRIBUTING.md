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

## 品質確認

通常の変更では、少なくとも対象範囲のテストを実行してください。

```bash
ruff check .
ruff format --check .
pytest \
  --ignore=tests/test_saas_fully_bayesian.py \
  --ignore=tests/test_saas_fully_bayesian_multitask.py \
  --ignore=tests/test_saas_fully_bayesian_mixed.py
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
