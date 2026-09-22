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
  `models.__all__` とpublic model contract testも同じ変更で同期してください。
  ドキュメント対応情報は `MODEL_REGISTRY` を正とし、`docs/model_coverage.json` を手編集せず、
  `python scripts/generate_model_coverage.py` で再生成してください。生成物を更新し忘れないよう、
  CIでも再生成結果とcommit済みファイルの一致を検証します。
- 新しいCI jobは、既存CIでは検証できない明確な理由がある場合だけ追加してください。
- GitHub APIや自動編集で複数行コードを書き換えた場合は、更新後の実ファイルを再取得し、
  エスケープ文字（特にリテラルの `\\n`）がコードへ混入していないことを確認してください。
  改行を含む置換では `\\n` を文字列として挿入せず、必ず実改行を使用してください。
- Pythonコードは100文字の行長制限を前提に編集してください。長いtuple、関数呼び出し、条件式、
  パス文字列を追加・変更した場合は、push前に100文字超過がないことを確認してください。
- 内部helperやprivate symbolを移動・削除する場合も、リポジトリ全体の参照を検索し、
  import元・派生モデル・テストを同じ変更で新しい実装へ完全移行してください。
- Pythonコードを変更した場合は、PR作成・更新前に `ruff check .` と
  `ruff format --check .` の両方を確認してください。行長だけでなくRuff formatterの
  自動整形結果を正とし、手動整形との差分を残さないでください。
- 制約付き探索のテストデータは、固定値を手書きして実行時の制約判定へ依存させず、
  対象strategyのfeasible samplerなどで制約を満たす点を生成してください。
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


## Capability metadata

モデルや acquisition の public capability を変更した場合は、実装だけでなく registry metadata と対応する contract test も同じ変更で更新してください。モデル名の文字列解析から capability を推測する実装は追加しません。

モデル registry を変更した場合、生成物である `docs/model_coverage.json` を直接編集せず、既存の生成スクリプトを使用してください。
