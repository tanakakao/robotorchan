# Installation

## Requirements

robotorchan は Python 3.11 以上を対象とし、BoTorch / GPyTorch / PyTorch と組み合わせて利用します。

## PyPI

通常の利用では次を使用します。

```bash
pip install robotorchan
```

Fully Bayesian SAAS を利用する場合は追加依存を含めます。

```bash
pip install "robotorchan[fully-bayesian]"
```

## Development install

リポジトリを clone して editable install する場合:

```bash
pip install -e .
```

Notebook も実行する場合:

```bash
pip install -e ".[examples]"
```

Fully Bayesian SAAS の Notebook も実行する場合:

```bash
pip install -e ".[examples,fully-bayesian]"
```

## Next step

インストール後は [Quickstart](quickstart.md) で共通モデル API を確認し、[Model selection](model_selection.md) から問題設定に合うモデルファミリーを選んでください。
