# Installation

## Requirements

robotorchan は Python 3.11 以上を対象とし、BoTorch / GPyTorch / PyTorch と組み合わせて利用します。

## Current installation

現在の利用対象はリポジトリの開発版です。リポジトリを clone して editable install します。

```bash
git clone https://github.com/tanakakao/robotorchan.git
cd robotorchan
pip install -e .
```

Fully Bayesian SAAS を利用する場合は追加依存を含めます。

```bash
pip install -e ".[fully-bayesian]"
```

PyPI 公開後の通常インストール手順は、実際の公開を確認した時点でこのページへ追加します。

## Optional dependencies

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
