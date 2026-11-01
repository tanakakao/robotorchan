# Installation

## Requirements

robotorchan は Python 3.11 以上を対象とし、BoTorch / GPyTorch / PyTorch と組み合わせて利用します。
BoTorch の対応範囲は `>=0.18.1,<0.19` です。GPU は必須ではなく、CIではCPU版のPyTorchを使用します。

## 開発環境の準備

ソースから開発する場合は、既存のチェックアウトで仮想環境を作成します。
リポジトリをまだ取得していなければ、先に
`git clone https://github.com/tanakakao/robotorchan.git` を実行してください。

```bash
cd robotorchan
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"
python -m pip install -e ".[dev,examples]"
python -m pip check
```

上記の有効化コマンドはLinux / macOS用です。Windows PowerShellでは
`.venv\Scripts\Activate.ps1` を使います。
CUDA版を利用する場合は、PyTorch公式の環境に合うインストール手順で
PyTorchを導入してから、robotorchanをインストールしてください。

既にチェックアウトのルートにいる場合は `cd robotorchan` を省略します。
以降の `python -m pip` は、実行に使うPython環境へ依存関係を導入します。

## PyPI

公開済みのリリースを利用する場合は次を使用します。
リリース版と `main` のAPIが異なる場合は、利用するバージョンに対応した文書を参照してください。

```bash
python -m pip install robotorchan
```

Fully Bayesian SAAS を利用する場合は追加依存を含めます。

```bash
python -m pip install "robotorchan[fully-bayesian]"
```

## Development install

リポジトリを clone して editable install する場合:

```bash
python -m pip install -e .
```

Notebook も実行する場合:

```bash
python -m pip install -e ".[examples]"
```

Fully Bayesian SAAS の Notebook も実行する場合:

```bash
python -m pip install -e ".[examples,fully-bayesian]"
```

追加依存の用途は次のとおりです。必要な機能に合わせて指定してください。

| extra | 用途 |
| --- | --- |
| `dev` | pytest、coverage、Ruffによる開発・検証 |
| `examples` | JupyterLab、描画、Notebookの実行 |
| `fully-bayesian` | Fully Bayesian SAASのNUTS推論 |
| `ngboost` | NGBoost surrogate |
| `cmaes` | CMA-ES候補最適化backend |

例えばNGBoostとCMA-ESも開発する場合は
`python -m pip install -e ".[dev,examples,ngboost,cmaes]"` とします。
extraの導入だけでは、その機能の学習・探索や検証を実行したことにはなりません。

## 動作確認

仮想環境を有効にしたまま、リポジトリのルートで実行します。

```bash
python -m pip check
python -m pytest tests/models/standard/test_single_task_gp.py -q
MPLBACKEND=Agg python examples/run_notebooks.py --only 01_single_task_gp.ipynb --timeout 300
```

最後のコマンドには `examples` extraが必要です。Notebookはメモリ上で実行され、
保存済みのセル出力を上書きしません。Jupyterで手動実行する場合も、
依存関係を導入したPythonをカーネルに選んでください。
開発時の全体チェックは [貢献手順](../../CONTRIBUTING.md) を参照してください。

## Next step

インストール後は [Quickstart](quickstart.md) で共通モデル API を確認し、[Model selection](model_selection.md) から問題設定に合うモデルファミリーを選んでください。
