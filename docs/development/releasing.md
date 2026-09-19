# リリース手順

`robotorchan` のPyPIリリースでは、`pyproject.toml` の `project.version` をバージョンの基準とします。

## バージョン規約

Semantic Versioningを基本とし、Git tag / GitHub Releaseは `v{version}` 形式にします。

例:

- package version: `0.1.0`
- Git tag: `v0.1.0`
- GitHub Release: `v0.1.0`
- PyPI version: `0.1.0`

公開workflowは、Release tagと `pyproject.toml` のversionが一致しない場合は停止します。

## リリース前チェック

1. `main` が最新で、CIが成功していることを確認します。
2. `pyproject.toml` の `project.version` を次のversionへ更新します。
3. version変更をPull Request経由で `main` にマージします。
4. 必要に応じてローカルでも配布物を確認します。

```bash
python -m build
python -m twine check dist/*
```

5. `main` の対象commitに `v{version}` tagを作成します。
6. 同じtagからGitHub Releaseを公開します。
7. `Publish to PyPI` workflowの成功を確認します。
8. PyPIから通常インストールできることを確認します。

```bash
pip install robotorchan
```

## 公開workflowの安全条件

`.github/workflows/publish.yml` は次を確認してからPyPIへ公開します。

- GitHub Releaseのtagが `v{project.version}` と一致する
- tagのcommitが `main` に含まれる
- sdist / wheelのbuildが成功する
- `twine check` が成功する
- build済みwheelをinstallできる
- installされたpackage versionがRelease versionと一致する

PyPIへの認証にはTrusted Publishing (OIDC) を使用します。PyPI API tokenをGitHub Secretsに保存する運用は想定していません。

## PyPI Trusted Publishing の事前設定

初回リリース前にPyPI側でTrusted Publisherを設定します。

- owner: `tanakakao`
- repository: `robotorchan`
- workflow: `publish.yml`
- environment: `pypi`

GitHub側でも `pypi` environment を作成し、必要に応じてrequired reviewerを設定してください。公開jobだけがこのenvironmentとOIDC権限を使用します。

## LICENSE

robotorchanはBSD-3-Clauseで公開します。配布物にはリポジトリ直下の `LICENSE` を含め、package metadataとの一致をCIで検証します。
