# Documentation closeout

この文書は、モデル実装に対する user-facing documentation の完成条件を固定するための
closeout record です。新しい public model を追加した場合は、この状態を維持します。

## 完成条件

public model は `robotorchan.models.__all__` を source of truth とし、次を満たします。

1. `docs/model_coverage.json` に登録されている。
2. モデル選択の入口として `docs/models.md` がある。
3. 統計的仮定を説明する `docs/theory/` の章へ到達できる。
4. 同じ理論・学習契約を共有する representative Notebook へ到達できる。
5. manifest が参照する guide / theory / Notebook は実在する。
6. non-model public export は暗黙に無視せず、明示的に除外する。

この契約は `tests/test_model_documentation_coverage.py` が検証します。

## 現在の状態

- public model: 102
- 明示的な non-model export: `UnsupportedModelOperationError`
- theory: 01–21
- user-facing Notebook: 01–23
- 軽量 Notebook 15–22 は既存の Notebook CI に統合済み
- 08 / 11 は計算負荷のため通常 Notebook CI から分離
- Phase 9 完了時点の CI は成功

## 設計上の境界

### 1 model = 1 Notebook にはしない

Mixed、MultiTask、Kronecker、reducer variants などは、理論と学習契約を共有する場合、
同じ theory chapter / representative Notebook を参照できます。class 数を増やすことと
Notebook 数を増やすことを機械的に結び付けません。

### API の互換層を作らない

ドキュメントや Notebook が古い API を参照した場合、互換 alias、deprecated wrapper、
互換関数を追加して解決しません。呼び出し側、テスト、Notebook、ドキュメントを現行 API
へ完全に移行します。

### audit 文書を入口にしない

coverage audit や phase closeout は開発記録です。利用者の入口は README、
`docs/models.md`、`docs/theory/`、model-specific docs、`examples/` とします。

## 今後モデルを追加するとき

public export を追加する変更では、同じ PR で少なくとも次を確認します。

- `docs/model_coverage.json` の coverage entry
- 既存 theory chapter で説明できるか、新章が必要か
- 既存 representative Notebook で十分か、新規 Notebook が必要か
- `docs/models.md` や専門ガイドのモデル選択説明を更新すべきか
- 新しい統計的仮定や学習 API がある場合、その契約をテストすべきか

単なる named wrapper の追加だけを理由に theory chapter や Notebook を複製しません。
