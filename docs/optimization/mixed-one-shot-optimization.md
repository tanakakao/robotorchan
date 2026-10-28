# Mixed one-shot 獲得関数の最適化

## 適用範囲

必要な fantasy 操作をモデルが提供する場合に、mixed one-shot の評価と候補最適化を行います。実候補と fantasy 意思決定点でカテゴリが異なる mixed qKG・mixed 多忠実度 qMFKG を含みます。

## 通常の `optimize_acqf_mixed` では不十分な理由

one-shot KG は実験候補と内部の fantasy 意思決定点を含む拡張 batch を最適化します。通常の `optimize_acqf_mixed` は `fixed_features_list` の各割当を batch 全体へ適用するため、実候補と全 fantasy 点のカテゴリが同じになります。

fantasy posterior の最適カテゴリは観測候補と異なり得るため、この制限は一般には正しくありません。通常の mixed optimizer を qKG・qMFKG の汎用解法として案内しません。

## 現在の対応

- mixed qMFKG モデルの構築、目標 fidelity 上の incumbent 最適化。
- コスト効用、目標 fidelity 射影、one-shot 評価。
- fantasy 意思決定点間で異なるカテゴリ。
- `optimize_mixed_one_shot_acqf` による q=1 の行別カテゴリ厳密列挙。

q > 1 と厳密列挙の上限を超える割当空間は未対応です。実行検証には、カテゴリをまたぐ明示的拡張 batch と、実際の mixed qKG・qMFKG による候補生成を含みます。

## 初期化

各行の厳密なカテゴリ割当ごとに拡張 batch 全体を平坦化し、条件付き連続最適化問題として BoTorch の標準初期化を使います。カテゴリを拡張行ごとに固定することで、実候補と fantasy 行で異なるカテゴリを保ちます。

連続 qKG・qMFKG が専用 one-shot 初期化を使う経路とは異なります。この helper は q=1 と自動初期化のみが対象で、公開 API として明示的な拡張 `batch_initial_conditions` を受け付けるとは主張しません。

## 最適化の契約

カテゴリ割当は拡張行ごとに変化でき、全行共通の固定値とはしません。最適化後の実候補抽出は BoTorch one-shot の意味を保ちます。テストには実候補と fantasy 最適点のカテゴリが異なる場合を含めます。

q=1 と有限で上限のある割当空間を対象とし、範囲外では通常の mixed optimizer に fallback せず失敗します。

## 処理手順

実 batch サイズ q、拡張サイズ q_aug に対して、拡張行ごとのカテゴリ割当行列を用います。

1. 実候補行の合法なカテゴリを列挙する。
2. fantasy 行のカテゴリを独立に割り当てる。
3. 各割当の下で連続座標を最適化する。
4. 条件付き最適値間で獲得値を比較する。
5. one-shot の抽出規則で実候補だけを返す。

現行の厳密経路は q=1 の有限カテゴリが対象です。枝刈り・近似探索を行うとは主張しません。多カテゴリ・大きな q の高速化は将来の性能拡張であり、正しさの要件とは分けます。

### 不変条件

- fantasy 行は実候補と異なるカテゴリを選べる。
- 条件付き連続最適化中のカテゴリ座標は固定する。
- fidelity は最適化変数として残し、目標への射影は獲得関数が担う。
- 獲得関数には拡張 batch 全体を渡す。
- 返り値から fantasy 行を除外する。
- 報告獲得値は最適化した拡張 batch に対応する。
- 通常の non-one-shot mixed は BoTorch 標準 mixed optimizer を使う。

列挙前にカテゴリ列、非空の値集合、bounds 内の値を検証します。不正な固定値を bounds 外へ逃がしません。大きすぎる割当空間は明示的に拒否します。

## 経路の選択と列挙上限

通常の連続獲得関数は `optimize_acqf`、通常の Mixed は `optimize_acqf_mixed`、mixed one-shot qKG・qMFKG は専用 helper を使います。

平坦化は条件付き連続最適化に限り、カテゴリ固定列は拡張行ごとに構成します。抽出は獲得関数の `extract_candidates` に委譲します。qMFKG の実候補 fidelity は bounds 内で最適化し、全拡張行を目標 fidelity に固定しません。

列挙規模は `row_assignment_count ** augmented_q` です。q=1 と `max_assignments` で組合せ経路を明示的に制限します。大きな batch や scalable なカテゴリ探索へ暗黙に切り替えません。
