# High-dimensional benchmarks

高次元 Bayesian optimization の比較条件、評価予算、集計契約をまとめます。

| Benchmark | 主な比較対象 | q | 解釈 |
| --- | --- | ---: | --- |
| [Sequential BO](sequential_bo.md) | acquisition search strategy | 1 | 同一 surrogate 下の sequential search |
| [Batch BO](batch_bo.md) | BO policy | > 1 | joint batch candidate policy |

## 共通原則

両 benchmark はモデルや探索法の「総合ランキング」を作るためのものではありません。比較対象以外の条件を可能な限り固定し、どの設計差が結果へ寄与したかを解釈できることを優先します。

固定対象には objective、初期設計規則、seed grid、入力次元、観測予算、surrogate の基本契約を含みます。strategy 固有の state や embedding は、その手法のアルゴリズムに従って trajectory 間で管理します。

実行コードはリポジトリ直下の `benchmarks/`、回帰テストは `tests/benchmarks/` にあります。モデル選択は [Model guides](../../models/README.md)、探索戦略の仕様は [Optimization guides](../../optimization/README.md) を参照してください。
