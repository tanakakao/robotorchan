# アーキテクチャの読み方

robotorchan は BoTorch のモデル・予測分布・獲得関数を基盤として拡張するライブラリです。標準機能は上流へ委譲し、データの保持、追加モデル、候補探索などの責務を必要な範囲で追加します。

モデルは観測から予測分布を構成し、獲得関数は予測から候補の効用を計算し、最適化器は効用の高い候補を探索します。モデルの次元削減と、候補探索空間の次元削減は別々の処理です。

1. [GP の実装ガイド](../guide/02_gaussian_process.md)でモデルと学習目的の対応を読む。
2. [獲得関数の実装ガイド](../guide/04_acquisition_function.md)で予測・目的・制約の合成を理解する。
3. [候補最適化ガイド](../guide/optimization/01_foundations.md)で初期化と探索の責務を確認する。
4. 変更対象の[モデルガイド](../models/README.md)とテストを確認する。

モジュール境界と拡張時の規則の正本は[Architecture contract](../../docs-internal/architecture/architecture.md)です。[Acquisition architecture](../../docs-internal/architecture/acquisition-architecture.md)と[Optimizer architecture](../../docs-internal/architecture/optimization/optimizer-architecture.md)に各層の詳細を置きます。同じ API 仕様を本ページへ重複掲載しません。
