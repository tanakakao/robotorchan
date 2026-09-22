# Robust / Noise MultiTask 設計契約

Robust / Noise 系モデルを MultiTask へ拡張するときの共通方針を定義します。
この文書は cross-product class を機械的に増やすための一覧ではなく、
各観測モデルの生成仮定を保ったまま task 構造を追加するための設計契約です。

## 基本方針

robotorchan では次の三つを独立した設計軸として扱います。

1. **data covariance**: continuous / mixed input に対する入力間 covariance
2. **task covariance**: task 間の相関構造
3. **observation model**: outlier、heavy tail、heteroskedasticity などの観測機構

MultiTask 化では observation model を通常の Gaussian noise に戻してはいけません。
Mixed 化では task feature を categorical data feature として扱ってはいけません。

## Long-format MultiTask

`MultiTaskGP` と同様に、long-format では `train_X` の一列を
`task_feature` として使用します。

Robust / Noise MultiTask でも、

- task feature は task covariance が扱う
- data covariance は task feature を除外する
- `cat_dims` は data feature のみを指定する
- `cat_dims` と `task_feature` の重複を禁止する

という契約を維持します。

## Mixed MultiTask

Mixed MultiTask の data covariance は既存 `MixedMultiTaskGP` と同じ
mixed-kernel convention を使用します。continuous + categorical +
continuous/categorical interaction を data covariance とし、task covariance は
独立して保持します。

Robust family ごとに mixed kernel 実装を複製しません。共通の
`make_mixed_covar_module` と feature validation を再利用します。

## Kronecker MultiTask

Kronecker 形式は task identity が `train_Y` の出力列にあります。
long-format と observation model の意味が異なるため、通常 MultiTask 版が存在することを
理由に Kronecker 版を自動的に追加しません。

Phase 5 の具体的な採否と受入条件は
[`robust_kronecker_design.md`](robust_kronecker_design.md) に記録します。

各 robust/noise family について、

- likelihood / observation model が block design と整合するか
- task ごとの noise parameterization が明確か
- posterior contract を自然に維持できるか

を確認した上で個別に採否を決めます。

## Family ごとの拡張判断

| family | MultiTask | Mixed MultiTask | Kronecker | 方針 |
| --- | --- | --- | --- | --- |
| Robust Relevance Pursuit | planned | planned | evaluate | 最初の reference implementation |
| Heteroskedastic | planned | planned | evaluate | response と noise の task 構造を明示 |
| Joint Heteroskedastic | evaluate | evaluate | evaluate | latent response/noise の joint 構造を先に設計 |
| Student-t | planned | planned | evaluate | heavy-tail parameter の共有範囲を明示 |
| Contaminated | planned | planned | evaluate | contamination parameter の共有範囲を明示 |
| Replicate Noise | evaluate | evaluate | evaluate | replicate grouping と task identity の意味を確認 |
| Nonstationary | planned | planned | evaluate | input-dependent smoothness と task covariance を分離 |

`planned` は実装対象、`evaluate` は生成仮定と API を確認してから採否を決めることを
意味します。未実装の cross-product class が存在すると解釈してはいけません。

## Public API contract

新しい Robust MultiTask model も既存 robotorchan model と同じ契約を満たします。

- caller supplied raw training tensors を保持する
- Exact GP では `make_mll()` を提供する
- BoTorch の `posterior()` contract を維持する
- acquisition function へ元入力空間の候補点を渡せる
- dtype / device を保持する
- feature index は負 index を含めて正規化する
- unsupported operation は暗黙に近似せず明示する

API 変更が必要な場合、旧名 alias、deprecated wrapper、互換関数は追加しません。
呼び出し側、tests、examples、notebooks、docs を新仕様へ完全移行します。

## 実装順序

最初に Robust Relevance Pursuit を reference implementation として実装します。
そこで確立した MultiTask / Mixed MultiTask の構造を、統計モデルとして共有可能な範囲だけ
Heteroskedastic、Student-t、Contaminated、Nonstationary へ展開します。

継承や helper の共通化は、少なくとも二つの family で同じ責務が確認できてから行います。
将来の組合せを想定した抽象基底 class を先に作ることは避けます。
