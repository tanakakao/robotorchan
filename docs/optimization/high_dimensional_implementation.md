# High-dimensional optimization: implementation notes

This guide preserves robotorchan-specific search strategy contracts separated from [high-dimensional optimization theory](../theory/optimization/09_high_dimensional.md).

## 9.13 BAxUSState

BAxUS の状態は ambient dimension と残り evaluation budget から

- initial target dimension
- expansion schedule
- split budget
- failure tolerance

を導出します。

target dimension は利用者が任意に固定する旧方式ではなく、
evaluation budget を含む state transition の一部です。

success / failure に応じて trust-region length を更新し、
restart_triggered になった後に expand_subspace() で embedding を拡張します。

## 9.19 Candidate constraints と reduced search space

original-space constraint g(X) を latent coordinate z へ移すには、

    g(project(z))

という constraint mapping が必要です。

linear constraint であっても embedding 後の係数や q-batch semantics を
正しく変換しなければなりません。

この写像を省略すると元の制約付き最適化問題とは異なる問題を解くことになります。

そのため original-space CandidateConstraints は

- LatentSpaceStrategy
- REMBOStrategy
- HeSBOStrategy
- ALEBOStrategy
- BAxUSStrategy

では unmapped constraint として明示的に拒否します。

これは未検証の座標変換を silently 適用しないための correctness boundary です。

## 9.22 Surrogate と Search Strategy の分離

高次元探索では surrogate と search geometry の変更を区別します。

REMBO / HeSBO / BAxUS の通常 strategy は original-space surrogate を共有できます。

これにより benchmark では同じ surrogate / acquisition に対して
search strategy だけを変更し、探索性能差を比較できます。

ALEBO は論文上の metric-aware surrogate geometry が手法の一部なので、
ALEBOGP と strategy の対応を明示的に扱います。

## 9.23 Stateful と Stateless

高次元 search strategy は状態の有無でも分けられます。

| Strategy | State |
| --- | --- |
| OriginalSpace | stateless |
| RandomSearch | stateless |
| LatentSpace | fitted reducer を利用 |
| REMBO | fixed embedding |
| HeSBO | fixed embedding |
| ALEBO | fixed embedding / polytope |
| BAxUS | adaptive state |
| BAxUS Thompson Sampling | adaptive state |

BAxUS では objective evaluation 後に values と target_candidates を
update_state() へ明示的に返す必要があります。

search strategy 自体は objective function を評価しません。

## 9.25 Strategy 選択の考え方

| 状況 | 主な候補 |
| --- | --- |
| d がまだ直接探索可能 | OriginalSpace |
| 単純で堅牢な基準が必要 | RandomSearch |
| fitted low-dimensional representation がある | LatentSpace |
| low effective dimension を仮定 | REMBO |
| sparse embedding を使いたい | HeSBO |
| clipping distortion を避けたい | ALEBO |
| effective dimension が未知 | BAxUS |
| BAxUS + posterior sampling | BAxUS Thompson Sampling |

高次元だから常に embedding が優れるわけではありません。

effective dimension assumption、projection distortion、evaluation budget、
acquisition optimization cost を合わせて選択する必要があります。
