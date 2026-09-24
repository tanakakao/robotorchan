# Kronecker extension research gates

この文書は、aligned block-design `train_Y[..., n, m]` を持つ Kronecker MultiTask GP に
追加の observation / prior family を組み合わせる際の、現在有効な研究・実装ゲートをまとめます。
完了済み Phase の記録ではなく、public model を追加する前に満たすべき契約だけを保持します。

## 共通契約

Kronecker variant は次を満たす場合だけ追加します。

1. task identity は `train_Y` の output axis に保持し、long-format へ変換しない。
2. data covariance と task covariance と observation model を分離する。
3. `raw_train_X` / `raw_train_Y` を caller coordinates のまま保持する。
4. posterior は output task axis `... x q x m` を保持する。
5. inference / training objective を `make_mll()` または `training_loss()` で正直に表現する。
6. public promotion 前に posterior sampling と代表的 MC acquisition を runtime 検証する。

Mixed variant は continuous Kronecker variant が成立した後に検討し、`cat_dims` は
`train_X` の data feature のみを指します。

## Current gate matrix

| Family | Status | Main gate |
| --- | --- | --- |
| Replicate noise | prototype | task-specific fixed `G x m` observation variance integration |
| Robust Relevance Pursuit | research-gated | block-aware sparse observation-task support and fitting lifecycle |
| Fully Bayesian SAAS | hold | stable block-design Pyro / MCMC sample-loading contract |
| Student-t | prototype | shared block-design variational Kronecker latent primitive |
| Contaminated Gaussian | prototype | same variational primitive plus mixture observation head |

実装済みの Heteroskedastic / Nonstationary Kronecker はこの research-gate 文書の対象外です。
現行仕様は source、tests、model guide、theory を正とします。

## Replicate noise

raw replicates `train_X[r,d]`, `train_Y[r,m]` を同一 raw row ごとに集約し、
group mean と variance-of-the-mean を task ごとに保持します。

`mean_variance[g,t] = replicate_variance[g,t] / R_g`

必要な observation operator は
`diag(vec(mean_variance))` です。task 間で variance を平均したり、
`G x 1` を暗黙 broadcast してはいけません。

Public promotion の主要ゲートは、Kronecker response model が明示的な task-specific
fixed-noise matrix を posterior semantics を壊さず取り込めることです。

## Robust Relevance Pursuit

RRP の sparse support は既定で observation-task cell `(i,t)` 単位とします。
row-level support は明示的 option としてのみ許容します。

必要なのは、support selection / pruning、Kronecker ordering への写像、latent GP fitting、
sparse correction fitting を一貫して扱う block-aware observation component です。
single-task `noise_covar` を alias したり、`n x m` を long-format に flatten して
互換性を作ってはいけません。

通常の exact MLL だけでは fitting lifecycle を表せない場合、`make_mll()` が
通常 fitting を装ってはいけません。

## Fully Bayesian SAAS

統計モデルは有効ですが、SAAS shrinkage は data dimensions のみに作用し、
task covariance は独立した output factor として保持する必要があります。

現時点では public implementation を Hold とします。再検討条件は次のいずれかです。

- BoTorch が block-design fully Bayesian Kronecker model を提供する。
- stable public Pyro / sample-loading interface だけで薄い extension を実装できる。
- robotorchan が custom fully Bayesian inference を所有する方針を明示的に採用する。

MCMC sample dimension と output task dimension を混同してはいけません。
fantasize / KG / lookahead は専用 runtime evidence が得られるまで advertise しません。

## Shared variational Kronecker gate

Student-t と Contaminated Gaussian は exact Gaussian Kronecker posterior を利用できません。
両者のために別々の latent stack を作らず、次を満たす共通 variational primitive を
先に設計します。

- data-axis inducing points
- explicit task covariance
- output task axis を保持する variational distribution
- intended Kronecker latent prior に対する KL
- observation-task likelihood expectation
- BoTorch-compatible multi-output posterior

inducing inputs に task index を追加して long-format 化する実装は、この契約では
Kronecker support とみなしません。

### Student-t observation head

`y[i,t] | f[i,t] ~ StudentT(df, loc=f[i,t], scale=sigma_t)`

初期 contract では shared `df > 2` と shared または明示的 task-specific scale を使います。
training は ELBO-style `training_loss()` を基本とし、exact MLL を advertise しません。

### Contaminated Gaussian observation head

初期 contract は shared mixture probability `p` と task-specific
`sigma_in[t]`, `sigma_out[t]` を使います。

`0 < p < 1`、`0 < sigma_in[t] < sigma_out[t]` を満たし、
task-specific `p[t]` は最初の実装では導入しません。

contamination probability diagnostic は latent GP posterior / acquisition output とは
別の診断 surface として扱います。

## Public promotion checklist

research-gated family を public model に昇格する前に、少なくとも次を検証します。

- block-design X/Y を flatten しない
- family 固有 parameter shape / validation
- finite training objective と gradient
- finite posterior mean / variance / samples
- posterior sample shape が `... x q x m`
- nontrivial task covariance が prediction に反映される
- representative scalarized MC acquisition
- compatible optimizer path
- actual inference contract と training API / capability metadata の一致

モデル名や隣接 family の成功だけから acquisition compatibility を推定しません。

## Revisit policy

この文書の status は upstream BoTorch API、robotorchan の共通 latent / observation
infrastructure、または runtime prototype のいずれかが変化した時に更新します。
実装が public contract として成立した family は、この文書から削除し、
source / tests / model guide / theory を source of truth とします。
