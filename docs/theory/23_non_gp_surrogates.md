# ベイズ最適化の non-GP surrogate

## 1. 予測表現と統計的な意味

BO に必要なのは GP という名称ではなく、獲得効用に必要な予測分布またはサンプリング可能な予測表現です。経験 ensemble の予測を次で表します。


$$
\{f_1(X),\ldots,f_S(X)\}
$$

S はメンバー数、q は候補数、m は出力数です。経験分布から Monte Carlo sample を引けますが、平均・分散の統計的解釈は ensemble の作り方に依存します。自動的に Bayesian 事後平均・事後分散になるわけではありません。

## 2. Random Forest と Extra Trees

各学習済み tree の予測を一つのメンバーとします。Random Forest はデータ再抽出と特徴選択、Extra Trees はさらに強い分割ランダム化で多様性を作ります。tree 間の不一致は epistemic 不確実性の近似指標になり得ますが、観測ノイズの同定や較正済みの信用区間を保証しません。

## 3. Boosting stage を sample とみなせない理由

Gradient boosting は加算形を持ちます。


$$
F_M(x)=F_0(x)+\sum_{j=1}^{M}\eta h_j(x).
$$

各 h_j は残差構造を補正するよう逐次学習され、予測分布からの交換可能な抽出ではありません。stage 間の広がりを事後不確実性とみなすと統計的な意味を取り違えます。

代わりに独立な bootstrap データ D_s ごとに完成した boosting モデルを学習し、次を経験予測 sample として扱います。


$$
\{F^{(1)}_M(X),\ldots,F^{(S)}_M(X)\}
$$

histogram-based boosting でも、stage ではなく完成したモデルをメンバーとする考え方は同じです。

## 4. 多出力と候補間の依存

m > 1 では各 bootstrap メンバーが予測ベクトルを返します。同じメンバー内で再抽出 index を共有すれば出力の対応を保てますが、出力別回帰器を独立に学習した場合、完全な Bayesian 出力間共分散モデルとは解釈しません。

同じメンバーを候補 batch 全体に適用すると予測パターンの依存を保持できます。ただし GP の条件付き共分散と同じ生成機構ではありません。Gaussian の周辺平均・分散だけから候補間の joint Gaussian 相関を仮定することもできません。

## 5. 獲得関数と探索

経験分布には必要な条件を満たす sample-based 効用を使います。sample 可能というだけで、任意の MC 効用、joint Gaussian、条件付き再推論へ対応するとは限りません。解析的 Gaussian 獲得関数は一般には正当化されません。

tree 予測は区分的で、候補入力への微分が存在しない・有用でない場合があります。勾配に依存しない候補探索を組み合わせます。整数・カテゴリを含む候補の幾何と、予測モデルは別の責務です。

## 6. 出力制約と意思決定

出力 g(x) が制約を表す場合、たとえば g(x) <= 0 を実行可能とする効用を定義できます。surrogate は出力をモデル化し、意思決定側が実行可能性を扱います。既知の入力制約・bounds と混同せず、同じ予測モデルを異なる意思決定規則へ利用できます。

## 7. NGBoost の分布的予測

NGBoost は boosting stage を posterior メンバーとせず、条件付き予測分布を学習します。Gaussian の予測則を使う場合、次の形です。


```text
Y | X=x
 ~ Normal(
     mu(x),
     sigma(x)^2
   )
```


この分散は学習した条件付き予測則の scale であり、GP の潜在関数の事後分散ではありません。単一予測分布だけから epistemic・aleatoric を一意に分解できず、分散だけを根拠に BALD のような分解を要求する基準へ適用できません。

モーメントに基づく回帰 Active Learning や MC BO は、それぞれの仮定を満たせば利用できます。ただし Gaussian 周辺予測があるだけで、解析的 GP 効用や候補間共分散に対応するとは推論しません。

## 8. 較正と評価

forest の tree disagreement、bootstrap boosting の完成モデル間の不一致、分布的 boosting の予測 scale は異なる量です。不確実性スコアの式だけでなく、その入力の生成機構を確認してください。

予測 RMSE、区間構成規則を定めた後の coverage、BO simple regret、計算費用を対象問題で評価します。決定的な予測 benchmark だけで、生の ensemble spread が較正済みとは主張できません。分散・区間幅・sample 不一致を使う Active Learning も、モデルごとの統計的解釈と検証が必要です。

## 9. GP family との境界

| Family | 予測表現 | 推論・学習 | 入力勾配 |
| --- | --- | --- | --- |
| Exact GP | Gaussian posterior | 周辺尤度 | 通常あり |
| Fully Bayesian GP | hyperparameter を積分した GP posterior | MCMC | 条件付き経路ではあり |
| DeepGP | 確率的変分 posterior | 変分推論 | 再パラメータ化できる経路であり |
| Tree ensemble | 経験メンバー | 推定器の学習 | 通常なし |
| Bootstrap boosting | 経験的な完成モデル | 再抽出と反復学習 | 通常なし |
| NGBoost | 条件付き分布 | 分布的 boosting | tree 予測では通常なし |

無限幅 BNN の共分散や Spectral Mixture を使う GP は、特殊な kernel でも GP です。用途上の「高次元」「表現力」ではなく、予測モデルと推論の意味で分類します。

## 参考文献

1. Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5-32.
2. Geurts, P., Ernst, D., and Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63, 3-42.
3. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189-1232.
4. Duan, T., Avati, A., Ding, D. Y., Basu, S., Ng, A. Y., and Schuler, A. (2020). NGBoost: Natural gradient boosting for probabilistic prediction. *International Conference on Machine Learning*.
5. Wilson, J., Hutter, F., and Deisenroth, M. (2018). Maximizing acquisition functions for Bayesian optimization. *Advances in Neural Information Processing Systems*.

## 関連章

- [獲得関数](04_acquisition_function.md)
- [高次元モデル](21_advanced_high_dimensional_models.md)
- [表現力の高い GP](22_expressive_gp.md)
- [高次元探索](20_high_dimensional_search.md)
