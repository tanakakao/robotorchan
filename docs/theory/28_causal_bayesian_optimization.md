# 28. 因果ベイズ最適化（Causal Bayesian Optimization）

本章では、普通のベイズ最適化が「ある入力を与えたときの目的値」を探すのに対し、**因果ベイズ最適化（CBO）は何を操作し、どの値に設定するか**を探索する考え方を学びます。数式・仮定・比較の注意を扱います。実装との対応は[実装ガイド](../guide/28_causal_bayesian_optimization.md)、実際の候補提案は[Notebook 58](../../examples/notebooks/58_causal_bayesian_optimization.ipynb)を参照してください。

## 1. 予測と介入は違う

製造工程で温度 \(X\) と歩留まり \(Y\) が相関していても、「温度を意図的に変えると歩留まりが変わる」とは限りません。たとえば設備状態 \(U\) が温度と歩留まりの両方へ影響すると、温度が高いグループの歩留まりだけを観察しても、温度変更の効果は判定できません。

- **観測・条件付け**：\(P(Y\mid X=x)\) は、データの中で \(X=x\) だった場合の \(Y\) の分布です。
- **介入**：\(P(Y\mid \operatorname{do}(X=x))\) は、外部から \(X\) を \(x\) に設定し、ほかの因果機構は維持した場合の分布です。

一般に

\[
P(Y\mid X=x)\ne P(Y\mid\operatorname{do}(X=x)).
\]

違いを数式で確認しましょう。独立な平均0の雑音 \(U,\varepsilon_X,\varepsilon_Y\) に対して

\[
X=U+\varepsilon_X,\qquad Y=X+U+\varepsilon_Y
\]

とします。\(U\) と \(\varepsilon_X\) が正規分布なら、

\[
\mathbb E[Y\mid X=x]
=x+\frac{\operatorname{Var}(U)}
{\operatorname{Var}(U)+\operatorname{Var}(\varepsilon_X)}x,
\]

ですが、温度を強制設定すると、\(X\) を決める元の式だけが置き換わるため

\[
\mathbb E[Y\mid\operatorname{do}(X=x)]=x.
\]

したがって、単なる回帰モデルで条件付き平均を学習しても、介入応答を正しく予測できるとは限りません。**この区別がCBOの出発点です。**

## 2. 既知のDAGと構造的因果モデル

DAG（Directed Acyclic Graph）は、変数を矢印で結んだ**有向非巡回グラフ**です。矢印は因果方向の仮定を表します。一般の構造的因果モデル（SCM）は

\[
V_j=f_j(\operatorname{Pa}(V_j),U_j)
\]

という構造方程式からなり、\(\operatorname{Pa}(V_j)\) は直接の親、\(U_j\) は外生要因を表します。DAGが正しく、外生要因や方程式について妥当な仮定を持つことが重要です。DAGの「既知」は、データから自動的に因果方向を見つけたことを意味しません。

**Hard intervention** \(\operatorname{do}(X=x)\) は、\(X\) の構造方程式を \(X:=x\) に置き換えます。\(X\) に入る矢印を切り、\(X\) から出る矢印と、ほかのノードの式は維持します。同じ値 \(X=x\) を観測した場合と、生成機構が異なることが重要です。

## 3. 介入集合を選ぶ最適化

操作する変数の集合を \(S\subseteq V\setminus\{Y\}\)、その操作値を \(x_S\in\mathcal X_S\) とします。単一の目的出力 \(Y\) を最大化する問題を

\[
(S^\star,x_S^\star)\in
\operatorname*{arg\,max}_{S\in\mathcal S,\ x_S\in\mathcal X_S}
f_S(x_S),\qquad
f_S(x_S):=\mathbb E[Y\mid\operatorname{do}(S=x_S)]
\]

と表します。ここで \(\mathcal S\) は許可された**介入集合の候補**です。たとえば \(S=\{X\}\) では \(Z\) の構造方程式は自然に動き、\(S=\{X,Z\}\) では両方の方程式を置き換えます。この2つは一般に同じ関数ではありません。

評価コスト \(c(S)>0\) を考える場合、より安価な介入を優先できるようにします。ただし、**低コスト介入が必ず最適な目的値を達成できるわけではない**ため、改善量とコストを分けて記録する必要があります。

## 4. 介入GP（Intervention GP）

同じ集合 \(S\) を実際に介入した記録だけを集め、

\[
\mathcal D_S=
\{(x_{S,i},y_i):\operatorname{do}(S=x_{S,i})\text{ を実行した}\}
\]

を作ります。集合 \(S\) ごとに \(f_S\) のGP代理モデルを学習すれば、事後予測平均 \(\mu_{S,t}(x_S)\) と標準偏差 \(\sigma_{S,t}(x_S)\) が得られます。

\[
f_S\sim\operatorname{GP}(m_S,k_S),\quad
f_S(x_S)\mid\mathcal D_S \approx
\mathcal N(\mu_{S,t}(x_S),\sigma^2_{S,t}(x_S)).
\]

**観測データや別の介入集合のデータを、同じ \(S\) の介入結果として混ぜてはいけません。** 複数の集合が持つ構造を共有する高度な因果GPとは異なり、集合ごとに独立のGPを持つ構成では集合間の共分散は学習しません。正確な応答が得られるかはデータ数、GPの近似・カーネルの仮定、外挿の難しさにも依存します。

## 5. 介入の獲得関数とコスト

全介入集合の実測目的値から共通の暫定最良値 \(y^+_t=\max_{i\le t}y_i\) を求めます。最大化におけるExpected Improvementは

\[
\operatorname{EI}_{S,t}(x_S)=
\mathbb E[(f_S(x_S)-y^+_t)_+\mid\mathcal D_S],
\qquad (z)_+:=\max(z,0).
\]

GPが正規事後で \(\sigma>0\) の場合、

\[
\operatorname{EI}=(\mu-y^+)\Phi(z)+\sigma\phi(z),
\quad z=\frac{\mu-y^+}{\sigma}.
\]

\(\Phi\) と \(\phi\) は標準正規の累積分布関数と密度です。**改善量は目的値そのものではなく、次に実験する価値を表す期待値**です。最小化では符号・改善方向を反転して考えます。

数値的に安定なLogEIで各集合の候補値を探し、

\[
x_{S,t+1}\in\arg\max_{x_S}\log\operatorname{EI}_{S,t}(x_S),
\quad
S_{t+1}\in\arg\max_{S\in\mathcal S}
\left[\log\operatorname{EI}_{S,t}(x_{S,t+1})-\log c(S)\right]
\]

と比較できます。これは \(\mathrm{EI}/c(S)\) を最大化することと同じ順位です。**LogEIそのものをコストで割るのは誤り**です。ここでは \(c(S)\) を集合ごとの正の定数と仮定するため、集合内の候補位置はコストで変わりません。候補の値にも依存するコストなら、最適化式を変更しなければなりません。

## 6. 逐次的な実験（Ask-and-Tell）

1. 各介入集合に対して初期実験を行い、実際の介入観測を蓄積します。
2. 集合ごとのGPを学習し、事後予測を更新します。
3. 獲得関数を最適化して**介入集合 \(S\)** と**介入値 \(x_S\)** を提案します。
4. 実験者が \(\operatorname{do}(S=x_S)\) を実施し、結果を記録します。
5. 追加データでGPを再学習し、次の実験へ進みます。

まだGPを学習できない初期段階では、Sobolなどの空間充填設計を利用できます。失敗した実験の結果を架空の観測値として学習してはいけません。また、候補提案は**介入の実行**ではありません。現実の装置への指示や安全保証と混同しないでください。

## 7. ベンチマークは何を比べているか

既知SCM \(\;X\to M\to Y\leftarrow Z\;\) の合成例で

\[
M=0.25+0.75X,\qquad
Y=-(M-0.8)^2-\tfrac12(Z-0.65)^2
\]

を考えます。\(do(X,Z)\) を許せば最大値は \(y^\star=0\) で、\(X=(0.8-0.25)/0.75,\ Z=0.65\) が最適介入の一例です。別の集合を介入した場合には、未介入変数の方程式が残ります。

結果の評価は**試行回数軸**と**介入コスト軸**を分けます。第 \(t\) 回までの真の目的値を \(y_1,\dots,y_t\) とすると、

\[
b_t=\max_{i\le t}y_i,\qquad
r_t=\max(0,y^\star-b_t),\qquad
C_t=\sum_{i=1}^t c(S_i).
\]

\(b_t\) は最良値、\(r_t\) はSimple Regret、\(C_t\) は累積コストです。試行回数が同じでも \(C_t\) は一致しません。比較は**同じ初期介入データ、同じ追加試行回数、同じシード**を使いますが、固定集合の普通のGP-BOと集合選択可能なCBOは探索できる行動空間も異なります。これだけで因果推論の優越性を立証するものではありません。

比較実験と図の読み方は[Notebook 59](../../examples/notebooks/59_causal_bo_benchmark.ipynb)と[Benchmark資料](../benchmarks/README.md)を参照してください。

## 8. 前提と限界

本章の基礎例は、既知で正しいと仮定したDAG、指定された介入集合、連続変数、単一目的、逐次 \(q=1\)、集合ごとの固定コストを扱います。構造探索・観測データからの因果効果同定・未知の交絡の処理・カテゴリ介入・一般の安全制約・並列介入は扱いません。実験の安全性や因果仮定の妥当性は、別途検証する必要があります。

## 参考資料

- Judea Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed., Cambridge University Press, 2009.
- Miguel A. Martínez-Cantin, *Bayesian Optimization in the Presence of Black-box Constraints*（制約と介入コストは異なる概念です）。
- [Gaussian Process理論](02_gaussian_process.md)、[獲得関数理論](04_acquisition_function.md)、[実装ガイド](../guide/28_causal_bayesian_optimization.md)
