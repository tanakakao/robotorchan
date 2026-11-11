# 組合せベイズ最適化（Combinatorial Bayesian Optimization）

## 1. 問題設定

組合せ最適化は有限集合 \(\mathcal X=\prod_{j=1}^{d}\mathcal X_j\) 上で
未知の目的関数 \(f:\mathcal X\to\mathbb R\) を少ない評価回数で最大化する問題です。
二値変数では \(\mathcal X=\{0,1\}^d\) であり、候補数は \(2^d\) と指数的に増えます。
連続空間の勾配法をそのまま適用するより、離散構造を利用することが重要です。

## 2. Hamming近傍とDiscrete Trust Region

Hamming距離は \(d_H(x,x')=\sum_j\mathbf 1[x_j\ne x'_j]\) です。
半径1の近傍探索では1変数だけ異なる候補を評価し、獲得関数が改善する候補へ移動します。
多点初期化で局所解への依存を軽減できますが、大域最適性は保証しません。

Discrete Trust Regionは中心 \(c_t\) と半径 \(r_t\) に対して
\(\mathcal T_t=\{x:d_H(x,c_t)\le r_t\}\) を探索します。
改善の連続回数で半径を拡大し、失敗が続けば縮小または再開始します。
この領域制限は探索の局所性を制御するもので、安全制約を保証するものではありません。

## 3. BOCSの二次モデル

二値入力に対して
\[
f(x)\approx \beta_0+\sum_i\beta_i x_i+
\sum_{i<j}\beta_{ij}x_ix_j
\]
と展開します。係数の事前分布と観測ノイズを指定して事後分布を求め、
期待改善量やUCBなどの獲得関数で次の候補を選択します。
原著BOCSではスパースな事前分布などが重要ですが、単純なガウス事前分布を使う実装は
**BOCS-style二次ベイズ回帰**であり、原著の完全な再現ではありません。

## 4. COMBOとグラフ拡散カーネル

各離散変数をグラフの頂点とみなし、直積グラフ上の拡散を用いて候補間の類似性を表します。
二値空間では各次元が2頂点のグラフであり、各次元の拡散カーネルの積から
ガウス過程の共分散を構成できます。
\[
k(x,x')=\prod_{j=1}^{d}k_j(x_j,x'_j)
\]
この構造により、Hamming的な近接性を反映した事後予測を行えます。
一方、カーネルだけを実装して全候補列挙でUCBを最大化する方法は、
原著COMBOの全ての探索・推論手法を再現したものではありません。

## 5. 候補選択と計算量

UCBは最大化問題で \(a(x)=\mu(x)+\sqrt{\beta}\sigma(x)\) と表せます
（\(\beta\) のパラメータ化は使用する獲得関数に従います）。
全候補列挙は小規模二値問題には有効ですが、\(O(2^d)\) の候補数を要します。
大規模な問題では近傍探索、TR、近似探索などが必要です。

## 6. 適用範囲

本章のHamming距離は整数・カテゴリにも一般化できますが、BOCS二次モデルと
COMBOの現在の逐次探索は二値入力・単一目的・\(q=1\) を基本とします。
制約・Mixed・バッチ最適化は、各モデルと候補探索手法で対応状況が異なります。
実装上の制限は[Guide](../guide/27_combinatorial_bayesian_optimization.md)を参照してください。

## 参考文献

- Baptista & Poloczek (2018), *Bayesian Optimization of Combinatorial Structures* (BOCS).
- Oh et al. (2019), *Combinatorial Bayesian Optimization using the Graph Cartesian Product* (COMBO).
