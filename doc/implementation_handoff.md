# BC-HG 実験実装引継ぎ書（M0–M1）

作成日：2026-09-14  
対象研究：同時間スケールで強化学習するフォロワーの下での BC-HG の評価および改良  
実装対象：マイルストーン M0–M1  
実装対象外：M2 以降のオンライン leader–follower 同時学習、tracking-aware 改良、連続・高次元環境、leader entropy

## 1. この文書の役割

この文書は、新しい Codex チャットまたは別の実装担当者が、過去の会話履歴に依存せず、既存 BC-HG コードの調査から M0–M1 の実装・試験までを開始するための引継ぎ書である。

本書だけで研究全体の仕様を置き換えるものではない。数式、既存実験条件、情報アクセス条件の最終的な根拠は、次節の資料と実際のコードにある。不明点は、まず資料とコードを照合し、それでも実装結果が変わる場合に限って質問する。

## 2. 最初に読む資料

以下の順序で確認する。

1. `implementation_handoff.md`（本書）
2. `BC-HG_experiment_plan.md`
3. `BC-HG_research_plan.md`
4. 既存コードの `README.md`、設定ファイル、実行スクリプト、テスト
5. BC-HG の和文拡張原稿
   - `Proposal and Evaluation of a Sample-Efficient Hypergradient_Estimation_for_Decentralized_Bi-Level_Reinforcement_Learning_20260913.tex`
6. BC-HG の英文原稿
   - `Sample-Efficient Hypergradient_Estimation_for_Decentralized_Bi-Level_Reinforcement_Learning_20260913.tex`
7. 必要な場合のみ、博論研究概要および方策反復法論文

ファイル名の先頭に整理用の番号が付いている場合がある。その場合も、題名が一致するファイルを用いる。

既存コードは、少なくとも以下に相当する実装を対象とする。

- `configurable_mdp/`
- `markov_game/`
- 必要に応じて `figures/`

ZIP しか存在しない場合は、共通の親フォルダへ展開してから調査する。コードが存在しない場合は実装を推測で新設せず、不足物として報告する。

## 3. 研究目的と今回の境界

BC-HG は、最大エントロピー目的を持つフォロワーの応答を利用して、リーダーのバイレベル目的のハイパー勾配を推定する。既存の Four-Rooms および MG toy では、全状態・行動へ有限回 soft Q-iteration を適用する Reset-Q／Carry-Q 条件が評価されている。

新研究では、フォロワーをサンプルベースの RL に置き換え、最適反応への収束を毎回待たずに leader を更新する。ただし今回の M0–M1 では、オンライン同時学習に進む前の基礎を確立する。

- **M0**：既存基準の短い再現、依存環境、入力、報酬、温度、更新カウンタの試験を完了する。
- **M1**：離散環境で固定 leader 下の tabular soft Q-learning および SAC-Discrete follower が学習し、参照解との評価が動作する状態にする。

M1 が成立しない場合、M2 の leader–follower 同時更新へ進んではならない。学習不能、評価不能、理論との不整合のいずれかを切り分ける。

## 4. 情報アクセス契約

主手法が利用してよい情報は、共有論文の条件に合わせる。

利用可能：

- フォロワーの現在の方策の評価およびサンプリング
- フォロワーの現在の $Q_F$／$V_F$ の評価
- follower の温度 $\beta$ と割引率 $\gamma_F$
- 学習軌跡中の状態、両者の行動、報酬、次状態、終端・打切り情報
- 適用する式が必要とする既知の報酬微分
- Conf-MDP で必要となる場合の遷移・初期分布の score。ただし実際の確率モデルと整合するものに限る

主手法が要求してはならない情報・操作：

- follower の学習履歴を通した自動微分
- follower optimizer の内部状態を leader 勾配計算へ利用すること
- follower 更新の Hessian
- 実行中の leader が follower の学習則を改変すること
- oracle の最適方策、真の価値、真のハイパー勾配をオンライン手法の入力へ流すこと

実験者が比較条件として更新頻度や学習率を設定することと、オンライン leader が follower optimizer を制御することを混同しない。「フォロワーの最適化過程に依存しない」と「方策・価値も観測できない完全ブラックボックス」も区別する。本研究の主設定は後者ではない。

## 5. 変更前に行うコード調査

最初はコードを変更せず、以下を調査して報告する。

1. リポジトリ構成、使用言語、主要ライブラリ、パッケージ管理方法
2. `configurable_mdp` と `markov_game` の実行入口
3. Four-Rooms と MG toy の環境実装および登録方法
4. 既存 soft Q-iteration、Reset-Q、Carry-Q の実装位置
5. 既存 `SACDiscrete` と follower wrapper の実装位置
6. leader、follower、環境間で受け渡される observation、action、reward、done 情報の形状と意味
7. 温度、割引率、報酬スケール、終端処理の設定箇所
8. follower training の呼出し数と、内部の実 `optimizer.step()` 数
9. leader actor／critic の更新回数、更新間隔、actor delay
10. replay buffer の内容、保存単位、初期化・継承条件
11. seed、checkpoint、ログ、設定保存の既存機構
12. 既存の参照解、KL、cosine、最適性ギャップ等の診断コード
13. 論文・計画とコードの不一致、または意味が確定できない箇所
14. 既存ユーザー変更と未コミット差分

調査報告には、変更予定ファイル、各変更の理由、既存挙動への影響、受入試験を含める。既存コードに同等機能がある場合は再実装せず、最小限の修正・計測追加で再利用する。

## 6. 数学的・実装上の不変条件

### 6.1 共通記法

Conf-MDP では $x=s$、MG では follower の拡張状態を $x=(s,a)$ とする。MG の follower は leader action を観測する。

$$
J_F(\theta,g)=\mathbb{E}_{\theta,g}\sum_{t\ge 0}\gamma_F^t
\left[r_F^\theta(x_t,b_t)-\beta\log g(b_t\mid x_t)\right],
\qquad \beta>0.
$$

### 6.2 Tabular soft Q-learning

$$
V_Q(x)=\beta\log\sum_b\exp(Q(x,b)/\beta),
\qquad
g_Q(b\mid x)=\exp\!\left(\frac{Q(x,b)-V_Q(x)}{\beta}\right).
$$

$$
Q(x,b)\leftarrow Q(x,b)+\alpha_Q
\left[r_F^\theta(x,b)+\gamma_F(1-d)V_Q(x')-Q(x,b)\right].
$$

ここで $d$ は真の終端だけを表す。time-limit truncation を継続問題の終端として扱わない。log-sum-exp は数値安定な実装を用いる。報酬最大化とコスト最小化の符号をコード全体で確認し、暗黙に反転させない。

MG toy では次の follower 状態は $(s',a')$ である。$a'\sim f_\theta(\cdot\mid s')$ をサンプルするのか、離散 action を列挙して期待値を取るのかを設定とログで明示する。後者を使う場合も、環境遷移全体を列挙する model-based 更新と混同しない。

ミニバッチ内で同じ表成分が複数回現れる場合の更新規則を固定する。一遷移更新数、ミニバッチ更新数、収集した新規遷移数を別々に数える。

### 6.3 SAC-Discrete

現在 actor、twin critic、target critic を区別する。温度は M1 の主条件では固定し、論文・既存基準と報酬スケールに対する比を一致させる。小規模な離散 action の期待値は原則として厳密和で計算する。

同一の $Q$ に対して、次を区別する。

$$
V^{g,Q}(x)=\sum_b g(b\mid x)\left[Q(x,b)-\beta\log g(b\mid x)\right],
$$

$$
V^{\mathrm{soft},Q}(x)=\beta\log\sum_b\exp(Q(x,b)/\beta).
$$

$$
V^{\mathrm{soft},Q}(x)-V^{g,Q}(x)
=\beta D_{\mathrm{KL}}\!\left(g(\cdot\mid x)\Vert g_Q(\cdot\mid x)\right).
$$

SAC critic の policy-evaluation target と、soft Bellman 最適性作用素の log-sum-exp target を混用しない。actor が $g_Q$ と一致しない場合、両 value は等しくない。BC-HG に渡す value の選択は、後の M2 で追跡可能な設定項目として保存する。ただし M1 では leader 更新を開始しない。

### 6.4 終端と打切り

ログおよび replay では、少なくとも次を区別する。

- 真の環境終端 `terminated`
- time limit 等による `truncated`
- bootstrap に用いるマスク

既存 API が単一 `done` を使う場合、その生成規則を明示し、必要なら後方互換性を保ちながら分離する。

### 6.5 参照解の隔離

離散環境では、完全モデルを使う soft Bellman 固定点や高精度 soft Q-iteration を診断用参照として使用してよい。ただし参照計算は以下を満たすこと。

- online follower の replay、target、action selection、optimizer update へ値を流さない
- online 計算費用と oracle／診断計算費用を別々に記録する
- 「高精度参照」と「厳密解」を数値残差に応じて区別する
- 参照計算を継続した回数を、online follower の更新数に含めない
- 既存の cosine 指標を、真の $h^*$ に対する cosine と取り違えない

## 7. M0：既存基準と計測基盤

### 7.1 目標

既存コードを壊さずに短い再現実行ができ、M1 の結果を解釈するための設定・計測・試験基盤が整っていること。

### 7.2 作業項目

- 依存環境を再現し、CPU での最小 smoke test を優先する
- Four-Rooms と MG toy の既存実行入口を確認する
- 既存の有限回 soft Q-iteration を短い seed・予算で実行する
- 既存基準の Reset-Q／Carry-Q 条件を、全面的な主実験ではなく短い回帰条件として確認する
- observation、leader action、follower action、reward、next observation の shape・dtype・範囲を自動検査する
- Four-Rooms のランダム goal 等が follower 最適方策に影響する場合、goal／context が状態または条件として保存されていることを確認する
- MG toy で $Q_F(s,a,b)$ と $g(b\mid s,a)$ が leader action を落としていないことを確認する
- follower／leader の報酬、符号、割引率、温度、初期分布、終了条件を設定とログへ保存する
- 実際の optimizer step、表形式更新、target update、環境 step を計測するカウンタを追加する
- seed、完全設定、コード版、実行時間を保存する
- checkpoint の保存・読込について、少なくとも方策、critic、target、optimizer、乱数状態を検査する。buffer を保存しない場合はその方針を記録する
- true termination と truncation の単体試験を追加する
- oracle／診断経路から online 経路への情報流入がないことを試験する

### 7.3 M0 の最低受入条件

- クリーンな手順で依存環境を構築できる、または未解決依存を再現可能な形で報告できる
- Four-Rooms と MG toy の環境 reset／step smoke test が通る
- 既存の短い baseline 実行が最後まで完走し、主要メトリクスが有限値で保存される
- 同一 seed・同一設定で初期状態と短い実行が再現する
- shape、dtype、reward sign、$\gamma_F$、$\beta$、termination mask の試験が通る
- follower critic／actor、leader critic／actor、target、environment の各更新数を区別できる
- 診断用の追加 soft Q-iteration が online 更新数へ混入しない
- 実行コマンド、設定、出力場所、既知の問題が文書化される

既存論文と数値が完全一致しない場合でも、直ちに値を合わせるための恣意的変更をしない。seed、依存版、評価方法、設定、コード差分を順に切り分ける。

## 8. M1：固定 leader 下の離散 follower 学習

### 8.1 対象と順序

以下の順序を守る。

1. Four-Rooms + tabular soft Q-learning
2. MG toy + tabular soft Q-learning
3. Four-Rooms + SAC-Discrete
4. MG toy + SAC-Discrete

SAC-Discrete が不調な場合、必要性が確認されてから tabular actor–critic を中間診断として追加する。最初から実験範囲を拡大しない。

各条件では leader を固定し、leader optimizer を進めない。まず固定 leader 下で follower 単体が学習することを確認する。

### 8.2 Tabular follower の実装・試験

- 全状態更新ではなく、収集遷移に基づく sample-based 更新を実装または有効化する
- 原則として現在の soft 方策で探索する
- 必要な初期ランダム探索や探索混合率は設定として固定し、実験後に都合よく変えない
- Q、soft value、Boltzmann policy の有限性・正規化を試験する
- 状態・行動訪問数、未訪問率、方策 entropy を保存する
- update-to-data ratio を復元できるよう、新規遷移数と更新数を保存する
- 参照 $Q^*$／$g^*$ と現在 $Q$／$g$ を同じ状態定義・温度・割引・終端条件で比較する
- 学習曲線が参照へ近づくかを、複数の指標で確認する

### 8.3 SAC-Discrete follower の実装・試験

- 既存 `SACDiscrete` と follower wrapper を出発点とし、まず単体試験を行う
- actor、twin critic、target critic の出力 shape と action 軸を試験する
- discrete action expectation が正しい軸で厳密和されていることを試験する
- 固定温度と automatic temperature tuning を混在させない
- critic target、actor loss、entropy 項の符号とスケールを試験する
- target network の更新間隔・soft update 係数を設定とログへ保存する
- replay warm-up、batch size、actor delay、critic step、actor step を別々に記録する
- $V^{g,Q}$ と $V^{\mathrm{soft},Q}$ の双方を診断可能にし、actor–Boltzmann 不整合を測る
- 関数近似器の損失が有限でも方策が参照へ近づくとは限らないため、actor loss 単独で成功判定しない

### 8.4 M1 で最低限保存する評価値

固定 leader $\theta_0$ ごとに、少なくとも以下を保存する。

- entropy 項を含む follower return $J_F(\theta_0,g_k)$
- 参照 follower return $J_F(\theta_0,g^*)$
- follower optimality gap
  $$
  \Delta_{F,k}=J_F(\theta_0,g^*)-J_F(\theta_0,g_k)
  $$
- 既存論文と同方向の policy KL
  $$
  D_{\mathrm{KL},\mu}(g_k\Vert g^*)
  $$
- uniform $\mu$ による KL。補助的に訪問分布重み付き KL
- Q error（参照と比較可能な離散条件）
- soft Bellman optimality residual の sup norm と平均値（完全列挙可能な診断）
- SAC の actor–Boltzmann KL または同値な value gap
- critic の held-out TD 指標。ただし真の Bellman residual と呼ばない
- 状態・行動訪問数、未訪問率、方策 entropy
- 新規環境遷移数、評価遷移数、oracle 計算、表更新数、critic step、actor step、target update 数、wall-clock

近似参照の不足により $\Delta_F<0$ となる可能性を隠さない。負値をゼロに clipping せず、参照解の残差と追加計算予算を確認する。

### 8.5 M1 の受入条件

M1 は、単にコードが完走するだけでは完了としない。各離散タスクについて次を満たすこと。

- leader が全学習中に固定され、leader optimizer step が 0 であることを自動確認できる
- tabular follower が複数 seed で学習し、初期値より follower return、policy KL、Q error または残差のうち複数が改善する
- SAC-Discrete follower が複数 seed で学習し、return と policy／value 診断の少なくとも一方が参照方向へ改善する
- 現在方策と参照方策を同じ observation/context 定義で比較している
- Four-Rooms の goal/context と MG の leader action が入力から欠落していない
- true termination と truncation に対する bootstrap が意図どおりである
- 温度、割引率、報酬スケール、参照解の条件が online follower と一致する
- oracle／診断情報を除いても学習結果が変わらないことを、oracle 無効化試験等で確認する
- 保存ログから、学習データ量とすべての実更新数を復元できる
- 結果、設定、失敗 seed、既知の制約が短い M1 レポートにまとめられる

「改善」の数値閾値は、既存コードと baseline を調査した後、評価 seed の本実行前に設定ファイルまたは試験仕様へ固定する。結果を見てから成功閾値を変更しない。

## 9. 失敗時の切り分け順序

固定 leader 条件が不成立の場合、次の順に調べる。

1. observation/context の欠落、shape、action index、dtype
2. reward の意味・符号・スケール、$\gamma_F$、$\beta$
3. true termination／truncation と bootstrap mask
4. softmax／log-sum-exp の数値安定性と方策正規化
5. 状態・行動の訪問不足、初期探索、replay warm-up
6. Q target または SAC target の数式と action expectation
7. actor、critic、target の更新回数と更新順序
8. learning rate、batch size、target update、gradient norm
9. 参照解側の状態定義・目的・残差
10. seed・依存ライブラリ版・既存実装との差

デバッグのために oracle 値を一時利用する場合、そのコードパスは診断専用に隔離し、通常実行では無効であることを試験する。学習不能 seed を黙って除外せず、失敗率と原因を記録する。

## 10. 実装方針

- 既存 API、設定形式、実験結果を可能な限り維持する
- 大規模な全面改修より、M0–M1に必要な小さい変更を段階的に行う
- 1つの論理変更ごとに単体試験または smoke test を追加する
- 既存の未コミット変更を上書き・削除しない
- ハードコードせず、温度、割引率、seed、更新数、評価間隔等を設定化する
- 実験出力へ完全設定とコード版を保存する
- 乱数源を列挙し、Python、NumPy、PyTorch／JAX、環境へ一貫して seed を設定する
- CPU smoke test と GPU 本実行を分ける
- JAX と PyTorch／garage の依存環境が競合する場合は環境を分離する
- M0–M1 の段階では連続環境、tracking-aware 更新、leader entropy を追加しない

## 11. 推奨テスト構成

既存テスト規約に合わせ、少なくとも次を用意する。

### 単体試験

- numerically stable soft value と Boltzmann policy
- policy normalization と有限値
- tabular Q update の手計算可能な1遷移
- terminal／truncated target
- MG の $(s,a)$ 入力と次 leader action expectation
- SAC-Discrete の action expectation 軸
- $V^{\mathrm{soft},Q}-V^{g,Q}=\beta D_{\mathrm{KL}}(g\Vert g_Q)$
- 各更新カウンタ
- oracle 情報遮断
- checkpoint round-trip

### 統合試験

- Four-Rooms の短い固定-leader tabular 学習
- MG toy の短い固定-leader tabular 学習
- Four-Rooms の短い固定-leader SAC-Discrete 学習
- MG toy の短い固定-leader SAC-Discrete 学習
- 同一 seed の短い再現性
- 既存 soft Q-iteration baseline の回帰試験

### 試験の時間区分

- `unit`：数秒から数十秒
- `smoke`：CPU で数分以内を目標
- `integration`：短い学習を含み、通常 CI とは分離可能
- `experiment`：本評価。自動テストとは別管理

## 12. 成果物

M0–M1 完了時に、以下を残す。

1. 変更済みコード
2. 追加・更新したテスト
3. 依存環境の構築手順または lock file
4. M0／M1 の設定ファイル
5. CPU smoke test コマンド
6. GPU 実行コマンドと必要資源
7. ログ schema または項目一覧
8. 短い `M0_report.md`
9. 短い `M1_report.md`
10. 未解決事項と M2 へ進めるかの判定

レポートには、実行コマンド、コード版、依存版、seed、完全設定、結果表、所要時間、失敗条件を記載する。

## 13. M2 へ進むためのゲート

次をすべて満たすまで、leader を更新するオンライン主比較へ進まない。

- M0 の最低受入条件を満たす
- 固定 leader 下で tabular follower の学習が確認される
- 固定 leader 下で SAC-Discrete follower の学習、または不成立原因の明確な診断が得られる
- 参照解との比較経路が動作し、online／oracle の情報が分離される
- 更新数と計算費用が正しく計測される
- M1 レポートで、M2 へ進む判断が明示される

SAC-Discrete が一部条件で不成立でも、原因が actor 最適化、関数近似、探索等に切り分けられ、tabular 条件が成立している場合は、研究上の判断として限定的に M2 へ進む余地がある。ただし担当者が独断で進めず、結果と選択肢を提示して確認を取る。

## 14. 実装担当 Codex への開始指示

最初の応答ではコードを変更せず、次を提示する。

1. 読み取ったリポジトリ構成と実行入口
2. M0–M1について既に実装済みの機能
3. 不足機能と理論・計画との不一致
4. 変更予定ファイルと変更理由
5. 段階的な実装計画
6. 受入テスト一覧
7. 実装を始める前に確認が必要な、本当に未解決の質問

その後、M0 を実装・試験し、結果を報告してから M1 へ進む。既存実装を確認せずに新しい訓練基盤を作り直したり、M2 以降の機能を先回りして追加したりしない。

## 15. 根拠と優先順位

仕様が衝突した場合は、次の優先順位で判断する。

1. ユーザーの最新の明示指示
2. 共有論文における問題設定・情報アクセス条件・数式
3. `BC-HG_experiment_plan.md`
4. `BC-HG_research_plan.md`
5. 本引継ぎ書
6. 既存コードの慣例

コードと論文が食い違う場合、どちらかを無言で正しいものとして扱わず、再現条件と新規研究条件を分離して報告する。既存再現を保つ互換モードと、新規の理論整合モードを分ける必要がある場合は、その理由と設定差を明示する。
