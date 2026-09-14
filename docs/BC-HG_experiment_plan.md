# BC-HG：オンライン追従フォロワー下での実験計画

研究テーマ：同時間スケールで強化学習するフォロワーの下での評価および改良

# 目的と研究範囲

本研究の第一テーマは，強化学習によって追従学習するフォロワーの最適反応への収束を待たずにリーダーを更新した場合の BC-HG の評価および改良である．フォロワーの追従遅れ，サンプリング雑音，関数近似，actor–critic 不整合がハイパー勾配推定とリーダー性能に及ぼす影響を段階的に調べる．第二テーマであるエントロピー正則化リーダー目的への拡張は，第一テーマの評価・分析を優先した後に扱う． 本計画は共有された2026年9月13日版の論文，分割された configurable_mdp／markov_game コード，およびこれまでの議論に基づく．コードは静的に確認したものであり，学習の再実行や依存環境の動作確認を完了したという意味ではない．以下の予算や設定値は実験前の候補であり，予備実験後，独立した評価 seed を実行する前に確定する．

## 検証する問い

1.  既存 Carry-Q で観察された追従可能性は，サンプルベースの tabular-Q および SAC でも維持されるか．

2.  更新数不足とリーダーの変化速度は，追従誤差，ハイパー勾配方向，最終的なリーダー性能にどう影響するか．

3.  不調の主因は follower の未学習，critic／actor の不整合，leader critic，あるいは古い軌跡の利用のいずれか．

4.  追従状態に応じた更新制御は，同じ資源を用いた固定学習率や固定更新間隔より有効か．

BC-HG が必ず破綻することや改良が必ず優位になることを前提にしない．素朴な Online BC-HG が安定する領域を示すことも研究成果とする．

# 問題設定と情報アクセス

## 共通の目的

Conf-MDP ではリーダー変数を $\theta$，フォロワー方策を $g$ とする．MG ではリーダー方策を $f_\theta(a\mid s)$，リーダー行動を観測したフォロワー方策を $g(b\mid s,a)$ とする．状態を共通記号 $x$ で表すと，Conf-MDP では $x=s$，MG では $x=(s,a)$ である．

$$
J_F(\theta,g)=\mathbb{E}_{\theta,g}\sum_{t\ge0}\gamma_F^t[r_F^\theta(x_t,b_t)-\beta\log g(b_t\mid x_t)],\quad g^*(\theta)\in\arg\max_g J_F(\theta,g).
$$

$$
F(\theta)=J_L(\theta,g^*(\theta)),\quad h^*(\theta)=\nabla F(\theta).
$$

MG の $r_F^\theta$ は元の報酬が必ず陽に $\theta$ に依存するという意味ではなく，リーダーを固定した誘導過程を表す．温度 $\beta>0$，割引率，初期分布，報酬スケール，終端処理を全手法で統一する．最適反応の一意性や $F$ の微分可能性は対象クラスで確認し，連続空間で無条件に仮定しない．

## 主設定の情報契約

共有論文の方策・価値関数へのアクセス可能性を維持する．リーダーはフォロワーの現在の方策の評価・サンプリング，現在の $Q_F$／$V_F$ の評価，温度・割引率，および学習軌跡中の状態・行動・必要な報酬を利用する．Conf-MDP では適用する式に必要な既知の報酬微分，遷移・初期分布の score へのアクセスも明示する．決定的リーダー版で入力行動に関する価値微分が必要なら，そのアクセスを別途明記する． フォロワーの optimizer の状態，学習履歴を通した微分，フォロワー更新の Hessian，学習則の改変を主手法は要求しない．実験者による更新頻度・学習率の条件設定と，実行中のリーダーが follower optimizer を制御することは別である．主改良法はリーダー側を調整する．「最適化過程に依存しない」と「方策・価値も得られない完全ブラックボックス」は区別し，後者を本研究で実現したとは主張しない． 厳密解・高精度解は診断プロセスに隔離する．オンライン手法の入力に最適方策，真の価値，真のハイパー勾配を混入させない．oracle 条件だけは追加情報を使う参照比較として表示する．

## 推定する勾配

BC-HG の間接項は，leader にとっての follower 行動の便益と，follower soft value の感度の積から構成される．共通の説明用記号では，

$$
B_L(x,b)=Q_L(x,b)-\mathbb{E}_{b'\sim g(\cdot\mid x)}Q_L(x,b'),\quad S_F(x,b)=\nabla_\theta Q_F^*(x,b).
$$

間接項は適切な割引占有分布の下で $\beta^{-1}\mathbb{E}[B_LS_F]$ の形を持つ．直接項，割引正規化，MG の時間インデックス，入力への全微分は各設定の論文の式に従う．$S_F$ は follower の学習履歴を逆伝播して得る量ではなく，既存の score／報酬微分と trajectory suffix によって推定する．学習途中の $g_k,Q_{F,k}$ を代入した量を $h^*(\theta_k)$ の近似と位置付け，有限回学習そのものの meta-gradient と解釈しない．

# 既存実験との関係

有限回 soft Q-iteration は新規提案ではなく既存実験である．Four-Rooms では $K_F\in\{1,2,5,10,20,50\}$，基準100回，MG toy では $K_F\in\{1,2,5,10,50,100\}$，基準は最大1000回の更新が用いられた．原稿の更新数 $\eta$ は学習率との混同を避け，本計画では $K_F$ と記す． Reset-Q は各反復で Q を再初期化し，Carry-Q は有限回更新後の Q を次へ継承する．既存結果では Reset-Q の小更新数でギャップが残り，Carry-Q ではギャップが速やかに低下して性能差が小さい．評価には近似最適反応下のリーダー目的と，現在方策から参照方策への平均 KL が用いられている．

$$
D_{\mathrm{KL},\mu}(g\Vert g^*)=\mathbb{E}_{x\sim\mu}\sum_b g(b\mid x)\log\frac{g(b\mid x)}{g^*(b\mid x)}.
$$

この結果を全状態・行動に正確な作用素を適用できる追従の基準とし，RL 化に伴う追加の難しさを検証する．既存実装には有限回の中間 Q と診断用の収束側 Q の双方を計算する経路があるため，実際の計算時間を単に $K_F$ 回相当と見なさない．計算効率を測る際は診断用継続計算を別計上する．また既存の cosine 指標と，真の $h^*$ に対する cosine を取り違えない． 既存比較には leader actor を同一 follower 応答の下で複数回更新する設定が含まれる．したがって原稿の outer iteration と，本研究の「リーダー1更新」を同一視しない．新規主設定では1反復1回の leader actor 更新とし，既存再現条件は別に残す．

# 段階的な実験系列

1.  既存 Four-Rooms／MG toy の有限回 soft iteration：参照結果の整合性確認と必要な診断の追加．

2.  同タスクの tabular soft Q-learning：全状態更新をサンプル更新に置換．

3.  同タスクの SAC-Discrete：actor–critic と関数近似を導入．必要なら tabular actor–critic を中間に追加．

4.  Building Thermal Control および bilevel-LQR MG：既存連続タスクで RL follower を評価．

5.  CartPole／Pendulum 系のバイレベル化：非線形ダイナミクスと連続状態・行動へ展開．

6.  Reacher または HalfCheetah：高次元状態・多次元行動へ展開．

全条件を全タスクで実施しない．離散タスクで詳細な切り分けを行い，連続・高次元では代表条件を検証する．LQR 系と非線形系は，前者を診断性，後者を一般性の検証として併用する．

# 離散タスク

## Four-Rooms

既存の報酬設計，ゴール条件，初期分布，温度，リーダー制約を引き継ぐ．ランダムなゴール等が follower 最適方策に影響する場合，その条件を状態・文脈に含めるか，条件付き方策と Q を保持する．単純にゴールごとの Q を潰して部分観測問題に変更しない． まず leader を固定して follower 単体の学習を検証し，soft iteration 参照に近づくことを確認する．次に Carry-Q 型の状態保持で同時更新へ進む．Reset-Q の全面再探索は不要で，既存結果を動機として引用する．

## MG toy

既存のリーダー行動を観測する構造を保持し，$Q_F(s,a,b)$ と $g(b\mid s,a)$ を用いる．主診断には既存の $R=1$，追加の頑健性確認には $R=2$ を候補とする．MG では follower の次状態は $(s',a')$ であり，$a'\sim f_\theta(\cdot\mid s')$ の扱いを明記する．

## Tabular soft Q-learning

$$
V_Q(x)=\beta\log\sum_b\exp(Q(x,b)/\beta),\quad g_Q(b\mid x)=\exp((Q(x,b)-V_Q(x))/\beta).
$$

$$
Q(x,b)\leftarrow Q(x,b)+\alpha_Q[r_F^\theta(x,b)+\gamma_F(1-d)V_Q(x')-Q(x,b)].
$$

$d$ は真の終端を示す．人工的 time limit を継続問題の終端と誤認しない．ミニバッチでは同じ成分が複数回現れる場合の集約方式を固定する．一更新を一遷移更新とするかミニバッチ更新とするかを明示し，両方の数を記録する．MG toy では小さい離散 leader action に関する期待値を列挙できるが，そうする場合は期待値 target として明示する．環境遷移全体は列挙せず，サンプルベースという区別を保つ． 原則として現在の soft 方策で探索し，必要な初期ランダム探索・探索混合率は事前固定する．未訪問率，状態行動頻度，方策エントロピーを記録する．この段階では明示的 actor を持たないため，SAC との違いをすべて関数近似だけに帰さない．必要なら tabular actor–critic を加えて actor 最適化誤差を分離する．

## SAC-Discrete

共有コードの SACDiscrete と follower wrapper を出発点とする．実装の存在は動作・理論整合性の保証ではないため，固定 leader 下の単体試験を行う．現在の actor，twin critic，target critic を区別する．温度は固定し，報酬スケールとの比を一致させる．小規模タスクでは行動期待値を厳密和で計算し，連続行動サンプリングの誤差を持ち込まない．

$$
V^{g,Q}(x)=\sum_b g(b\mid x)[Q(x,b)-\beta\log g(b\mid x)],\quad V^{\mathrm{soft},Q}(x)=\beta\log\sum_b e^{Q(x,b)/\beta}.
$$

$$
V^{\mathrm{soft},Q}(x)-V^{g,Q}(x)=\beta D_{\mathrm{KL}}(g(\cdot\mid x)\Vert g_Q(\cdot\mid x)).
$$

同一の Q に対して成り立つこの恒等式を利用し，policy evaluation の soft value と，最適性作用素の log-sum-exp value を分ける．現在 actor が $g_Q$ と一致しない場合に両者を混用しない．BC-HG に渡す価値の選択は設定に保存し，離散診断で比較する．

# 既存連続タスク

## Building Thermal Control：Conf-MDP

既存の建物パラメータを leader が調整し，follower が温度偏差・制御コストを最小化する設定を保持する．Riccati follower から，まず Gaussian 方策と二次 critic の制限された RL follower，次に neural SAC follower へ移行する．前者は任意の必須ベンチマークではなく，SAC の不調を診断する橋渡し条件である． 重要な区別として，共有環境は温度クリッピングと状態依存の切断ノイズを含む．そのままでは純粋な線形・二次モデルではなく，Riccati 方策を厳密最適反応とは呼べない．(i) 非飽和の線形・適合ノイズ条件を定義した解析用 variant と，(ii) 既存環境を保持する実装 variant を分ける．後者の Riccati 解は近似参照として表示し，必要なら固定 leader 下の追加 RL 学習で改善余地を測る． 遷移を $\theta$ が変えるため，過去の $(s,b,s')$ を報酬再計算だけで現在の遷移へ変換することはできない．遷移 score を利用する際は，実際のクリッピング，切断分布の正規化，パラメータ依存性との整合性を確認する．正規分布の score を根拠なく代用しない．

## Bilevel-LQR MG

線形 Gaussian leader と線形・二次 follower という解析可能な構造を維持する条件から始める．follower は $(s,a)$ を入力に取る．共有 LQREnv-v4 の leader 報酬は一般の二次形式ではないため，follower の Riccati 解が得られても leader 目的の真のハイパー勾配が自動的に閉形式で得られるわけではない．leader 評価は数値積分，高精度 rollout，または共通乱数を用いる差分を参照とし，数値誤差を報告する． 共有 LQR の行動域は非有界である．非有界 Gaussian 最適反応と bounded tanh-Gaussian SAC を同一の最適化問題と見なさない．主解析用条件では非有界 Gaussian actor を用いた SAC 型学習を検討し，安定化条件・モーメントを監視する．有界行動 variant は別問題として実施し，Riccati 解を厳密 oracle と呼ばない．leader を任意の nonlinear neural policy に変えた場合も同様に解析可能性を再確認する． 状態の発散，非有限値，割引二次コストの有限性，安定化可能性を確認する．数値的な停止閾値は性能の悪い seed を削除する理由ではなく，失敗率として記録する．

# 非線形・高次元タスク

## CartPole と Pendulum

標準 Gymnasium CartPole は離散行動であり SAC-Discrete を使う［資料C］．一方，共有コードの GuidedCartPole は follower action を連続 Box に変更し，leader の作用も導入した独自 MG である．この variant には連続行動 SAC を使える．両者を同じ標準 CartPole と表示しない．共有 GuidedPendulum も初期候補とし，状態入力，報酬，leader の作用，適用する BC-HG 式を単体検証する． 簡単な Conf-MDP の新設候補は reward-design 型である．

$$
r_F^\theta(s,b)=r_0(s,b)+\theta^\top\varphi(s,b),\quad J_L(\theta,g)=\mathbb{E}_{\theta,g}\sum_t\gamma_L^t r_{\mathrm{target}}(s_t,b_t)-\lambda\|\theta-\theta_0\|^2.
$$

CartPole では姿勢維持と指定位置への誘導，Pendulum では姿勢・速度・制御入力に対する follower 報酬係数を候補とする［資料D］．角度誤差は周期性を反映し，$1-\cos(\vartheta-\vartheta^*)$ 等を用いる．leader 変数の範囲やコストは問題設定上の理由を持たせる．全報酬係数の同時拡大は固定温度に対する実効温度変更でもあるため，一係数固定や正規化によって尺度の自由度を制御する． 全特徴を potential-based shaping のみにすると最適方策が変わらない場合があるため，所望の response が生じるか事前確認する．最適値が内点にあることを必須とせず，境界解も許容する．直接項しか変わらない自明な問題や，単なる報酬尺度変更で解ける問題を避け，複数の $\theta$ で follower response が変わるか確認する． reward-design 型では物理遷移を $\theta$ が変更しないので，遷移 score 項はゼロとなる．決定論的な物理パラメータを直接 $\theta$ 化する場合に $\nabla_\theta\log p^\theta$ を形式的に適用しない．MG の確率的 leader score を使う導出とは分けて判断する．

## 高次元制御

Reacher または HalfCheetah の1タスクを優先し，余力があれば2タスク目へ進む．HalfCheetah の報酬設計例を以下とする［資料E］．

$$
r_F^\theta(s,b)=\theta_v v_x-\theta_u\|b\|^2,\quad r_{\mathrm{target}}(s,b)=-(v_x-v^*)^2-\lambda_u\|b\|^2.
$$

存在が未確認の contact cost を標準環境の報酬として導入しない．環境 ID，版，wrapper，観測・行動次元，報酬定義を固定する．非線形タスクの follower を長く訓練した解は厳密 oracle ではなく，追加学習予算を明示した参照応答と呼ぶ．

# 更新方式と時間スケール

## 計数単位

$N$ を一収集ブロックの新規遷移数，$U_Q,U_\pi$ をそれぞれ follower critic と actor の optimizer step 数，$U_C$ を leader critic の step 数とする．leader actor はブロック末に1回更新する．便宜的な $U_F$ は follower の一学習サイクル数であり，一サイクル中の twin critic／actor／target の更新内訳を固定する．actor を持たない tabular-Q では $U_F=U_Q$ とする．

$$
U_F=\frac{n_{F,\mathrm{cycle}}}{n_{L,\mathrm{actor}}},\quad \rho_\alpha=\frac{\alpha_L}{\alpha_{F,\pi}},\quad \rho_{\mathrm{eff}}\approx\frac{\alpha_L}{U_\pi\alpha_{F,\pi}}.
$$

Tabular-Q では分母に $\alpha_Q$ を使う．SAC では actor と critic の学習率を別記する．$\rho_{\mathrm{eff}}$ は解釈用の粗い目安であり，異なる勾配尺度，曲率，Adam の前処理を無視した厳密な速度比ではない． 共有コードの follower training 呼出し数，内部 gradient step 数，leader update interval，leader actor delay を別々に記録する．外側カウンタのみから $U_F$ を判断せず，実際の optimizer.step の呼出し数を計測する．

## 同時間スケールの意味

主設定は有限個の更新を繰り返す single-loop とする．減少学習率の理論では，適切に更新頻度を含めた利得比が正の有限値へ収束する設定を同時間スケールと呼ぶ．$U_F=1$ だけで同時間スケールや難しさは決まらず，固定の $U_F=20$ だけで漸近的二時間スケールになるわけでもない．大きい固定更新数は follower-fast 条件と呼び，毎回収束させる nested 条件とは区別する．定数学習率では相対速度と定常近傍を議論し，厳密な最適反応への追従を自動的に仮定しない．

## 主更新プロトコル

1.  $(\theta_k,g_k)$ を固定した短いブロックで $N$ 遷移を収集する．状態，両者の行動・報酬，次状態，終端と打切り，収集時の方策版，必要な報酬特徴を保存する．

2.  follower を $U_Q,U_\pi$ 回更新し，方策・critic・optimizer・buffer を継承する．

3.  学習に用いる方策版と収集版の差を記録し，leader critic を $U_C$ 回更新する．follower 更新直後の方策で必要な行動期待値と価値を評価する．

4.  同一推定器・同じ trajectory suffix 予算で BC-HG を推定し，leader actor を1回更新する．独立した sensitivity critic は現行の軌跡推定に必須ではなく，追加する場合は別の改良条件とする．

5.  target network を所定の間隔で更新し，所定チェックポイントで学習を停止して独立評価を行う．

この順序でもデータは更新前の follower 由来である．小さい drift を記録した上で近似と明示する．離散診断では，follower 更新後に新しい診断軌跡を集める条件と比較し，その追加サンプルも計上する．trajectory suffix が必要なため，$N=1$ にして自動的に静止方策の長い軌跡が得られるわけではない．ブロック長・suffix 長・打切りバイアスを固定し，後段でブロック内更新を許す fully streaming 条件へ進む．

## 開始条件

Warm-start を主診断とし，固定 $\theta_0$ で follower を事前学習してから同時更新へ移る．seed ごとに同じ初期 snapshot を各 leader 手法に複製し，optimizer と buffer も一致させる．事前学習時間・サンプル数を除外した曲線と，全費用に含めた曲線の双方を出す．Cold-start は未学習からの追加頑健性条件とする．Carry-Q は継承方式の名前であり，開始時点で最適反応へ事前学習済みという意味ではない．

# 更新数と学習率の実験設計

全タスクで完全な grid search は行わない．最良設定を選ぶ tuning と，固定条件下の感度分析を分離する．

1.  固定 leader 条件で follower が安定して学習する $\alpha_Q,\alpha_{F,\pi}$ を選び，独立 tuning seed で基準 $\alpha_{L,0}$ を決める．

2.  主感度分析では $N$，follower 学習率，leader 学習率を固定し，$U_F\in\{1,2,5,10,20\}$ を変える．各 $U_F$ で学習率を再調整して主感度曲線を作らない．

3.  $U_F\in\{1,5\}$ で $\alpha_L\in\{0.1,1,10\}\alpha_{L,0}$ を比較する．10倍が単体の最適反応条件でも不安定なら，tuning 段階で倍率範囲を狭める．

4.  小規模環境1つで $U_F\in\{1,5,20\}$ と上記3学習率の9条件を実施し，追従誤差・性能・失敗率を可視化する．高次元では固定した代表3条件に限定する．

固定 $N$ の下で $U_F$ を変える主条件は，データ再利用回数の効果である．別の比較で follower の update-to-data ratio を固定し，leader 更新間隔を広げて新規データを増やす．後者は leader 更新当たりのサンプル数も変えるため，前者と同一視しない． 主感度解析に加え，必要なら同じ tuning 予算を与えた各手法の best-tuned 比較を出す．同じ設定を全手法に強制する公平性と，手法ごとに適切に調整する公平性を混同しない．

# Replay・価値・温度の整合性

## Replay の比較

まず Carry-buffer と recent-window の2条件を優先する．Reset-buffer と recency-weighted は問題が確認されたときの追加 ablation とし，全実験で必須にしない．follower buffer と leader の軌跡 buffer は別の操作因子であり，片方ずつ変更する．Reset-buffer が十分なデータを蓄える前に再初期化され，学習不能にならないようにする． reward-design Conf-MDP では生の報酬特徴を保存し，現在の $\theta$ で再計算する．遷移を変える Conf-MDP ではこれだけでは不十分である．MG では $(s,a,b)$ を保持すれば条件付き物理遷移が不変な場合があるが，次の leader action，周辺状態分布，長い軌跡の方策分布は変わる．古い次 leader action をそのまま target に使う場合と，現在方策で期待値化する場合を区別する．recency weighting は厳密な分布補正ではない．

## 温度・密度・ネットワーク

主実験は $\beta_{\mathrm{follower}}=\beta_{\mathrm{BC-HG}}$ を固定する．automatic temperature tuning は追加実験に回し，目的の変化を明記する．SAC は最大エントロピー型の off-policy actor–critic であり，本問題との接続が自然である［資料B］． 連続方策では tanh と action scaling の log-density 補正を含める．連続 soft value の積分を有限個の Q の単純な log-sum-exp だけで代用しない．離散の log-sum-exp と連続の log-integral は異なる．連続 actor の KL は正規化定数を解析できる場合だけ絶対値として報告し，それ以外の actor objective や勾配ノルムは代理指標として扱う． target network，勾配 clipping，reward normalization，critic reset は基準条件で固定する．既にある target network の利用だけを新規改良とは呼ばない．連続 follower の twin critic の最小値を使う場合はその選択を記録し，理論の $Q^*$ と同一視しない．

# 比較手法と診断条件

1.  Online BC-HG：学習中の follower の方策・価値をそのまま使う基準．

2.  Naive-PGD：同じ follower と leader critic を用い，BC-HG の間接項を除く．leader 報酬設計では直接項が小さい／ゼロになる可能性も報告する．

3.  Tracking-aware BC-HG：後述の leader 更新制御．

4.  Follower-fast BC-HG：大きい有限 $U_F$ を用いる．追加計算を計上する．

5.  Best-response BC-HG：可能な小規模環境で follower を高精度に解くが，BC-HG の勾配推定はサンプルベースのままとする．

6.  Oracle-gradient 条件：可能な離散環境で真の数値勾配を用いる診断上の参照．前項とは区別する．

既存 HPGD／SoBiRL／Bi-AC との連続性は必要なタスクで確認するが，全手法のオンライン版を無条件に同列へ追加しない．各手法の利用情報と follower 更新の自由度を調べてから比較する．unrolled follower は学習過程にアクセスする別情報クラスの参考比較であり，主ベースラインの必須要件ではない． 誤差の切り分けでは，独立 snapshot 上で (i) follower 方策のみ参照解に置換，(ii) follower 価値のみ置換，(iii) leader critic のみ置換，(iv) fresh 軌跡に置換を行う．学習全体に oracle を混入させる条件とは分けて表示する．

# 評価指標

## 性能と追従

$$
J_{\mathrm{online},k}=J_L(\theta_k,g_k),\quad J_{\mathrm{BR},k}=J_L(\theta_k,g^*(\theta_k)),\quad \Delta_{F,k}=J_F(\theta_k,g^*)-J_F(\theta_k,g_k).
$$

$J_{\mathrm{online},k}$ は snapshot を固定した実行方策性能とし，実際の学習中の累積収益も別途保存する．$J_{\mathrm{BR},k}$ は leader を固定して follower が適応を完了した後の品質である．$J_F$ はエントロピー項を含む．近似参照では $\Delta_F$ が負になり得るため，負値をゼロへ隠さず，参照の不十分さも疑う． 離散では既存と同方向の $D_{\mathrm{KL}}(g_k\Vert g^*)$ を均一分布 $\mu$ で評価し，訪問分布による重み付き値も補助的に示す．MG では $\mu$ の対象を $(s,a)$ とする．状態・行動訪問数，方策 entropy，leader／follower の方策 drift も記録する．

## Hypergradient fidelity

$$
E_{h,k}=\|\widehat h_k-h_k^*\|_2,\quad E_{h,k}^{\mathrm{rel}}=\frac{\|\widehat h_k-h_k^*\|_2}{\|h_k^*\|_2+\varepsilon},\quad C_k=\frac{\langle\widehat h_k,h_k^*\rangle}{\|\widehat h_k\|_2\|h_k^*\|_2}.
$$

両ノルムが事前閾値以上の snapshot だけで $C_k$ と $C_k<0$ の頻度を集計し，除外率と閾値を報告する．真の勾配が小さい領域では絶対誤差を優先する．Adam，projection，clipping 後の実際の更新方向も記録し，raw 勾配の cosine だけで実際の改善を保証しない． 離散では soft Bellman 固定点・方策評価を十分な残差精度で解き，implicit differentiation と有限差分の一致を確認する．診断に限り完全モデルを使う．連続の差分参照は共通乱数，複数差分幅，追加 rollout で数値精度を確認し，「真の勾配」と断定しない．

## Residual と不整合

$$
R_{B,k}=\|Q_{F,k}-\mathcal{T}_{\theta_k}^*Q_{F,k}\|,\quad R_{A,k}=\mathbb{E}_{x\sim\mu}D_{\mathrm{KL}}(g_k(\cdot\mid x)\Vert g_{Q_{F,k}}(\cdot\mid x)).
$$

完全列挙できる離散では sup norm residual と平均値を併記する．サンプルの squared TD error は遷移雑音の分散も含み，そのまま真の Bellman residual ではない．独立な held-out データと，可能なら次状態期待値の評価を使う．SAC の通常の critic target は policy evaluation 型であり，最適性 residual と区別する．actor loss の絶対値も KL や方策距離と同一視しない．leader critic の検証誤差，target/current 差，suffix 長，buffer age も保存する．

## 学習効率と統計

学習環境ステップ，事前学習ステップ，評価ステップ，oracle 計算，follower actor／critic 更新，leader actor／critic 更新，wall-clock を別々に記録する．同一環境ステップ比較を主とし，更新数・実時間の比較を補う．異なる予算を同時にすべて一致させられるとは限らない． 最終 snapshot と事前指定した学習曲線 AUC を主指標とし，評価曲線から最良時点を事後選択しない．小規模主実験は20 seed，高次元は10 seed を初期案とし，不確実性が大きければ20へ増やす．tuning は別の3–5 seed とする．同じ seed ID と初期条件を手法間で対応付けるが，更新後に完全に同じ trajectory が生じるとは仮定しない．平均と95% bootstrap 区間，中央値，失敗率を示す．seed を再標本化単位とし，時間点や episode を独立試行扱いしない．異なる報酬尺度のタスクを単純平均しない．

# 改良候補と ablation

第一候補は，独立データで測る追従の代理 residual に基づく leader 更新幅の調整である．

$$
\widetilde R_k=c_B\widetilde R_{B,k}+c_A\widetilde R_{A,k},\quad \alpha_{L,k}=\frac{\alpha_{L,0}}{1+c\widetilde R_k},\quad \theta_{k+1}=\Pi_\Theta(\theta_k+\alpha_{L,k}\widehat h_k).
$$

各 residual を tuning 時の基準尺度で正規化し，必要に応じ EMA で平滑化する．連続領域では KL が得られない場合の代理量を明記する．学習率変更と trust-region clipping を同時に入れず，まず一方を検証する．projection は問題が制約付きの場合に使う． 必須 ablation は (i) 固定基準学習率，(ii) 同程度に小さい固定学習率を tuning した条件，(iii) residual による適応，(iv) 必要なら同じ leader 更新回数の固定間引きである．更新抑制によって収束が遅くなっただけの改善を避け，サンプル予算と最終到達度の両方を見る．この式だけで安定性保証が得られたとは主張しない． 追加候補は recent-window／recency weighting，follower 方策や価値の target 化，leader 方策 KL 制約，応答予測である．target 化は分散を下げる一方で遅れを増やし得る．応答予測と replay 補正は原因分析の必要性が確認された場合に限り追加する．

# 理論計画と実験との対応

## 誤差評価の目標

以下は未証明の目標形であり，成立済み定理ではない．まず有限空間，有界報酬，$\gamma<1$，正の固定温度，leader の滑らかさ，必要な方策正値性・score の制御，近似価値の有界性，適切なサンプリング条件を明示した上で検討する．

$$
\|\mathbb{E}[\widehat h_k\mid\mathcal{F}_k]-h^*(\theta_k)\|\le C_g d(g_k,g_k^*)+C_F\epsilon_{F,k}+C_L\epsilon_{L,k}+C_V\epsilon_{V,k}+C_S\epsilon_{S,k}+C_D\epsilon_{D,k}+\epsilon_{H,k}.
$$

$\mathcal{F}_k$ は評価に用いる新しい推定乱数を生成する前の履歴である．$\epsilon_F,\epsilon_L$ は follower／leader 価値近似，$\epsilon_V$ は使用する soft value と方策の不整合，$\epsilon_S$ は sensitivity 推定の系統誤差，$\epsilon_D$ は replay・分布・ブロック内非定常性，$\epsilon_H$ は軌跡打切りを表す．sampling variance は条件付き二乗誤差等で別に評価する．分布項を省くには対応する on-policy／mixing 条件が必要である．推定器によって不要な項や重複する項は整理する． 残差から tracking を制御する理論は別段階とする．有限表形式の最適性作用素なら $\|Q-Q^*\|_\infty\le\|Q-\mathcal{T}^*Q\|_\infty/(1-\gamma_F)$ を利用できる．SAC の empirical TD loss や actor loss にこの評価を直接当てはめない．critic が $Q^g$ を学ぶ設定と $Q^*$ を近似する設定も区別する．

## 追従と結合学習

全状態 Carry-Q について，$Q_{k+1}=\mathcal{T}_{\theta_k}^{K_F}Q_k$，$e_k=\|Q_k-Q^*(\theta_k)\|_\infty$ と置く．固定点が局所 Lipschitz なら，

$$
e_{k+1}\le\gamma_F^{K_F}e_k+L_Q\|\theta_{k+1}-\theta_k\|.
$$

これは更新順序と縮小性に基づく基本評価であり，既存 Carry-Q の解釈の出発点となる．サンプル更新では全成分が毎回縮小するわけではない．十分な訪問頻度，探索，混合性，martingale noise，学習率を条件に，複数更新ブロックや期待値での追従評価を目指す．単に上式へ雑音を足すだけで証明したことにしない． 同時間スケールの結合系では follower 固定点の安定性と leader 側の滑らかさに加え，相互結合の強さや利得条件を検討する．減少学習率だけで追従誤差がゼロになるとは主張しない．定数学習率では残留誤差を含む近傍評価を目指す．

$$
\frac{\sum_{k<T}\alpha_{L,k}\mathbb{E}\|\mathcal{G}(\theta_k)\|^2}{\sum_{k<T}\alpha_{L,k}}\le E_{\mathrm{opt}}(T)+E_{\mathrm{track}}(T)+E_{\mathrm{est}}(T)+E_{\mathrm{noise}}(T).
$$

制約なしでは $\mathcal{G}=\nabla F$，制約付きでは適切な projected gradient mapping を用いる．係数・rate・成立条件は導出後に確定する．deep SAC 全体の大域収束や，非有界 LQR への有界報酬定理の直接適用は目標にしない．LQR はモーメント・安定性条件を別に扱う．最適反応での勾配表現が恒等的でも，学習途中の推定器へ既存の不偏性・収束定理が自動的に移るとはしない．

# 追加テーマ：leader entropy

第一テーマを確立した後，MG の確率的 leader に対して以下を検討する．

$$
F_\tau(\theta)=\mathbb{E}_{\theta,g^*(\theta)}\sum_t\gamma_L^t[r_L(s_t,a_t,b_t)-\tau\log f_\theta(a_t\mid s_t)].
$$

entropy 項の陽な $\theta$ 微分だけでなく，状態分布および follower response の変化も含めて導出する．単に通常の entropy gradient を付け足すだけで済むとは仮定しない．静的設計変数しか持たない Conf-MDP にはそのまま leader policy entropy を付けない． $\tau=0$，固定 $\tau$，annealing を比較し，entropy を含む目的と含まない元の leader return を両方報告する．follower 温度は固定して因子を分離する．目的を変える効果と tracking の安定化を区別する．

# 計算資源・予算・実施順序

利用上限は Xeon Gold 6230（20C/40T）2基，A5000 3基，RAM 32 GB 6枚であり，合計40物理コア／80スレッド，192 GBである．GPU メモリ容量・ドライバは実機で確認する．最初はGPUあたり1プロセスで測定し，小モデルはメモリ・性能を測って同居を増やす．独立 seed／条件を並列化し，単一学習を3GPUへ分散する設計を前提にしない．JAX と PyTorch／garage の依存環境は分離する． 初期CPU並列度は8–12ジョブを目安に，各ジョブの BLAS／Torch threads を制限して過剰並列を避ける．GPU の主ジョブは3本並列を基本とし，CPU・GPU・RAM 使用率から調整する．JAX のメモリ確保方式とプロセスのGPU割当も確認する． 最初の profiling は各タスク1–2 seed，予定ステップの1–5%とする．warm-up／JIT compile を分け，step/sec，ピークRAM／GPUメモリ，checkpoint容量，評価費用を測る．主予算の初期案は，離散20–50万，既存LQR20–50万，CartPole／Pendulum20–100万，高次元100–300万の学習遷移／seedとする．これは性能保証ではなく，固定 leader 下でも未学習なら見直す． 最小主比較を3手法，$U_F\in\{1,5,20\}$，20 seedとすると，1タスク180 runとなる．5点の更新数 sweep は基準法を中心にし，追加2点は40 run／タスクである．9条件の交互作用を3手法20 seedで行えば540 run／タスクに達するため，全タスクで繰り返さず，まず独立pilot seedで必要な領域を特定する．高次元では3手法・3条件・10 seedで90 run／タスクを上限の初期案とする．参照条件・追加 ablation は別枠で数える．

$$
T_{\mathrm{GPU,total}}=\sum_j n_jt_j,\quad T_{\mathrm{elapsed}}\gtrsim T_{\mathrm{GPU,total}}/3.
$$

$t_j$ は profiling から得た1 runのGPU占有時間である．下限式にはCPU待ち・評価・I/O・再実行・占有率低下が含まれない．CPU中心ジョブは別に見積もる．総研究期間の上限はまだ指定されていないため，資源仕様だけから完了日を断定しない．

## マイルストーン

1.  M0：既存基準の短い再現，依存環境，入力・報酬・温度・更新カウンタの試験を完了．

2.  M1：離散で固定 leader 下の tabular／SAC follower が学習し，参照解との評価が動作．ここが不成立なら同時学習へ進まず原因を切り分ける．

3.  M2：離散のオンライン主比較，限定grid，snapshot誤差分解を完了．不調原因が明確なら改良を選ぶ．

4.  M3：LQR系の解析用条件と既存環境条件を区別して検証し，非線形タスクへ進む．

5.  M4：代表的な高次元条件を独立seedで確認し，必要な頑健性実験を追加．

6.  M5：第一テーマの結果・理論をまとめ，余力があればleader entropyを追加．

# 付録A：記録・再現性・受入条件

各 run に一意のIDを付け，コードの版，完全設定，環境ID，seed，初期snapshot，開始条件，方策・critic構造，温度，学習率，optimizer，収集数，更新数，期待値サンプル数，buffer方式，評価方式，予算，実時間を保存する．学習snapshotには両者の方策・critic・target・optimizer・乱数状態と，必要ならbufferを含める．大型bufferは再現コストと保存容量を比較して方針を決める． 学習過程を自動微分していないこと，leader 更新で follower optimizer を進めないこと，oracle の計算結果がonline入力へ流れないことを試験する．episode の真の終端と time-limit の分離，follower の拡張状態，SACの密度補正，報酬と Q の符号，discount と温度の一致，leader critic の確率的行動期待値を確認する．確率的方策の評価で mode action を使う実装と，期待収益を定義する理論の違いを無視しない． 最終成果の受入条件は，(i) 主条件の独立seed評価と信頼区間，(ii) 両者の実更新数と費用の記録，(iii) 少なくとも離散での追従と勾配の対応，(iv) 改良の固定学習率対照，(v) oracle／近似参照の区別，(vi) 理論の仮定・未証明部分の明記である．

# 付録B：根拠資料

資料A  
共有された和文拡張原稿「Proposal and Evaluation of a Sample-Efficient Hypergradient Estimation for Decentralized Bi-Level Reinforcement Learning」2026年9月13日版：準最適フォロワー評価・結論．同日版英文原稿と共有 BC-HG コードも参照．既存実験の出典であり，新規実験結果ではない．

資料B  
[Haarnoja et al. (2018), Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://proceedings.mlr.press/v80/haarnoja18b.html)

資料C  
[Gymnasium, Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

資料D  
[Gymnasium, Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/)

資料E  
[Gymnasium, Half Cheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)

外部資料は2026年9月14日に確認した．本計画の具体的な設計・予算・未証明の理論目標は，上記資料の主張ではなく，本研究向けの提案である．
