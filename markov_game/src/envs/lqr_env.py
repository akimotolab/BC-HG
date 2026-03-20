import os
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from collections import deque
import akro

class LQREnv_0(gym.Env):
    def __init__(self, 
                 A=[[0.0, 1.0], 
                    [0.0, 0.0]], 
                 B=[[0.0], 
                    [1.0]], 
                 C=[[1.0], 
                    [0.0]], 
                 Q=[[1.0, 0.0], 
                    [0.0, 1.0]], 
                 R=[[1.0]], 
                 cost_lambda=1.0):
        super().__init__()
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.C = np.asarray(C)
        self.Q = np.asarray(Q)  # positive semi-definite
        self.R = np.asarray(R)  # positive definite
        self.cost_lambda = cost_lambda
        self._obs_dim = self.A.shape[0]
        self._action_dim = self.B.shape[-1]  
        self._leader_action_dim = self.C.shape[-1]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._action_dim,), dtype=np.float32
        )
        self.leader_action_space = akro.Box(
            low=-np.inf, high=np.inf, shape=(self._leader_action_dim,), dtype=np.float32
        )

        # For rendering
        self.viewer = None
        self.state_history = deque(maxlen=1000)
        self.last_leader_action = None
        self.last_follower_action = None

        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.leader_action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = self.observation_space.sample()  # Normal distribution
        self.state_history.clear()
        self.last_leader_action = None
        self.last_follower_action = None
        return np.copy(self.state)

    def step(self, action):
        """
        x_{t+1} = A x_t + B u_t + C a_t
            - x_t: state
            - u_t: follower action
            - a_t: leader action
            Note: 
                - a_t ~ N(K x, W * W^T), K and W are parameters of leader policy
                - w ~ N(0, W * W^T) is observable to follower and refered to as "a" in MaxEntLQR
        """
        leader_action = action[0]  # (leader_action_dim,)
        follower_action = action[1]  # (action_dim,)
        # For rendering
        self.last_leader_action = leader_action
        self.last_follower_action = follower_action
        self.state_history.append(self.state.copy())

        follower_reward = - self.cost(self.state, leader_action, follower_action)
        leader_reward = - self.leader_cost(self.state, leader_action, follower_action)
        done = False
        self.state = self.next_state(self.state, leader_action, follower_action)
        info = {'leader_action': leader_action, 'target_reward': leader_reward}
        return np.array(self.state, dtype=np.float32), follower_reward, done, info

    def next_state(self, x, a, u):
        return self.A @ x + self.B @ u + self.C @ a  # (obs_dim,)
    
    def cost(self, x, a, u):
        return (x.T @ self.Q @ x + u.T @ self.R @ u).item()  # scalar
    
    def leader_cost(self, x, a, u):
        """
        フォロワーコストに加えて、位置（x[0]）が0以下になるとペナルティを与える
        """
        return (self.cost(x, a, u) - self.cost_lambda * min(x[0], 0))  # scalar

    def render(self, mode='human'):
        try:
            from gym.envs.classic_control import rendering
        except ImportError:
            raise ImportError("Cannot import rendering module from gym. Make sure gym is installed.")
        
        screen_width = 600
        screen_height = 600
        state_range = 2.5  # 状態の表示範囲
        leader_arrow_scale = 100.0  # リーダー影響矢印のスケール調整
        follower_arrow_scale = 1.0  # フォロワー影響矢印のスケール調整
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            # 背景
            self.viewer.set_bounds(-state_range, state_range, -state_range, state_range)
            
            # グリッド線を描画
            for i in range(-int(state_range), int(state_range) + 1): # 1.0 刻み
                # 縦線
                line = rendering.Line((i, -state_range), (i, state_range))
                line.set_color(0.8, 0.8, 0.8)
                self.viewer.add_geom(line)
                # 横線
                line = rendering.Line((-state_range, i), (state_range, i))
                line.set_color(0.8, 0.8, 0.8)
                self.viewer.add_geom(line)
            
            # 原点
            origin = rendering.make_circle(0.1)
            origin.set_color(0, 0, 0)
            self.viewer.add_geom(origin)
            
            # 現在の状態を表す点
            self.state_point = rendering.make_circle(0.08)  # 半径を0.1に変更
            self.state_point.set_color(1, 0, 0)  # 赤色
            self.state_trans = rendering.Transform()
            self.state_point.add_attr(self.state_trans)
            self.viewer.add_geom(self.state_point)

            # フォロワー影響矢印（B @ follower_action）
            self.follower_effect_arrow = rendering.FilledPolygon([(-0.05, 0), (0.05, 0), (0, 0.3)])
            self.follower_effect_arrow.set_color(0, 0, 1)  # 青色
            self.follower_effect_trans = rendering.Transform()
            self.follower_effect_arrow.add_attr(self.follower_effect_trans)
            self.viewer.add_geom(self.follower_effect_arrow)

            # リーダー影響矢印（C @ leader_action）
            self.leader_effect_arrow = rendering.FilledPolygon([(-0.05, 0), (0.05, 0), (0, 0.3)])
            self.leader_effect_arrow.set_color(1, 0.5, 0)  # オレンジ色
            self.leader_effect_trans = rendering.Transform()
            self.leader_effect_arrow.add_attr(self.leader_effect_trans)
            self.viewer.add_geom(self.leader_effect_arrow)
            
            # 軌跡を表示するための線のリスト
            self.trajectory_lines = []
            
            try:
                # rendering.Textが使用可能な場合
                # X軸ラベル（s_1）
                x_label_text = rendering.Text("s\u2081", fontsize=20)
                x_label_text.set_color(0, 0, 0)  # 黒色
                x_label_trans = rendering.Transform(translation=(state_range - 0.3, -state_range + 0.2))
                x_label_text.add_attr(x_label_trans)
                self.viewer.add_geom(x_label_text)
                # Y軸ラベル（s_2）
                y_label_text = rendering.Text("s\u2082", fontsize=20)
                y_label_text.set_color(0, 0, 0)  # 黒色
                y_label_trans = rendering.Transform(translation=(-state_range + 0.2, state_range - 0.3))
                y_label_text.add_attr(y_label_trans)
                self.viewer.add_geom(y_label_text)
            except:
                pass
        
        if self.state is None:
            return None
        
        # 現在の状態を更新
        x, y = self.state[0], self.state[1] if len(self.state) > 1 else 0
        self.state_trans.set_translation(x, y)
        
        # フォロワー影響矢印を更新（B @ follower_action）
        if hasattr(self, 'last_follower_action') and self.last_follower_action is not None:
            follower_effect = self.B @ self.last_follower_action  # (2,) 2次元ベクトル
            effect_x, effect_y = follower_effect[0], follower_effect[1]
            
            # 矢印の長さと角度を計算
            effect_magnitude = np.sqrt(effect_x**2 + effect_y**2)
            # 矢印のスケール調整
            effect_magnitude *= follower_arrow_scale
            if effect_magnitude > 0.01:  # 微小な影響は表示しない    
                # 角度を計算（atan2で正しい象限を得る）
                angle = np.arctan2(effect_y, effect_x)
                
                # 矢印の頂点を更新
                self.follower_effect_arrow.v = [(-0.05, 0), (0.05, 0), (0, effect_magnitude)]
                self.follower_effect_trans.set_rotation(angle - np.pi/2)  # 上向きが0度になるよう調整
                self.follower_effect_trans.set_translation(x + 0.2, y)  # 現在位置から少しずらして表示
            else:
                # 影響が小さい場合は非表示
                self.follower_effect_trans.set_translation(x + 10, y + 10)  # 画面外に移動
        
        # リーダー影響矢印を更新（C @ leader_action）
        if hasattr(self, 'last_leader_action') and self.last_leader_action is not None:
            leader_effect = self.C @ self.last_leader_action  # (2,) 2次元ベクトル
            effect_x, effect_y = leader_effect[0], leader_effect[1]
            
            # 矢印の長さと角度を計算
            effect_magnitude = np.sqrt(effect_x**2 + effect_y**2)
            # 矢印のスケール調整
            effect_magnitude *= leader_arrow_scale
            if effect_magnitude > 0.01:  # 微小な影響は表示しない
                # 角度を計算
                angle = np.arctan2(effect_y, effect_x)
                
                # 矢印の頂点を更新
                self.leader_effect_arrow.v = [(-0.05, 0), (0.05, 0), (0, effect_magnitude)]
                self.leader_effect_trans.set_rotation(angle - np.pi/2)  # 上向きが0度になるよう調整
                self.leader_effect_trans.set_translation(x - 0.2, y)  # 現在位置から少しずらして表示
            else:
                # 影響が小さい場合は非表示
                self.leader_effect_trans.set_translation(x + 10, y + 10)  # 画面外に移動

        
        # 軌跡を描画（最新の100ポイント）
        if hasattr(self, 'state_history') and len(self.state_history) > 1:
            # 古い軌跡線を削除
            for line in self.trajectory_lines:
                self.viewer.geoms.remove(line)
            self.trajectory_lines = []
            
            # 新しい軌跡線を追加
            states = list(self.state_history)[-100:]  # 最新100ポイント
            for i in range(len(states) - 1):
                x1, y1 = states[i][0], states[i][1] if len(states[i]) > 1 else 0
                x2, y2 = states[i+1][0], states[i+1][1] if len(states[i+1]) > 1 else 0
                line = rendering.Line((x1, y1), (x2, y2))
                line.set_color(0.5, 0.5, 1.0)  # 薄い青色
                self.trajectory_lines.append(line)
                self.viewer.add_geom(line)

        result = self.viewer.render(return_rgb_array=mode == 'rgb_array')
        if mode == 'human':
            time.sleep(0.1)  # 0.1秒待機
            self.viewer.window.dispatch_events()

        return result

    def close(self):
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class LQREnv_1(LQREnv_0):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def leader_cost(self, x, a, u):
        """
        位置（x[0]）が0以下になるとペナルティを与えるリーダーコスト
        """
        return - self.cost_lambda * min(x[0], 0)
# 後方互換性のためのエイリアス
LQREnv = LQREnv_1
    

class LQREnv_2(LQREnv_0):
    def __init__(self, *args, s_2_abs_range=(0.0, 0.05), **kwargs):
        super().__init__(*args, **kwargs)
        if s_2_abs_range[0] < s_2_abs_range[1]:
            self.s_2_abs_range_lower = s_2_abs_range[0]
            self.s_2_abs_range_upper = s_2_abs_range[1]
        else:
            self.s_2_abs_range_lower = s_2_abs_range[1]
            self.s_2_abs_range_upper = s_2_abs_range[0]
    
    def leader_cost(self, x, a, u):
        """
        速度（x[1]）がある範囲内に収まると報酬が1、範囲外だと報酬が0になるようなリーダーコスト
        """
        if self.s_2_abs_range_lower <= abs(x[1]) <= self.s_2_abs_range_upper:
            c = - 1.0
        else:
            c = 0.0
        return self.cost_lambda * c

class LQREnv_3(LQREnv_0):
    def __init__(self, *args, initial_state=[1.0, 0.0], target_state=[-0.5, 0.0], **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_state = np.asarray(initial_state)
        self.target_state = np.asarray(target_state)

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = self.initial_state
        self.state_history.clear()
        self.last_leader_action = None
        self.last_follower_action = None
        return np.copy(self.state)
    
    def leader_cost(self, x, a, u):
        """
        Target stateとの距離が0.25より小さくなると報酬が1，それ以外は報酬が0になるようなリーダーコスト
        """
        if np.linalg.norm(x - self.target_state, ord=2) < 0.25:
            c = - 1.0
        else:
            c = 0.0
        return self.cost_lambda * c

class LQREnv_4(LQREnv_0):
    def __init__(self, 
                 *args, 
                 initial_state=[1.0, 0.0], 
                 target_state=[-0.5, 0.0], 
                 xstd=[0.1, 0.01],
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_state = np.asarray(initial_state)
        self.target_state = np.asarray(target_state)
        self.xstd = np.asarray(xstd)
        if not self.initial_state.shape == self.target_state.shape == self.xstd.shape == (self._obs_dim,):
            raise ValueError("initial_state, target_state, and xstd must have the same dimension as state space.")

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = self.initial_state
        self.state_history.clear()
        self.last_leader_action = None
        self.last_follower_action = None
        return np.copy(self.state)
    
    def leader_cost(self, x, a, u):
        """
        Target stateに近いほど報酬が高くなるようなリーダーコスト
        """
        r = np.exp(- np.sum((x - self.target_state)**2 / 2 / self.xstd**2))
        return - self.cost_lambda * r.item()