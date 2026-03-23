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
        In addition to follower cost, apply a penalty when position (x[0]) is <= 0.
        """
        return (self.cost(x, a, u) - self.cost_lambda * min(x[0], 0))  # scalar

    def render(self, mode='human'):
        try:
            from gym.envs.classic_control import rendering
        except ImportError:
            raise ImportError("Cannot import rendering module from gym. Make sure gym is installed.")
        
        screen_width = 600
        screen_height = 600
        state_range = 2.5  # Display range of states
        leader_arrow_scale = 100.0  # Scale factor for leader-effect arrow
        follower_arrow_scale = 1.0  # Scale factor for follower-effect arrow
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            # Background
            self.viewer.set_bounds(-state_range, state_range, -state_range, state_range)
            
            # Draw grid lines
            for i in range(-int(state_range), int(state_range) + 1): # step of 1.0
                # Vertical line
                line = rendering.Line((i, -state_range), (i, state_range))
                line.set_color(0.8, 0.8, 0.8)
                self.viewer.add_geom(line)
                # Horizontal line
                line = rendering.Line((-state_range, i), (state_range, i))
                line.set_color(0.8, 0.8, 0.8)
                self.viewer.add_geom(line)
            
            # Origin
            origin = rendering.make_circle(0.1)
            origin.set_color(0, 0, 0)
            self.viewer.add_geom(origin)
            
            # Point representing the current state
            self.state_point = rendering.make_circle(0.08)  # Changed radius to 0.1
            self.state_point.set_color(1, 0, 0)  # Red
            self.state_trans = rendering.Transform()
            self.state_point.add_attr(self.state_trans)
            self.viewer.add_geom(self.state_point)

            # Follower-effect arrow (B @ follower_action)
            self.follower_effect_arrow = rendering.FilledPolygon([(-0.05, 0), (0.05, 0), (0, 0.3)])
            self.follower_effect_arrow.set_color(0, 0, 1)  # Blue
            self.follower_effect_trans = rendering.Transform()
            self.follower_effect_arrow.add_attr(self.follower_effect_trans)
            self.viewer.add_geom(self.follower_effect_arrow)

            # Leader-effect arrow (C @ leader_action)
            self.leader_effect_arrow = rendering.FilledPolygon([(-0.05, 0), (0.05, 0), (0, 0.3)])
            self.leader_effect_arrow.set_color(1, 0.5, 0)  # Orange
            self.leader_effect_trans = rendering.Transform()
            self.leader_effect_arrow.add_attr(self.leader_effect_trans)
            self.viewer.add_geom(self.leader_effect_arrow)
            
            # List of lines for displaying the trajectory
            self.trajectory_lines = []
            
            try:
                # If rendering.Text is available
                # X-axis label (s_1)
                x_label_text = rendering.Text("s\u2081", fontsize=20)
                x_label_text.set_color(0, 0, 0)  # Black
                x_label_trans = rendering.Transform(translation=(state_range - 0.3, -state_range + 0.2))
                x_label_text.add_attr(x_label_trans)
                self.viewer.add_geom(x_label_text)
                # Y-axis label (s_2)
                y_label_text = rendering.Text("s\u2082", fontsize=20)
                y_label_text.set_color(0, 0, 0)  # Black
                y_label_trans = rendering.Transform(translation=(-state_range + 0.2, state_range - 0.3))
                y_label_text.add_attr(y_label_trans)
                self.viewer.add_geom(y_label_text)
            except:
                pass
        
        if self.state is None:
            return None
        
        # Update the current state
        x, y = self.state[0], self.state[1] if len(self.state) > 1 else 0
        self.state_trans.set_translation(x, y)
        
        # Update the follower-effect arrow (B @ follower_action)
        if hasattr(self, 'last_follower_action') and self.last_follower_action is not None:
            follower_effect = self.B @ self.last_follower_action  # (2,) 2D vector
            effect_x, effect_y = follower_effect[0], follower_effect[1]
            
            # Compute arrow length and angle
            effect_magnitude = np.sqrt(effect_x**2 + effect_y**2)
            # Adjust arrow scaling
            effect_magnitude *= follower_arrow_scale
            if effect_magnitude > 0.01:  # Do not show negligible effects
                # Compute angle (atan2 gives the correct quadrant)
                angle = np.arctan2(effect_y, effect_x)
                
                # Update arrow vertices
                self.follower_effect_arrow.v = [(-0.05, 0), (0.05, 0), (0, effect_magnitude)]
                self.follower_effect_trans.set_rotation(angle - np.pi/2)  # Adjust so upward is 0 degrees
                self.follower_effect_trans.set_translation(x + 0.2, y)  # Display with a slight offset from current position
            else:
                # Hide when the effect is small
                self.follower_effect_trans.set_translation(x + 10, y + 10)  # Move out of screen
        
        # Update the leader-effect arrow (C @ leader_action)
        if hasattr(self, 'last_leader_action') and self.last_leader_action is not None:
            leader_effect = self.C @ self.last_leader_action  # (2,) 2D vector
            effect_x, effect_y = leader_effect[0], leader_effect[1]
            
            # Compute arrow length and angle
            effect_magnitude = np.sqrt(effect_x**2 + effect_y**2)
            # Adjust arrow scaling
            effect_magnitude *= leader_arrow_scale
            if effect_magnitude > 0.01:  # Do not show negligible effects
                # Compute angle
                angle = np.arctan2(effect_y, effect_x)
                
                # Update arrow vertices
                self.leader_effect_arrow.v = [(-0.05, 0), (0.05, 0), (0, effect_magnitude)]
                self.leader_effect_trans.set_rotation(angle - np.pi/2)  # Adjust so upward is 0 degrees
                self.leader_effect_trans.set_translation(x - 0.2, y)  # Display with a slight offset from current position
            else:
                # Hide when the effect is small
                self.leader_effect_trans.set_translation(x + 10, y + 10)  # Move out of screen

        
        # Draw trajectory (latest 100 points)
        if hasattr(self, 'state_history') and len(self.state_history) > 1:
            # Remove old trajectory lines
            for line in self.trajectory_lines:
                self.viewer.geoms.remove(line)
            self.trajectory_lines = []
            
            # Add new trajectory lines
            states = list(self.state_history)[-100:]  # Latest 100 points
            for i in range(len(states) - 1):
                x1, y1 = states[i][0], states[i][1] if len(states[i]) > 1 else 0
                x2, y2 = states[i+1][0], states[i+1][1] if len(states[i+1]) > 1 else 0
                line = rendering.Line((x1, y1), (x2, y2))
                line.set_color(0.5, 0.5, 1.0)  # Light blue
                self.trajectory_lines.append(line)
                self.viewer.add_geom(line)

        result = self.viewer.render(return_rgb_array=mode == 'rgb_array')
        if mode == 'human':
            time.sleep(0.1)  # Wait for 0.1 seconds
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
        Leader cost that applies a penalty when position (x[0]) is <= 0.
        """
        return - self.cost_lambda * min(x[0], 0)
# Alias for backward compatibility
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
        Leader cost where reward is 1 when speed (x[1]) is within a range, otherwise 0.
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
        Leader cost where reward is 1 if distance to target state is < 0.25, otherwise 0.
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
        Leader cost where reward increases as the state gets closer to the target state.
        """
        r = np.exp(- np.sum((x - self.target_state)**2 / 2 / self.xstd**2))
        return - self.cost_lambda * r.item()