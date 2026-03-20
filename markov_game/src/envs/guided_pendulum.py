import gym
import time
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import importlib.util

class GuidedPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0, seed=None, max_leader_action=10.0, flip_leader_follower=False):
        self.max_speed = 8
        self.max_torque = 2.
        self.max_wind_speed = 1.0
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.wind_mag = max_leader_action
        self.viewer = None
        self.last_leader_u = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        if not flip_leader_follower:
            self.action_space = spaces.Box(
                low=-self.max_torque,
                high=self.max_torque, shape=(1,),
                dtype=np.float32
            )
            self.leader_action_space = spaces.Box(
                low=-self.max_wind_speed,
                high=self.max_wind_speed, shape=(1,),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=-self.max_wind_speed,
                high=self.max_wind_speed, shape=(1,),
                dtype=np.float32
            )
            self.leader_action_space = spaces.Box(
                low=-self.max_torque,
                high=self.max_torque, shape=(1,),
                dtype=np.float32
            )
        self.flip_leader_follower = flip_leader_follower

        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.leader_action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        if not self.flip_leader_follower:
            leader_u = u[0]
            u = u[1]
        else:
            leader_u = u[1]
            u = u[0]

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        w = self.wind_mag

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        # Leader action
        leader_u = np.clip(leader_u, -self.max_wind_speed, self.max_wind_speed)[0]
        thcas_l = np.cos(th) * l
        self.last_leader_u = leader_u  # for rendering
        wind_effect = (leader_u * w) * (thcas_l * 2)
        wind_torque = wind_effect * thcas_l
        self.last_wind_torque = wind_torque  # for rendering
        wind_thacc = wind_torque / (m * l ** 2)
        # Target reward
        minus_thsin = np.sin(th + np.pi)
        target_reward = np.clip(minus_thsin, -np.inf, 0)  # pendulumが左側にあるとき，負の報酬

        #newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newthdot = thdot + (-3 * g / (2 * l) * minus_thsin 
                            + 3. / (m * l ** 2) * u
                            + wind_thacc) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])

        if not self.flip_leader_follower:
            info = {'leader_action': leader_u, "target_reward": target_reward}
            return self._get_obs(), -costs, False, info
        else:
            info = {'leader_action': u, "target_reward": -costs}
            return self._get_obs(), target_reward, False, info

    def reset(self, initial_state=None):
        if initial_state is not None:
            assert self.observation_space.contains(initial_state)
            self.state = initial_state
        else:
            high = np.array([np.pi, 1])
            self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()
    
    def set_state(self, state):
        self.state = state

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            spec = importlib.util.find_spec('gym.envs.classic_control')
            classic_control_path = path.dirname(spec.origin)
            fname = path.join(classic_control_path, "assets", "clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)
            # wind visualization
            self.arrow = rendering.FilledPolygon([(0, 1.4), (0, 1.8), (0, 1.6)])
            self.arrowtrans = rendering.Transform()
            self.arrow.add_attr(self.arrowtrans)
            self.arrow.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(self.arrow)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        # 矢印の大きさを更新
        wind_velocity = self.last_leader_u if self.last_leader_u is not None else 0
        arrow_size = 1.0 * abs(wind_velocity)
        arrow_size = arrow_size if wind_velocity >= 0 else -arrow_size
        self.arrow.v = [(0, 1.4), (0, 1.8), (arrow_size, 1.6)]

        # Modify the frame rate
        time.sleep(1.0 / self.metadata['video.frames_per_second'])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
