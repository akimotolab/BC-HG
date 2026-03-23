"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
import time
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class GuidedCartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation               Min             Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                -24 deg         24 deg
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Box(1)
        Num	Action
        0	Push cart to the left or right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Leader Actions:
        Type: Box(1)
        Num	Action
        0	Blow wind to the pole from the left or right

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average reward is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, 
                 max_leader_action=1.0,
                 target_interval=[-0.25, 0.25],
                 flip_leader_follower=False):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Leader's action setting
        self.wind_mag = max_leader_action

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                         dtype=np.float32)
        
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)


        if not flip_leader_follower:
            # Note: The leader can blow wind with a fixed speed from the left or right, or stop the wind.
            # Note: The leader can vary wind speed within [1, 1-].
            #self.action_space = spaces.Discrete(2) (Original gym version)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)  # Continuous action space for SAC
            self.leader_action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)  
            self.leader_action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)  # Continuous action space for SAC
        
        self.target_interval = target_interval
        self.flip_leader_follower = flip_leader_follower

        self.seed()
        self.viewer = None
        self.state = None
        self.leader_action = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.leader_action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def step(self, action):

        if not self.flip_leader_follower:
            leader_action = action[0]
            action = action[1]
        else:
            leader_action = action[1]
            action = action[0]

        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        err_msg = "%r (%s) invalid" % (leader_action, type(leader_action))
        assert self.leader_action_space.contains(leader_action), err_msg

        x, x_dot, theta, theta_dot = self.state
        #force = self.force_mag if action == 1 else -self.force_mag
        force = self.force_mag * float(action)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Wind by leader's action
        #wind_force = leader_action * self.wind_speed
        wind_force = float(leader_action)
        wind_effect = wind_force * self.length * 2 * costheta * self.wind_mag
        torque = wind_effect * costheta * self.length
        wind_thetaacc = torque / (self.polemass_length * self.length)
        thetaacc += wind_thetaacc

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)
        self.leader_action = leader_action

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        # Define the target reward
        if self.target_interval[0] < x and x < self.target_interval[1]:
            target_reward = 1.0
        else:
            target_reward = 0.0

        if not self.flip_leader_follower:
            info = {'leader_action': leader_action, 'target_reward': target_reward}
            return np.array(self.state, dtype=np.float32), reward, done, info
        else:
            info = {'leader_action': action, 'target_reward': reward} 
            return np.array(self.state, dtype=np.float32), target_reward, done, info

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)
    
    def set_state(self, state):
        self.state = state

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)

            # Show the target region with lines
            track_interval_start = max(0, (self.target_interval[0] * scale + screen_width / 2.0))
            track_interval_end = min(screen_width, (self.target_interval[1] * scale + screen_width / 2.0))

            # First, draw black lines outside the target region
            # Left black line
            if track_interval_start > 0:
                left_track = rendering.Line((0, carty), (track_interval_start, carty))
                left_track.set_color(0, 0, 0)
                self.viewer.add_geom(left_track)

            # Right black line
            if track_interval_end < screen_width:
                right_track = rendering.Line((track_interval_end, carty), (screen_width, carty))
                right_track.set_color(0, 0, 0)
                self.viewer.add_geom(right_track)

            # Draw a red band in the target region
            band_height = 3  # Keep the band height small to make it look thin
            red_band = rendering.FilledPolygon([
                (track_interval_start, carty - band_height),
                (track_interval_start, carty + band_height),
                (track_interval_end, carty + band_height),
                (track_interval_end, carty - band_height)
            ])
            red_band.set_color(1, 0, 0)  # Red
            self.viewer.add_geom(red_band)

            # Draw boundary lines
            left_boundary = rendering.Line((track_interval_start, carty - 10), (track_interval_start, carty + 10))
            left_boundary.set_color(1, 0, 0)  # Red
            self.viewer.add_geom(left_boundary)

            right_boundary = rendering.Line((track_interval_end, carty - 10), (track_interval_end, carty + 10))
            right_boundary.set_color(1, 0, 0)  # Red
            self.viewer.add_geom(right_boundary)

            # wind visualization
            self.arrow = rendering.FilledPolygon([(-15, 0), (15, 0), (0, 50)])
            self.arrowtrans = rendering.Transform()
            self.arrow.add_attr(self.arrowtrans)
            self.arrow.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(self.arrow)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        # Edit the position of the wind arrow
        cartx = self.carttrans.translation[0]
        arrow_x = cartx
        arrow_y = carty + cartheight / 2 + polelen + 20
        self.arrowtrans.set_translation(arrow_x, arrow_y)

        # Update arrow size
        wind_velocity = self.leader_action[0] if self.leader_action is not None else 0
        arrow_size = 50 * abs(wind_velocity)
        self.arrow.v = [(-15, 0), (15, 0), (0, arrow_size)]

        # Edit the direction of the wind arrow
        if wind_velocity >= 0:
            self.arrowtrans.set_rotation(-math.pi/2)  # right
        else:
            self.arrowtrans.set_rotation(math.pi/2)  # left

        # Modify the frame rate
        time.sleep(1.0 / self.metadata['video.frames_per_second'])
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
