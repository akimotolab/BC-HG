"""An environment wrapper that normalizes action, observation and reward."""
import akro
import numpy as np

from ._environment import EnvStep, Wrapper

class NormalizedEnv(Wrapper):
    """An environment wrapper for normalization.

    This wrapper normalizes action, and optionally observation and reward.

    Args:
        env (Environment): An environment instance.
        scale_reward (float): Scale of environment reward.
        normalize_obs (bool): If True, normalize observation.
        normalize_reward (bool): If True, normalize reward. scale_reward is
            applied after normalization.
        expected_action_scale (float): Assuming action falls in the range of
            [-expected_action_scale, expected_action_scale] when normalize it.
        flatten_obs (bool): Flatten observation if True.
        obs_alpha (float): Update rate of moving average when estimating the
            mean and variance of observations.
        reward_alpha (float): Update rate of moving average when estimating the
            mean and variance of rewards.

    """

    def __init__(
        self,
        env,
        scale_reward=1.,
        scale_target_reward=1.,
        normalize_obs=False,
        normalize_reward=False,
        expected_action_scale=1.,
        flatten_obs=True,
        obs_alpha=0.001,
        reward_alpha=0.001,
    ):
        super().__init__(env)

        self._scale_reward = [scale_reward, scale_target_reward]
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._expected_action_scale = expected_action_scale
        self._flatten_obs = flatten_obs

        self._obs_alpha = obs_alpha
        flat_obs_dim = self._env.observation_space.flat_dim
        self._obs_mean = np.zeros(flat_obs_dim)
        self._obs_var = np.ones(flat_obs_dim)

        self._reward_alpha = reward_alpha
        self._reward_mean = [0., 0.]
        self._reward_var = [1., 1.]

    def reset(self, initial_state=None):
        """Call reset on wrapped env.

        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)

        """
        first_obs, episode_info = self._env.reset(initial_state=initial_state)
        if self._normalize_obs:
            return self._apply_normalize_obs(first_obs), episode_info
        else:
            return first_obs, episode_info

    def step(self, action):
        """Call step on wrapped env.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            EnvStep: The environment step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment has been
                constructed and `reset()` has not been called.

        """
        action_spaces = [self.leader_action_space, self.action_space]
        scaled_action = [None, None]
        for i, (a, a_space) in enumerate(zip(action, action_spaces)):
            if isinstance(a_space, akro.Box):
                # rescale the action when the bounds are not inf
                lb, ub = a_space.low, a_space.high
                if np.all(lb != -np.inf) and np.all(ub != -np.inf):
                    scaled_action[i] = lb + (a + self._expected_action_scale) * (
                        0.5 * (ub - lb) / self._expected_action_scale)
                    scaled_action[i] = np.clip(scaled_action[i], lb, ub)
                else:
                    scaled_action[i] = a
            else:
                scaled_action[i] = a


        es = self._env.step(scaled_action)
        next_obs = es.observation
        reward = es.reward
        target_reward = es.env_info['target_reward']

        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)
            target_reward = self._apply_normalize_reward(target_reward, target=True)

        es.env_info['leader_action'] = action[0]
        es.env_info['target_reward'] = target_reward * self._scale_reward[0]

        return EnvStep(env_spec=es.env_spec,
                       action=action[1],
                       reward=reward * self._scale_reward[1],
                       observation=next_obs,
                       env_info=es.env_info,
                       step_type=es.step_type)

    def _update_obs_estimate(self, obs):
        flat_obs = self._env.observation_space.flatten(obs)
        self._obs_mean = (
            1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * flat_obs
        self._obs_var = (
            1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.square(
                flat_obs - self._obs_mean)

    def _update_reward_estimate(self, reward, target=False):
        i = 0 if target else 1
        self._reward_mean[i] = (1 - self._reward_alpha) * \
            self._reward_mean[i] + self._reward_alpha * reward
        self._reward_var[i] = (
            1 - self._reward_alpha
        ) * self._reward_var[i] + self._reward_alpha * np.square(
            reward - self._reward_mean[i])

    def _apply_normalize_obs(self, obs):
        """Compute normalized observation.

        Args:
            obs (np.ndarray): Observation.

        Returns:
            np.ndarray: Normalized observation.

        """
        self._update_obs_estimate(obs)
        flat_obs = self._env.observation_space.flatten(obs)
        normalized_obs = (flat_obs -
                          self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)
        if not self._flatten_obs:
            normalized_obs = self._env.observation_space.unflatten(
                self._env.observation_space, normalized_obs)
        return normalized_obs

    def _apply_normalize_reward(self, reward, target=False):
        """Compute normalized reward.

        Args:
            reward (float): Reward.

        Returns:
            float: Normalized reward.

        """
        i = 0 if target else 1
        self._update_reward_estimate(reward, target=target)
        return reward / (np.sqrt(self._reward_var[i]) + 1e-8)



normalize = NormalizedEnv
