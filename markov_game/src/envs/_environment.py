"""Base PTIA Environment API."""

import abc
from dataclasses import dataclass
from typing import Dict, Union
from gym.spaces import flatten_space
import akro
import torch
import numpy as np

# Can't use naive garage import, or Sphinx AutoAPI breaks.
from garage._dtypes import StepType
from garage.torch import as_torch, global_device


@dataclass(frozen=True)
class InOutSpec:
    """Describes the input and output spaces of a primitive or module."""
    input_space: akro.Space
    output_space: akro.Space

@dataclass(frozen=True, init=False)
class EnvSpec(InOutSpec):
    """Describes the observations, actions, and time horizon of an MDP.

    Args:
        observation_space (akro.Space): The observation space of the env.
        action_space (akro.Space): The action space of the env.
        max_episode_length (int): The maximum number of steps allowed in an
            episode.

    """

    def __init__(self,
                 observation_space,
                 action_space,
                 max_episode_length=None):
        object.__setattr__(self, 'max_episode_length', max_episode_length)
        super().__init__(input_space=action_space,
                         output_space=observation_space)

    max_episode_length: Union[int, None] = None

    @property
    def action_space(self):
        """Get action space.

        Returns:
            akro.Space: Action space of the env.

        """
        return self.input_space

    @property
    def observation_space(self):
        """Get state space of the env. 
        Note: This state is shared by both leader and follower. 
        The obersevation space for each policy and q-function of agents can be defferent.

        Returns:
            akro.Space: Observation space.

        """
        return self.output_space

    @action_space.setter
    def action_space(self, action_space):
        """Set action space of the env.

        Args:
            action_space (akro.Space): Action space.

        """
        self._input_space = action_space

    @observation_space.setter
    def observation_space(self, observation_space):
        """Set observation space of the env.

        Args:
            observation_space (akro.Space): Observation space.

        """
        self._output_space = observation_space


@dataclass(frozen=True, init=False)
class GlobalEnvSpec:
    _observation_space: akro.Space     # env's original observation space
    _action_space: akro.Space          # env's original action space
    _leader_action_space: akro.Space   # env's original leader action space
    _leader_policy_env_spec: EnvSpec   # env spec for leader policy
    _follower_policy_env_spec: EnvSpec # env spec for follower policy
    _leader_qf_env_spec: EnvSpec       # env spec for leader q-function
    _follower_qf_env_spec: EnvSpec     # env spec for follower q-function
    _leader_policy_obs_info: list
    _follower_policy_obs_info: list
    _leader_qf_obs_info: list
    _follower_qf_obs_info: list

    def __init__(self,
                 observation_space,
                 action_space,
                 leader_action_space,
                 max_episode_length=None):
        object.__setattr__(self, '_observation_space', observation_space)
        object.__setattr__(self, '_action_space', action_space)
        object.__setattr__(self, '_leader_action_space', leader_action_space)
        object.__setattr__(self, 'max_episode_length', max_episode_length)

        self.set_env_specs_for_agents()

    max_episode_length: Union[int, None] = None

    def set_env_specs_for_agents(self):
        """Get environment specifications for leader and follower agents.
        This method creates and returns the environment specifications for both
        the leader and follower agents. The leader's environment specification
        is created using the leader's action space and the shared observation
        space. The follower's environment specification is created by concatenating
        the flattened observation space and the flattened leader action space.
        Returns:
            tuple: A tuple containing two EnvSpec objects:
                - leader_es: The environment specification for the leader agent.
                - follower_es: The environment specification for the follower agent. 
                - obs_info: A dictionary containing information about the observations       
        """
        flatdim_o_space = akro.from_gym(flatten_space(self.observation_space))
        flatdim_la_space = akro.from_gym(flatten_space(self.leader_action_space))
        flatdim_fa_space = akro.from_gym(flatten_space(self.action_space))
        
        # For policies
        l_policy_es = EnvSpec(observation_space=self.observation_space,
                              action_space=self.leader_action_space,
                              max_episode_length=self.max_episode_length)
        object.__setattr__(self, '_leader_policy_env_spec', l_policy_es)
        object.__setattr__(self, '_leader_policy_obs_info', ['observation'])

        f_policy_obs_space = akro.concat(flatdim_o_space, flatdim_la_space)
        f_policy_es = EnvSpec(observation_space=f_policy_obs_space,
                              action_space=self.action_space,
                              max_episode_length=self.max_episode_length)
        object.__setattr__(self, '_follower_policy_env_spec', f_policy_es)
        object.__setattr__(self, '_follower_policy_obs_info', ['observation', 'leader_action'])

        # For Q-functions
        l_qf_obs_space = akro.concat(flatdim_o_space, flatdim_fa_space)        
        l_qf_es = EnvSpec(observation_space=l_qf_obs_space,
                          action_space=self.leader_action_space,
                          max_episode_length=self.max_episode_length)
        object.__setattr__(self, '_leader_qf_env_spec', l_qf_es)
        object.__setattr__(self, '_leader_qf_obs_info', ['observation', 'follower_action'])

        f_qf_es = EnvSpec(observation_space=f_policy_obs_space,
                          action_space=self.action_space,
                          max_episode_length=self.max_episode_length)
        object.__setattr__(self, '_follower_qf_env_spec', f_qf_es)
        object.__setattr__(self, '_follower_qf_obs_info', ['observation', 'leader_action'])

    def get_input_for(self, agent, module, 
                      obs=None, leader_act=None, follower_act=None, obs_info=None):
        
        return self.get_inputs_for(agent, module, [obs], [leader_act], [follower_act], obs_info)[0]
    
    def get_inputs_for(self, agent=None, module=None, 
                       obs=None, leader_act=None, follower_act=None, obs_info=None):
        assert agent in ['leader', 'follower'] and module in ['policy', 'qf']
        if obs_info is None:
            if module == 'policy':
                obs_info = self.leader_policy_obs_info if agent == 'leader' else self.follower_policy_obs_info
            elif module == 'qf':
                obs_info = self.leader_qf_obs_info if agent == 'leader' else self.follower_qf_obs_info

        inputs = dict(observation=[obs, self.observation_space], 
                      leader_action=[leader_act, self.leader_action_space],
                      follower_action=[follower_act, self.action_space])
        
        flatten_tensors = []
        for k in obs_info:
            if k not in inputs:
                raise KeyError(f"Key {k} not found in inputs")
            v = inputs[k][0]
                
            if isinstance(v, torch.Tensor):
                v_tensor = v.float()
                v_tensor = v_tensor.view(v_tensor.shape[0], -1).to(global_device())
            elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                v_tensor = [v_i.view(v_i.shape[0], -1) for v_i in v]
                v_tensor = torch.cat(v_tensor, dim=0).to(global_device())
            elif isinstance(v, list) and isinstance(v[0], np.ndarray):
                v_tensor = as_torch(inputs[k][1].flatten_n(np.asarray(v)))
            else: # If v is np.ndarray or int, which is expected to be observations from the env with its original shape
                v_tensor = as_torch(inputs[k][1].flatten_n(v))

            flatten_tensors.append(v_tensor)
        return torch.cat(flatten_tensors, axis=1)

    @property
    def action_space(self):
        """Get action space.

        Returns:
            akro.Space: Action space of the env.

        """
        return self._action_space
    
    @property
    def leader_action_space(self):
        """Get leader's action space.

        Returns:
            akro.Space: Action space of the env.

        """
        return self._leader_action_space

    @property
    def observation_space(self):
        """Get state space of the env. 
        Note: This state is shared by both leader and follower. 
        The obersevation space for each policy and q-function of agents can be defferent.

        Returns:
            akro.Space: Observation space.

        """
        return self._observation_space

    @property
    def leader_policy_env_spec(self):
        """Get leader's policy environment specification.

        Returns:
            EnvSpec: Environment specification for the leader agent.

        """
        return self._leader_policy_env_spec
    
    @property
    def follower_policy_env_spec(self):
        """Get follower's policy environment specification.

        Returns:
            EnvSpec: Environment specification for the follower agent.

        """
        return self._follower_policy_env_spec
    
    @property
    def leader_qf_env_spec(self):
        """Get leader's Q-function environment specification.

        Returns:
            EnvSpec: Environment specification for the leader agent.

        """
        return self._leader_qf_env_spec
    
    @property
    def follower_qf_env_spec(self):
        """Get follower's Q-function environment specification.

        Returns:
            EnvSpec: Environment specification for the follower agent.

        """
        return self._follower_qf_env_spec
    
    @property
    def leader_policy_obs_info(self):
        """Get leader's policy observation information.

        Returns:
            list: A list containing information about the observations.

        """
        return self._leader_policy_obs_info
    
    @property
    def follower_policy_obs_info(self):
        """Get follower's policy observation information.

        Returns:
            list: A list containing information about the observations.

        """
        return self._follower_policy_obs_info
    
    @property
    def leader_qf_obs_info(self):
        """Get leader's Q-function observation information.

        Returns:
            list: A list containing information about the observations.

        """
        return self._leader_qf_obs_info
    
    @property
    def follower_qf_obs_info(self):
        """Get follower's Q-function observation information.

        Returns:
            list: A list containing information about the observations.

        """
        return self._follower_qf_obs_info

    @action_space.setter
    def action_space(self, action_space):
        """Set action space of the env.

        Args:
            action_space (akro.Space): Action space.

        """
        self._action_space = action_space

    @leader_action_space.setter
    def leader_action_space(self, action_space):
        """Set leader's action space of the env.

        Args:
            action_space (akro.Space): Action space.

        """
        self._leader_action_space = action_space

    @observation_space.setter
    def observation_space(self, observation_space):
        """Set observation space of the env.

        Args:
            observation_space (akro.Space): Observation space.

        """
        self._observation_space = observation_space

    @leader_policy_env_spec.setter
    def leader_policy_env_spec(self, env_spec):
        """Set leader's policy environment specification.

        Args:
            env_spec (EnvSpec): Environment specification for the leader agent.

        """
        self._leader_policy_env_spec = env_spec

    @follower_policy_env_spec.setter
    def follower_policy_env_spec(self, env_spec):
        """Set follower's policy environment specification.

        Args:
            env_spec (EnvSpec): Environment specification for the follower agent.

        """
        self._follower_policy_env_spec = env_spec
    
    @leader_qf_env_spec.setter
    def leader_qf_env_spec(self, env_spec):
        """Set leader's Q-function environment specification.

        Args:
            env_spec (EnvSpec): Environment specification for the leader agent.

        """
        self._leader_qf_env_spec = env_spec
    
    @follower_qf_env_spec.setter
    def follower_qf_env_spec(self, env_spec):
        """Set follower's Q-function environment specification.

        Args:
            env_spec (EnvSpec): Environment specification for the follower agent.

        """
        self._follower_qf_env_spec = env_spec

    @leader_policy_obs_info.setter
    def leader_policy_obs_info(self, obs_info):
        """Set leader's policy observation information.

        Args:
            obs_info (list): A list containing information about the observations.

        """
        self._leader_policy_obs_info = obs_info
    
    @follower_policy_obs_info.setter
    def follower_policy_obs_info(self, obs_info):
        """Set follower's policy observation information.

        Args:
            obs_info (list): A list containing information about the observations.

        """
        self._follower_policy_obs_info = obs_info

    @leader_qf_obs_info.setter
    def leader_qf_obs_info(self, obs_info):
        """Set leader's Q-function observation information.

        Args:
            obs_info (list): A list containing information about the observations.

        """
        self._leader_qf_obs_info = obs_info

    @follower_qf_obs_info.setter
    def follower_qf_obs_info(self, obs_info):
        """Set follower's Q-function observation information.

        Args:
            obs_info (list): A list containing information about the observations.

        """
        self._follower_qf_obs_info = obs_info
    
    
@dataclass
class EnvStep:
    # pylint: disable=missing-return-doc, missing-return-type-doc, missing-param-doc, missing-type-doc  # noqa: E501
    r"""A tuple representing a single step returned by the environment.

    Attributes:
        env_spec (EnvSpec): Specification for the environment from
            which this data was sampled.
        action (numpy.ndarray): A numpy array of shape :math:`(A^*)`
            containing the action for the this time step. These must conform
            to :obj:`EnvStep.action_space`.
            `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
            sequence.
        reward (float): A float representing the reward for taking the action
            given the observation, at the this time step.
            `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
            sequence.
        observation (numpy.ndarray): A numpy array of shape :math:`(O^*)`
            containing the observation for the this time step in the
            environment. These must conform to
            :obj:`EnvStep.observation_space`.
            The observation after applying the action.
        env_info (dict): A dict containing environment state information.
        step_type (StepType): a `StepType` enum value. Can either be
            StepType.FIRST, StepType.MID, StepType.TERMINAL, StepType.TIMEOUT.

    """

    env_spec: EnvSpec
    action: np.ndarray
    reward: float
    observation: np.ndarray
    env_info: Dict[str, Union[np.ndarray, dict]]
    step_type: StepType

    @property
    def first(self):
        """bool: Whether this `TimeStep` is the first of a sequence."""
        return self.step_type is StepType.FIRST

    @property
    def mid(self):
        """bool: Whether this `TimeStep` is in the mid of a sequence."""
        return self.step_type is StepType.MID

    @property
    def terminal(self):
        """bool: Whether this `TimeStep` records a termination condition."""
        return self.step_type is StepType.TERMINAL

    @property
    def timeout(self):
        """bool: Whether this `TimeStep` records a time out condition."""
        return self.step_type is StepType.TIMEOUT

    @property
    def last(self):
        """bool: Whether this `TimeStep` is the last of a sequence."""
        return self.step_type is StepType.TERMINAL or self.step_type \
            is StepType.TIMEOUT


class Environment(abc.ABC):
    """The main API for garage environments.

    The public API methods are:

    +-----------------------+
    | Functions             |
    +=======================+
    | reset()               |
    +-----------------------+
    | step()                |
    +-----------------------+
    | render()              |
    +-----------------------+
    | visualize()           |
    +-----------------------+
    | close()               |
    +-----------------------+

    Set the following properties:

    +-----------------------+-------------------------------------------------+
    | Properties            | Description                                     |
    +=======================+=================================================+
    | action_space          | The action space specification                  |
    +-----------------------+-------------------------------------------------+
    | observation_space     | The observation space specification             |
    +-----------------------+-------------------------------------------------+
    | spec                  | The environment specifications                  |
    +-----------------------+-------------------------------------------------+
    | render_modes          | The list of supported render modes              |
    +-----------------------+-------------------------------------------------+

    Example of a simple rollout loop:

    .. code-block:: python

        env = MyEnv()
        policy = MyPolicy()
        first_observation, episode_info = env.reset()
        env.visualize()  # visualization window opened

        episode = []
        # Determine the first action
        first_action = policy.get_action(first_observation, episode_info)
        episode.append(env.step(first_action))

        while not episode[-1].last():
           action = policy.get_action(episode[-1].observation)
           episode.append(env.step(action))

        env.close()  # visualization window closed

    Make sure your environment is pickle-able:
        Garage pickles the environment via the `cloudpickle` module
        to save snapshots of the experiment. However, some environments may
        contain attributes that are not pickle-able (e.g. a client-server
        connection). In such cases, override `__setstate__()` and
        `__getstate__()` to add your custom pickle logic.

        You might want to refer to the EzPickle module:
        https://github.com/openai/gym/blob/master/gym/utils/ezpickle.py
        for a lightweight way of pickle and unpickle via constructor
        arguments.

    """

    @property
    @abc.abstractmethod
    def action_space(self):
        """akro.Space: The action space specification."""

    @property
    @abc.abstractmethod
    def leader_action_space(self):
        """akro.Space: The action space specification."""

    @property
    @abc.abstractmethod
    def observation_space(self):
        """akro.Space: The observation space specification."""

    @property
    @abc.abstractmethod
    def spec(self):
        """EnvSpec: The environment specification."""

    @property
    @abc.abstractmethod
    def render_modes(self):
        """list: A list of string representing the supported render modes.

        See render() for a list of modes.
        """

    @abc.abstractmethod
    def reset(self, init_state):
        """Resets the environment.

        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)

        """

    @abc.abstractmethod
    def step(self, leader_action, action):
        """Steps the environment with the action and returns a `EnvStep`.

        If the environment returned the last `EnvStep` of a sequence (either
        of type TERMINAL or TIMEOUT) at the previous step, this call to
        `step()` will start a new sequence and `action` will be ignored.

        If `spec.max_episode_length` is reached after applying the action
        and the environment has not terminated the episode, `step()` should
        return a `EnvStep` with `step_type==StepType.TIMEOUT`.

        If possible, update the visualization display as well.

        Args:
            action (object): A NumPy array, or a nested dict, list or tuple
                of arrays conforming to `action_space`.

        Returns:
            EnvStep: The environment step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment has been
                constructed and `reset()` has not been called.

        """

    @abc.abstractmethod
    def render(self, mode):
        """Renders the environment.

        The set of supported modes varies per environment. By convention,
        if mode is:

        * rgb_array: Return an `numpy.ndarray` with shape (x, y, 3) and type
            uint8, representing RGB values for an x-by-y pixel image, suitable
            for turning into a video.
        * ansi: Return a string (str) or `StringIO.StringIO` containing a
            terminal-style text representation. The text can include newlines
            and ANSI escape sequences (e.g. for colors).

        Make sure that your class's `render_modes` includes the list of
        supported modes.

        For example:

        .. code-block:: python

            class MyEnv(Environment):
                def render_modes(self):
                    return ['rgb_array', 'ansi']

                def render(self, mode):
                    if mode == 'rgb_array':
                        return np.array(...)  # return RGB frame for video
                    elif mode == 'ansi':
                        ...  # return text output
                    else:
                        raise ValueError('Supported render modes are {}, but '
                                         'got render mode {} instead.'.format(
                                             self.render_modes, mode))

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.

        """

    @abc.abstractmethod
    def visualize(self):
        """Creates a visualization of the environment.

        This function should be called **only once** after `reset()` to set up
        the visualization display. The visualization should be updated
        when the environment is changed (i.e. when `step()` is called.)

        Calling `close()` will deallocate any resources and close any
        windows created by `visualize()`. If `close()` is not explicitly
        called, the visualization will be closed when the environment is
        destructed (i.e. garbage collected).

        """

    @abc.abstractmethod
    def close(self):
        """Closes the environment.

        This method should close all windows invoked by `visualize()`.

        Override this function in your subclass to perform any necessary
        cleanup.

        Environments will automatically `close()` themselves when they are
        garbage collected or when the program exits.
        """

    def _validate_render_mode(self, mode):
        if mode not in self.render_modes:
            raise ValueError('Supported render modes are {}, but '
                             'got render mode {} instead.'.format(
                                 self.render_modes, mode))

    def __del__(self):
        """Environment destructor."""
        self.close()


class Wrapper(Environment):
    """A wrapper for an environment that implements the `Environment` API."""

    def __init__(self, env):
        """Initializes the wrapper instance.

        Args:
            env (Environment): The environment to wrap
        """
        self._env = env

    def __getattr__(self, name):
        """Forward getattr request to wrapped environment.

        Args:
            name (str): attr (str): attribute name

        Returns:
             object: the wrapped attribute.

        Raises:
            AttributeError: if the requested attribute is a private attribute,
            or if the requested attribute is not found in the
            wrapped environment.

        """
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name))
        if not hasattr(self._env, name):
            raise AttributeError('Attribute {} is not found'.format(name))
        return getattr(self._env, name)

    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._env.action_space
    
    @property
    def leader_action_space(self):
        """akro.Space: The action space specification."""
        return self._env.leader_action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._env.observation_space

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._env.spec

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return self._env.render_modes

    def step(self, action):
        """Step the wrapped env.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            EnvStep: The environment step resulting from the action.

        """
        return self._env.step(action)

    def reset(self, init_state=None):
        """Reset the wrapped env.

        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)

        """
        return self._env.reset(init_state)

    def render(self, mode):
        """Render the wrapped environment.

        Args:
            mode (str): the mode to render with. The string must be
                present in `self.render_modes`.

        Returns:
            object: the return value for render, depending on each env.

        """
        return self._env.render(mode)

    def visualize(self, mode='human'):
        """Creates a visualization of the wrapped environment."""
        self._env.visualize(mode)

    def close(self):
        """Close the wrapped env."""
        self._env.close()

    @property
    def unwrapped(self):
        """garage.Environment: The inner environment."""
        return getattr(self._env, 'unwrapped', self._env)
