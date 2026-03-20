"""This modules creates a MADDPG model in PyTorch."""
# yapf: disable
import abc
import copy
import numpy as np
import torch
from garage import make_optimizer, _Default
from garage.np.algos import RLAlgorithm
from garage.torch import as_torch_dict, torch_to_np, global_device
from garage.torch.policies.stochastic_policy import StochasticPolicy
from garage.experiment.deterministic import get_seed

# For type hints (not necessary)
from ..experiment import Trainer
from ..policies import JointPolicy

# yapf: enable


class AsyncMARL(RLAlgorithm):
    """
    Base class for Asynchronous Multi-Agent Reinforcement Learning (AsyncMARL) algorithms.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Leader policy network.
        qf (torch.nn.Module): Leader Q-function network.
        replay_buffer (ReplayBuffer): Replay buffer for storing experiences.
        init_steps (int, optional): Number of initial steps before training starts.
        update_after (int, optional): Number of steps to wait before starting updates.
        buffer_batch_size (int, optional): Batch size for sampling from replay buffer.
        min_buffer_size (int, optional): Minimum size of replay buffer before training.
        update_interval (int, optional): Interval (in follower updates) between leader updates.
        exploration_policy (garage.np.exploration_policies.ExplorationPolicy, optional): Exploration strategy.
        target_update_tau (float, optional): Interpolation parameter for soft target updates.
        discount (float, optional): Discount factor for cumulative return.
        policy_weight_decay (float, optional): L2 weight decay for policy network parameters.
        qf_weight_decay (float, optional): L2 weight decay for Q-function network parameters.
        clip_pos_returns (bool, optional): Whether to clip positive returns (not used).
        clip_return (float, optional): Clip return to be within [-clip_return, clip_return] (not used).
        policy_optimizer (Union[type, tuple[type, dict]], optional): Optimizer for policy network.
        qf_optimizer (Union[type, tuple[type, dict]], optional): Optimizer for Q-function network.
        policy_lr (float, optional): Learning rate for policy network.
        qf_lr (float, optional): Learning rate for Q-function network.
        reward_scale (float, optional): Scale factor for rewards.
        wb_follower (bool, optional): Whether the follower is a white-box model (required True).
        fixed_policy (bool, optional): Whether the leader's policy is fixed.
        hat_f_policy (object, optional): Follower's policy estimator (used if wb_follower).
        hat_f_qf (object, optional): Follower's Q-function estimator (used if wb_follower).
        hat_f_vf (object, optional): Follower's value function estimator (used if wb_follower).
        hat_f_p_optimizer (Union[type, tuple[type, dict]], optional): Optimizer for follower's policy estimator.
        hat_f_qf_optimizer (Union[type, tuple[type, dict]], optional): Optimizer for follower's Q-function estimator.
        hat_f_p_lr (float, optional): Learning rate for follower's policy estimator.
        hat_f_qf_lr (float, optional): Learning rate for follower's Q-function estimator.
        name (str, optional): Name of the algorithm.

    Methods:
        train(trainer): Start the training process for each epoch.
        train_once(trainer): Perform one iteration of leader training (abstract).
        log_statistics(trainer, prefix='Leader'): Log training statistics (abstract).
        to(device): Move all networks to the specified device.
        networks: List of networks in the model (abstract property).
    """
    name = 'AsyncMARL'

    def __init__(
            self,
            env_spec,
            policy,
            qf,
            replay_buffer,
            *,
            init_steps=0,
            update_after=0,
            buffer_batch_size=64,
            min_buffer_size=None,
            update_interval=1,
            exploration_policy=None,
            target_update_tau=0.01,
            discount=0.99,
            policy_weight_decay=0,
            qf_weight_decay=0,
            clip_pos_returns=False,
            clip_return=np.inf,
            max_policy_grad_norm=None,
            policy_optimizer=torch.optim.Adam,
            qf_optimizer=torch.optim.Adam,
            policy_lr=_Default(1e-4),
            qf_lr=_Default(1e-3),
            reward_scale=1.,
            wb_follower=True,
            fixed_policy=False,
            hat_f_policy = None,
            hat_f_qf = None,
            hat_f_vf = None,
            hat_f_p_optimizer=torch.optim.Adam,
            hat_f_qf_optimizer=torch.optim.Adam,
            hat_f_p_lr=_Default(1e-4),
            hat_f_qf_lr=_Default(1e-3),
            ):

        # Hyperparameters
        self._init_steps = init_steps  # Steps until which uniformly random leader actions are selected
        self._update_after = update_after  # Steps after which the leader starts updating
        self._buffer_batch_size = buffer_batch_size
        self._min_buffer_size = min_buffer_size if min_buffer_size is not None else buffer_batch_size
        self._update_interval = update_interval
        self._tau = target_update_tau
        self._policy_weight_decay = policy_weight_decay  # Not yet used
        self._qf_weight_decay = qf_weight_decay  # Not yet used
        self._clip_pos_returns = clip_pos_returns  # Not yet used
        self._clip_return = clip_return  # Not yet used
        self._max_policy_grad_norm = max_policy_grad_norm
        self._reward_scale = reward_scale

        self._max_episode_length = env_spec.max_episode_length

        # Experiment parameters
        self._wb_follower = wb_follower  # True: follower is white-box

        # For logging
        self._episode_rewards = []

        self.env_spec = env_spec
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.discount = discount
        self.exploration_policy = exploration_policy
        self.fixed_policy = fixed_policy
        self.is_deterministic_policy = not isinstance(self.policy, StochasticPolicy)

        self._policy = policy
        self._qf = qf
        self._target_policy = copy.deepcopy(self.policy)
        self._target_qf = copy.deepcopy(self._qf)
        self._policy_optimizer = make_optimizer(policy_optimizer,
                                                module=self.policy,
                                                lr=policy_lr)
        self._qf_optimizer = make_optimizer(qf_optimizer,
                                            module=self._qf,
                                            lr=qf_lr)
        self._p_lr = policy_lr
        self._qf_lr = qf_lr
        
        if not self._wb_follower:
            raise NotImplementedError(
                'AsyncMARL algorithm requires a white-box follower. '
                'Set wb_follower=True to use this algorithm.')
        else:
            self._hat_f_vf = None
            self._hat_f_qf = None
            self._hat_f_policy = None

    def train(self, trainer: Trainer):
        """Obtain samples and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer.

        Returns:
            float: The average return in last epoch cycle.

        """
        if self._wb_follower:
            self._hat_f_policy = trainer.follower.policy
            if self.name in ['Baseline', 'Baseline_STDPG']:
                self._hat_f_qf = None
            elif self.name in ['BCHG']:  # deterministic leader policy
                self._hat_f_qf = trainer.follower.make_q_function()
            elif self.name in ['STDPG', 'STDPGDiscrete']:  # stochastic leader policy
                self._hat_f_vf = trainer.follower.make_value_function()
            elif self.name in ['STMADDPG']:
                self._hat_f_qf = trainer.follower.make_q_function()
                self._hat_f_p_optimizer = trainer.follower.policy_optimizer
            else:
                raise NotImplementedError(f'AsyncMARL algorithm {self.name} is not supported.')
        
        self._last_follower_policy = copy.deepcopy(self._hat_f_policy)      
        self.f_discount = None  
        
        trainer.step_itr = 0  # number of iterations
        trainer.step_episode = {}
        trainer.follower.stats['episode_rewards'] = []
        f_train_results = None
        follower_update_count = 0
        
        joint_policy: JointPolicy = trainer.joint_policy

        # Initialize last_performance
        last_performance = trainer.evaluate_once(reset_env=True)

        # Episode start handling
        obs, _ = trainer.env.reset()
        a, _ = joint_policy.get_action(  # a[0]: leader action, a[1]: follower action
            obs,
            explorate_leader=(not trainer.total_env_steps > self._init_steps),
            explorate_follower=(not trainer.total_env_steps > trainer.follower._init_steps),
            deterministic_leader=(self.fixed_policy 
                                  and self.is_deterministic_policy),
            deterministic_follower=(trainer.follower.fixed_policy 
                                    and trainer.follower.is_deterministic_policy),
            )
        ret, t_ret, ep_len = 0, 0, 0
        episode = []
        last_ret, last_t_ret = 0, 0
        last_episode = []
        
        # Training loop
        for _ in trainer.step_epochs():

            # Set False at the beginning of each epoch.
            # Change them to True if the policy of the follower/leader is updated. 
            trainer.enable_follower_logging = False
            trainer.enable_leader_logging = False

            for _ in range(trainer.train_args.itr_per_epoch):
                for _ in range(trainer.train_args.steps_per_itr):

                    # --- Collect samples from the environment --- #
                    # Step the env
                    es = trainer.env.step(a)
                    trainer.total_env_steps += 1

                    act = es.action
                    leader_act = es.env_info.pop('leader_action')
                    r = es.reward
                    target_r = es.env_info.pop('target_reward')
                    n_obs = es.observation
                    d = es.terminal  # = (obs == terminal state)
                    last_flag = es.last  # garage._dtypes.StepType.TIMEOUT or garage._dtypes.StepType.TERMINAL
                    info = es.env_info

                    # Get the next action
                    n_a, _ = joint_policy.get_action(
                        n_obs,
                        explorate_leader=(not trainer.total_env_steps > self._init_steps),
                        explorate_follower=(not trainer.total_env_steps > trainer.follower._init_steps),
                        deterministic_leader=(self.fixed_policy 
                                              and self.is_deterministic_policy),
                        deterministic_follower=(trainer.follower.fixed_policy 
                                                and trainer.follower.is_deterministic_policy),
                        )
                    
                    # Store the sample for the follower in its replay buffer
                    f_obs = self.env_spec.get_input_for(
                        'follower', 'policy',
                        obs=np.copy(obs), 
                        leader_act=np.copy(leader_act)
                        ).detach().cpu().numpy()
                    f_n_obs = self.env_spec.get_input_for(
                        'follower', 'policy',
                        obs=np.copy(n_obs),
                        leader_act=np.copy(n_a[0])
                        ).detach().cpu().numpy()
                    
                    f_sample = dict(observation=f_obs,
                                    action=act,
                                    reward=r,
                                    next_observation=f_n_obs,
                                    terminal=d)
                    trainer.follower.replay_buffer.add_transition(**f_sample)

                    # Store the sample for the leader in its replay buffer
                    sample = dict(observation=obs,
                                  action=act,
                                  leader_action=leader_act,
                                  reward=r,
                                  target_reward=target_r,
                                  next_observation=n_obs,
                                  terminal=d,
                                  last=last_flag,
                                  time_step=ep_len)
                    self.replay_buffer.add_transition(**sample)

                    # Incrementation
                    ep_len += 1
                    obs = n_obs
                    a = n_a
                    ret += r
                    t_ret += target_r
                    episode.append(sample)

                    # End of episode handling
                    if d or (ep_len == self._max_episode_length):
                        last_ret = ret
                        last_t_ret = t_ret
                        last_episode = episode
                        trainer.follower.stats['episode_rewards'].append(ret)
                        self._episode_rewards.append(t_ret)

                        obs, _ = trainer.env.reset()
                        a, _ = joint_policy.get_action(  # a[0]: leader action, a[1]: follower action
                            obs,
                            explorate_leader=(not trainer.total_env_steps > self._init_steps),
                            explorate_follower=(not trainer.total_env_steps > trainer.follower._init_steps),
                            deterministic_leader=(self.fixed_policy 
                                                  and self.is_deterministic_policy),
                            deterministic_follower=(trainer.follower.fixed_policy 
                                                    and trainer.follower.is_deterministic_policy),
                            )
                        ret, t_ret, ep_len = 0, 0, 0
                        episode = []

                # --- Training step --- #

                # Train the follower
                if ((trainer.follower.replay_buffer.n_transitions_stored 
                         >= trainer.follower._min_buffer_size)
                    and not trainer.follower.fixed_policy):

                    for _ in range(trainer.follower._gradient_steps_n):
                        f_train_results = trainer.follower.train_once()
                    trainer.enable_follower_logging = True
                    follower_update_count += 1

                # Train the leader
                if (follower_update_count % self._update_interval == 0
                    and (self.replay_buffer.n_transitions_stored 
                         >= self._min_buffer_size)
                    and trainer.total_env_steps > self._update_after
                    and not self.fixed_policy):

                    actor_updated = self.train_once(trainer=trainer)
                    trainer.enable_leader_logging |= actor_updated

                trainer.step_itr += 1

            # --- End of epoch handling ---- #

            # Evaluation
            last_performance = trainer.evaluate_once(reset_env=True)
            
            # Log training statistics
            if trainer.enable_follower_logging:
                if not len(trainer.follower.stats['episode_rewards']) > 0:
                    trainer.follower.stats['episode_rewards'].append(last_ret)
                trainer.follower.log_statistics(*f_train_results)
                trainer.follower.stats['episode_rewards'] = []
            if trainer.enable_leader_logging:
                if not len(self._episode_rewards) > 0:
                    self._episode_rewards.append(last_t_ret)
                self.log_statistics(trainer)
                
            # Trainer stats
            trainer.step_episode = {
                k: [i[k] for i in last_episode] 
                for k in last_episode[0].keys()
                } if last_episode else {}
            
        return np.mean(last_performance['AverageTargetReturn'])

    @abc.abstractmethod
    def train_once(self, trainer):
        """Perform one iteration of training.

        Args:
            trainer (Trainer): Experiment trainer.
        """

    @abc.abstractmethod     
    def log_statistics(self, trainer, prefix='Leader'):
        """Log training statistics.

        Args:

        """

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        device = device or global_device()
        for net in self.networks:
            net.to(device)

    @property
    @abc.abstractmethod
    def networks(self):
        """List of networks in the model.

        Returns:
            list: List of networks in the model.

        """
