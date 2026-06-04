# yapf: disable
import copy
import numpy as np
from garage import StepType

from ..experiment import Trainer
from ..policies import JointPolicy
from ..utils import reset_module_parameters, reset_optimizer
from .biac_discrete_opt import BiAC_Opt

# yapf: enable

class BiAC_Subopt(BiAC_Opt):
    name='BiAC_Subopt'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, trainer: Trainer):
        """Obtain samples and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer.

        Returns:
            float: The average return in last epoch cycle.

        """
        if self._wb_follower:
            self._hat_f_policy = trainer.follower.policy
            self._hat_f_vf = trainer.follower.make_value_function()  # Assume the leader policy is stochastic
        
        self._last_follower_policy = copy.deepcopy(self._hat_f_policy)
        self.f_discount = None

        # for evaluation under the optimal follower policy
        self._opt_follower_policy = copy.deepcopy(trainer.follower.policy)
        
        trainer.step_itr = 0  # number of iterations (trainer.obtain_samples() calls)
        trainer.step_episode = {}
        trainer.follower.stats['episode_rewards'] = []
        max_diff, info = None, None
        
        # Initialize last_performance
        last_performance = trainer.evaluate_once(reset_env=True)

        # Training loop
        for _ in trainer.step_epochs():

            # Set False at the beginning of each epoch.
            # Change them to True if the policy of the follower/leader is updated. 
            trainer.enable_follower_logging = False
            trainer.enable_leader_logging = False

            for _ in range(trainer.train_args.itr_per_epoch):
                trainer.step_episode = trainer.obtain_samples(
                    trainer.step_itr,
                )
                path_returns = []
                path_target_returns = []
                for path in trainer.step_episode:
                    l_acts = path['env_infos'].pop('leader_action')
                    r = path['rewards'].reshape(-1, 1)
                    target_r = path['env_infos'].pop('target_reward').reshape(-1, 1)
                    n_obs = path['next_observations']
                    time_steps = np.array([i for i in range(len(n_obs))])
                    d = np.array([
                        step_type == StepType.TERMINAL
                        for step_type in path['step_types']
                    ]).reshape(-1, 1)
                    last_flags = np.array([
                        step_type == StepType.TERMINAL or
                        step_type == StepType.TIMEOUT
                        for step_type in path['step_types']
                    ]).reshape(-1, 1)
                    self.replay_buffer.add_transitions(
                        **dict(
                            observation=path['observations'],
                            action=path['actions'],
                            leader_action=l_acts,
                            reward=r,
                            target_reward=target_r,
                            next_observation=n_obs,
                            terminal=d,
                            last=last_flags,
                            time_step=time_steps
                        )
                    )
                    path_returns.append(sum(r))
                    path_target_returns.append(sum(target_r))
                assert len(path_returns) == len(trainer.step_episode)
                trainer.follower.stats['episode_rewards'].append(path_returns)
                self._episode_rewards.append(np.mean(path_target_returns))

                # --- Training step --- #

                # Train the leader
                if (self.replay_buffer.n_transitions_stored >= self._min_buffer_size
                    and not self.fixed_policy):

                    # If update critic from scratch
                    if self._reset_leader_qf:
                        reset_module_parameters(self._qf)
                        self._target_qf = copy.deepcopy(self._qf)
                    if self._reset_leader_qf_optimizer:
                        reset_optimizer(self._qf_optimizer, self._qf, self._qf_lr)

                    for _ in range(self._gradient_steps_n):
                        self.train_once(trainer=trainer)
                    trainer.enable_leader_logging = True

                # Train the follower
                if not trainer.follower.fixed_policy:
                    max_diff, info = trainer.follower.train(trainer=trainer)
                    trainer.enable_follower_logging = True

                trainer.step_itr += 1

            # --- End of epoch handling ---- #

            # Evaluation
            # Evaluation under the current follower policy
            last_performance = trainer.evaluate_once(
                reset_env=True,
                prefix='Evaluation',
            )
            # Evaluation under the optimal follower policy
            self._opt_follower_policy.set_policy_matrix(
                trainer.follower.policy_matrix(Q_table=info['final_Q'])
            )
            trainer.evaluate_once(
                agent_update=JointPolicy(
                    env_spec=trainer.env_spec,
                    leader_policy=trainer.leader.policy,
                    follower_policy=self._opt_follower_policy
                ),
                reset_env=True,
                prefix='Evaluation_opt',
            )

            # Compute the follower optimality gap (return-based)
            eval_dict = trainer.eval_tabular.as_dict
            _b = trainer.follower.beta
            opt_J_wo_entropy = eval_dict['Evaluation_opt/AverageDiscountedReturn']
            post_J_wo_entropy = eval_dict['Evaluation/AverageDiscountedReturn']
            opt_J = opt_J_wo_entropy - _b * eval_dict['Evaluation_opt/AverageDiscountedLogProb']
            post_J = post_J_wo_entropy - _b * eval_dict['Evaluation/AverageDiscountedLogProb']
            relative_optimality_gap = (opt_J - post_J) / (np.abs(opt_J) + 1e-10)
            trainer.eval_tabular.record(
                'Evaluation/FollowerRelativeOptimalityGap', relative_optimality_gap
            )

            # Log training statistics
            if trainer.enable_follower_logging:
                trainer.follower.log_statistics(max_diff, info)
                trainer.follower.stats['episode_rewards'] = []
            if trainer.enable_leader_logging:
                self.log_statistics(trainer)
            
        return np.mean(last_performance['AverageTargetReturn'])
