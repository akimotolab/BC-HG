"""This modules creates a MADDPG model in PyTorch."""
# yapf: disable
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
from garage import make_optimizer, _Default, StepType
from garage.np.algos import RLAlgorithm
from garage.torch import as_torch_dict, torch_to_np, global_device, as_torch
from garage.torch.policies.stochastic_policy import StochasticPolicy
from garage.experiment.deterministic import get_seed

from ..utils import reset_module_parameters, reset_optimizer, correlation_coefficient

# For type hints (not necessary)
from ..experiment import Trainer

# yapf: enable

class BCHGDiscrete_Opt(RLAlgorithm):
    name = 'BCHGDiscrete_Opt'

    def __init__(
            self,
            env_spec,
            policy,
            qf,
            replay_buffer,
            *,
            buffer_batch_size=64,
            min_buffer_size=int(1e4),
            init_steps=int(1e4),
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
            actor_update_steps_n=1,
            critic_update_steps_n=2,
            batch_size_for_fa_exp=64,
            discount_sampling=True,
            on_policy=True,
            target_policy_smoothing=False,
            reset_leader_qf=False,
            reset_leader_qf_optimizer=False,
            lambda_coef_1=1.0,
            lambda_coef_2=1.0,
            use_advantage=True,
            use_advantage_in_influence=True,
            grad_mode='default',
            grad_info=False,
            no_guidance=False,
            ):
        # Hyperparameters
        self._buffer_batch_size = buffer_batch_size
        self._min_buffer_size = min_buffer_size  # Steps after which the leader starts updating
        self._init_steps = init_steps  # Steps until which uniformly random leader actions are selected
        self._tau = target_update_tau
        self._policy_weight_decay = policy_weight_decay  # Not yet used
        self._qf_weight_decay = qf_weight_decay  # Not yet used
        self._clip_pos_returns = clip_pos_returns  # Not yet used
        self._clip_return = clip_return  # Not yet used
        self._max_policy_grad_norm = max_policy_grad_norm
        self._actor_update_steps_n = actor_update_steps_n
        self._critic_update_steps_n = critic_update_steps_n
        self._batch_size_for_fa_exp = batch_size_for_fa_exp
        self._discount_sampling = discount_sampling
        self._on_policy = on_policy
        self._target_policy_smoothing = target_policy_smoothing
        self._reset_leader_qf = reset_leader_qf
        self._reset_leader_qf_optimizer = reset_leader_qf_optimizer
        self._use_advantage = use_advantage
        self._use_advantage_in_influence = use_advantage_in_influence
        self._grad_mode = grad_mode
        self._reward_scale = reward_scale
        
        self.lambda_coef_1 = lambda_coef_1
        self.lambda_coef_2 = lambda_coef_2

        # Experiment parameters
        self._wb_follower = wb_follower  # True: follower is white-box
        
        self._max_episode_length = env_spec.max_episode_length

        self.env_spec = env_spec
        self.replay_buffer = replay_buffer
        self.discount = discount
        self.policy = policy
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
            raise NotImplementedError('The follower estimation model is not implemented yet.')
        else:
            self._hat_f_vf = None
            self._hat_f_qf = None
            self._hat_f_policy = None
        
        # For computing realized guidance effect
        self._last_benefit = []
        self._last_samples = []
        self._last_follower_policy = None

        # For logging
        self._grad_info = grad_info
        self._no_guidance = no_guidance
        self._episode_qf_losses = []
        self._epoch_ys = []
        self._epoch_qs = []
        self._episode_policy_losses = []    
        self._actor_loss_1 = []
        self._actor_loss_2 = []
        self._actor_loss_2_comp = []
        self._actor_loss_2_benefit = []
        if self._grad_info:
            self._actor_loss_grad_norm = []
            self._actor_loss_1_grad_norm = []
            self._actor_loss_2_grad_norm = []
            self._grad_cosine_similarity = []
        self._average_subseq_len = []
        self._average_hat_f_val = []
        self._average_hat_f_adv = []
        self._policy_entropies = []
        self._policy_opt_gaps = []
        self._realized_guidance_effect = []
        self._episode_rewards = []

    def train(self, trainer: Trainer):
        """Obtain samples and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer.

        Returns:
            float: The average return in last epoch cycle.

        """
        # when the follower is white-box        
        if self._wb_follower:
            self._hat_f_policy = trainer.follower.policy
            self._hat_f_vf = trainer.follower.make_value_function()
        
        self._last_follower_policy = copy.deepcopy(self._hat_f_policy)
        self.f_discount = None   
        
        trainer.step_itr = 0  # number of iterations (trainer.obtain_samples() calls)
        trainer.step_episode = {}
        trainer.follower.stats['episode_rewards'] = []
        f_train_results = None
        
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

                    self.train_once(trainer=trainer)
                    trainer.enable_leader_logging = True

                # Train the follower
                if not trainer.follower.fixed_policy:
                    f_train_results = trainer.follower.train(trainer=trainer)
                    trainer.enable_follower_logging = True

                trainer.step_itr += 1

            # --- End of epoch handling ---- #

            # Evaluation
            last_performance = trainer.evaluate_once(reset_env=True)

            # Log training statistics
            if trainer.enable_follower_logging:
                trainer.follower.log_statistics(*f_train_results)
                trainer.follower.stats['episode_rewards'] = []
            if trainer.enable_leader_logging:
                self.log_statistics(trainer)
            
        return np.mean(last_performance['AverageTargetReturn'])

    def train_once(self, trainer: Trainer):
        """Perform one iteration of training.

        Args:
            itr (int): Iteration number.
            episodes (EpisodeBatch): Batch of episodes.

        """
        # Update the follower estimators
        if not self._wb_follower:
            raise NotImplementedError('The follower estimation model is not implemented yet.')
        
        # Compute the guide_effect of the previous update
        self._realized_guidance_effect.append(self.compute_realized_guidance_effect())

        # Optimize the critic (leader's Q-function)
        for _ in range(self._critic_update_steps_n):
            samples = self.replay_buffer.sample_transitions(
                self._buffer_batch_size,
                replace=self._on_policy,
                discount=False, 
                with_subsequence=False
            )
            samples['target_reward'] *= self._reward_scale

            qf_loss, y, q = self.optimize_critic(samples)
            qf_loss, y, q = torch_to_np((qf_loss, y, q))
            self._episode_qf_losses.append(qf_loss)
            self._epoch_ys.append(y)
            self._epoch_qs.append(q)

        # Optimize the policy (leader's policy)
        for _ in range(self._actor_update_steps_n):
            samples = self.replay_buffer.sample_transitions(
                self._buffer_batch_size,
                replace=self._on_policy,
                discount=self._discount_sampling, 
                with_subsequence=True
                )
            samples['target_reward'] *= self._reward_scale
            subsequences = samples.pop('subsequence')
            average_subseq_len = np.mean([len(subseq) for subseq in subsequences['observation']])

            policy_loss, info = self.optimize_policy(samples, subsequences, trainer)
            policy_loss = torch_to_np((policy_loss,))
            self._episode_policy_losses.append(policy_loss)
            self._actor_loss_1.append(info['actor_loss_1'])
            self._actor_loss_2.append(info['actor_loss_2'])
            self._actor_loss_2_comp.append(info['actor_loss_2_comp'])
            self._actor_loss_2_benefit.append(info['actor_loss_2_benefit'])
            if self._grad_info:
                self._actor_loss_grad_norm.append(info['actor_loss_grad_norm'])
                self._actor_loss_1_grad_norm.append(info['actor_loss_1_grad_norm'])
                self._actor_loss_2_grad_norm.append(info['actor_loss_2_grad_norm'])
                self._grad_cosine_similarity.append(info['grad_cosine_similarity'])
            self._average_subseq_len.append(average_subseq_len)
            self._average_hat_f_val.append(info['hat_f_val'])
            self._average_hat_f_adv.append(info['hat_f_adv'])
            self._policy_entropies.append(info['policy_entropy'])
            self._policy_opt_gaps.append(info['policy_opt_gap'])

        # Update target networks
        self.update_target()

    def optimize_critic(self, samples_data):
        """Perform critic optimizing.
        Args:
            samples_data (dict): Processed batch data.
        Returns:
            qval_loss: Loss of Q-value predicted by the critic network.
            target_y: Target Q-value for the batch.
            qval: Q-value predicted by the critic network.
        """
        # Note: samples_data is a dict of numpy arrays.
        o = samples_data['observation']
        r = samples_data['target_reward'].reshape(-1, 1)
        fa = samples_data['action']
        la = samples_data['leader_action']
        n_o = samples_data['next_observation']
        d = samples_data['terminal'].reshape(-1, 1)

        r, d = as_torch(r), as_torch(d)

        # --- Target Q-value Calculation --- #

        with torch.no_grad():
            l_target_p_input = self.env_spec.get_inputs_for('leader', 'policy',
                                                            obs=n_o)
            n_o_flat_tensor = l_target_p_input
            n_la_dist, n_la_infos = self._target_policy(l_target_p_input)

            if self._target_policy_smoothing:
                # Use the smoothed target leader action
                n_la_target = n_la_dist.sample()  # (batch_size, )
            else:
                # Use the mode of the target leader action distribution
                n_la_target = n_la_infos['mode']  # (batch_size, )

            f_target_p_input = self.env_spec.get_inputs_for('follower', 'policy',
                                                            obs=n_o_flat_tensor, leader_act=n_la_target.cpu().numpy())
            n_fa_dist, n_fa_infos = self._hat_f_policy(f_target_p_input)
            
            if self._target_policy_smoothing:
                # Use the smoothed follower action
                n_fa_target = n_fa_dist.sample()
            else:
                # Use the mode of the follower action distribution
                n_fa_target = n_fa_infos['mode']
            
            l_target_q_input = self.env_spec.get_inputs_for('leader', 'qf',
                                                            obs=n_o, follower_act=n_fa_target.cpu().numpy())
            # _target_qf now outputs Q-values for all leader actions: (batch_size, num_leader_actions)
            target_qvals_all_leader_acts = self._target_qf(l_target_q_input)  # (batch_size, num_leader_actions)
            
            # Select Q-value for the sampled next leader action
            target_qval = target_qvals_all_leader_acts.gather(
                1, n_la_target.long().unsqueeze(-1))  # (batch_size, 1)

            clip_range = (-self._clip_return,
                          0. if self._clip_pos_returns else self._clip_return)

            target_y = r + (1.0 - d) * self.discount * target_qval  # (batch_size, 1)
            target_y = torch.clamp(target_y, clip_range[0], clip_range[1])

        # --- Critic (Leader's Q-function) Optimization ---
        l_q_input_critic = self.env_spec.get_inputs_for('leader', 'qf', 
                                                        obs=o, follower_act=fa)
        # _qf outputs Q-values for all leader actions: (batch_size, num_leader_actions)
        qvals_all_leader_acts_critic = self._qf(l_q_input_critic)
        
        la_as_idx_from_buffer = as_torch(la).long()  # la is (batch_size,) or (batch_size, 1)
        if la_as_idx_from_buffer.ndim == 1:
            la_as_idx_from_buffer = la_as_idx_from_buffer.unsqueeze(-1)
        
        qval = qvals_all_leader_acts_critic.gather(1, la_as_idx_from_buffer)  # (batch_size, 1)
        qf_loss_fn = torch.nn.MSELoss()
        qval_loss = qf_loss_fn(qval, target_y)
        self._qf_optimizer.zero_grad()
        qval_loss.backward()
        self._qf_optimizer.step()

        return qval_loss.detach(), target_y, qval.detach()

    def optimize_policy(self, samples_data, sampled_subsequences, trainer):
        """Perform policy (actor) optimizing.

        Args:
            samples_data (dict): Processed batch data.
            sampled_subsequences (dict): Lists of the subsequence trajectories from the each sample for each key.
            trainer (Trainer): Trainer

        Returns:
            action_loss: Loss of action predicted by the policy network.
            info: Dictionary containing additional information about the optimization step.
        """
        # Note: samples_data is a dict of numpy arrays.
        o = samples_data['observation']
        r = samples_data['target_reward'].reshape(-1, 1)
        fa = samples_data['action']
        la = samples_data['leader_action']
        n_o = samples_data['next_observation']
        d = samples_data['terminal'].reshape(-1, 1)

        r, d = as_torch(r), as_torch(d)

        # --- Actor (Leader's Policy) Optimization ---
        l_p_input_from_buffer = self.env_spec.get_inputs_for('leader', 'policy', obs=o)
        o_flat_tensor = l_p_input_from_buffer
        la_actor_dist, _ = self.policy(l_p_input_from_buffer)
        l_q_input_actor = self.env_spec.get_inputs_for('leader', 'qf', 
                                                       obs=o_flat_tensor, follower_act=fa)
        qvals_all_leader_acts_actor = self._qf(l_q_input_actor)
        
        # First term
        la_as_idx_from_buffer = as_torch(la).long()  # la is (batch_size,) or (batch_size, 1)
        if la_as_idx_from_buffer.ndim == 1:
            la_as_idx_from_buffer = la_as_idx_from_buffer.unsqueeze(-1)
        la_as_idx_from_buffer = la_as_idx_from_buffer.detach()
        la_log_probs_of_samples = la_actor_dist.log_prob(la_as_idx_from_buffer.squeeze(-1))
        qval_of_samples = qvals_all_leader_acts_actor.gather(1, la_as_idx_from_buffer).squeeze(-1).detach()

        if self._use_advantage:
            with torch.no_grad():
                v_of_samples = torch.zeros_like(qval_of_samples) # (batch_size,)
                la_probs_of_samples = la_actor_dist.probs.detach()
                l_q_vals_f_act = qvals_all_leader_acts_actor.detach() # (batch_size, num_leader_actions)
                v_of_samples = torch.mul(l_q_vals_f_act, la_probs_of_samples).sum(dim=1)
            advantages = qval_of_samples - v_of_samples # (batch_size,)
            actor_loss_1 = -(la_log_probs_of_samples * advantages).mean()
        else:
            actor_loss_1 = -(la_log_probs_of_samples * qval_of_samples).mean()

        if self._no_guidance:
            # only the first term optimized for short execution time
            actor_loss = self.lambda_coef_1 * actor_loss_1
            self._policy_optimizer.zero_grad()
            actor_loss.backward(retain_graph=self._grad_info)

            actor_loss_grad_norm = torch.tensor(0.0)
            if self._grad_info:
                actor_loss_grads = [p.grad.clone().detach() for p in self.policy.parameters() if p.grad is not None]
                actor_loss_grad_norm = torch.sqrt(
                    sum(torch.sum(g**2) for g in actor_loss_grads)) if actor_loss_grads else torch.tensor(0.0)

            self._policy_optimizer.step()

            policy_entropy = la_actor_dist.entropy().mean().detach().item() if hasattr(la_actor_dist, 'entropy') else 0.0

            policy_opt_gap = trainer.env.leader_policy_opt_gap(self.policy)

            info = dict(
                actor_loss_1=actor_loss.detach().item(),
                actor_loss_2=0.0,
                actor_loss_2_comp=0.0,
                actor_loss_2_benefit=0.0,
                hat_f_val=0.0,
                hat_f_adv=0.0,
                policy_entropy=policy_entropy,
                policy_opt_gap=policy_opt_gap,
            )
            if self._grad_info:
                info.update(
                    actor_loss_grad_norm=actor_loss_grad_norm.item(),
                    actor_loss_1_grad_norm=actor_loss_grad_norm.item(),
                    actor_loss_2_grad_norm=0.0,
                    grad_cosine_similarity=0.0,
                )
            return actor_loss.detach(), info

        # Second term
        # Benefit calculation
        with torch.no_grad():
            f_p_input_from_buffer = self.env_spec.get_inputs_for('follower', 'policy', 
                                                                 obs=o, leader_act=la)
            fa_actor_dist, _ = self._hat_f_policy(f_p_input_from_buffer)
            l_q_exp_on_fa = torch.zeros_like(qval_of_samples) # (batch_size,)
            for _ in range(self._batch_size_for_fa_exp):
                fa_for_exp = fa_actor_dist.sample().cpu().numpy() # Sample follower action
                l_q_input_for_exp = self.env_spec.get_inputs_for('leader', 'qf', 
                                                                 obs=o, follower_act=fa_for_exp)
                # Q-values for all leader actions, given o and sampled fa_for_exp
                qvals_all_leader_acts_for_exp = self._qf(l_q_input_for_exp)
                # Expected Q-value over current leader policy
                qvals_for_exp = qvals_all_leader_acts_for_exp.gather(
                    1, la_as_idx_from_buffer).squeeze(-1) # (batch_size,)
                l_q_exp_on_fa += qvals_for_exp

            baseline_for_benefit = l_q_exp_on_fa / self._batch_size_for_fa_exp
            benefit = qval_of_samples - baseline_for_benefit # (batch_size,)

            # For computing the realized guidance effect
            self._last_benefit.append(benefit.clone().detach())
            self._last_samples.append({'observation': o.copy(), 
                                       'action': fa.copy(), 
                                       'leader_action': la.copy()})

        # influence calculation
        influence = []
        
        if self.f_discount is None:
            self.f_discount = as_torch(
                [trainer.follower.discount ** i for i in range(self._max_episode_length)]
                ).unsqueeze(1) # (max_episode_length, 1)

        hat_f_vals_log = []
        hat_f_advs_log = []
        for i, (o_subseq, la_subseq) in enumerate(zip(sampled_subsequences['observation'], 
                                                      sampled_subsequences['leader_action'])):
            
            l_p_input_subseq = self.env_spec.get_inputs_for('leader', 'policy',
                                                            obs=o_subseq)
            o_subseq_flat_tensor = l_p_input_subseq
            la_subseq_dist, _ = self.policy(l_p_input_subseq)
            la_subseq_torch = as_torch(la_subseq) # (subseq_len,)
            la_subseq_log_probs = la_subseq_dist.log_prob(la_subseq_torch) # (subseq_len,)

            with torch.no_grad():
                if self._wb_follower:
                    if self._use_advantage_in_influence:
                        la_subseq_probs = la_subseq_dist.probs # (subseq_len, num_leader_actions)
                        _subseq_v2 = []
                        for la_i in range(self.env_spec.leader_action_space.n):
                            l_acts = np.full(la_subseq.shape, la_i)
                            f_q_input_subseq = self.env_spec.get_inputs_for('follower', 'qf',
                                                                            obs=o_subseq_flat_tensor, leader_act=l_acts)
                            v = self._hat_f_vf(f_q_input_subseq).squeeze(-1) # (subseq_len,)
                            _subseq_v2.append(v)
                        _subseq_v2 = torch.stack(_subseq_v2, dim=1) # (subseq_len, num_leader_actions)
                        hat_fv_vals_subseq = _subseq_v2.gather(
                            1, la_subseq_torch.long().unsqueeze(-1)).squeeze(-1) # (subseq_len,)
                        hat_f_vals_log.append(hat_fv_vals_subseq.mean().item())
                        baseline = (_subseq_v2 * la_subseq_probs).sum(dim=1) # (subseq_len,)
                        hat_fv_vals_subseq -= baseline # (subseq_len,)
                        hat_f_advs_log.append(hat_fv_vals_subseq.mean().item())
                    else:
                        f_q_input_subseq = self.env_spec.get_inputs_for('follower', 'qf', 
                                                                        obs=o_subseq_flat_tensor, leader_act=la_subseq)
                        hat_fv_vals_subseq = self._hat_f_vf(f_q_input_subseq).squeeze(-1) # (subseq_len,)
                        hat_f_vals_log.append(hat_fv_vals_subseq.mean().item())
                else:
                    raise NotImplementedError('Follower estimation model is not implemented yet.')

            log_hat_f_p_hat_fv_subseq = (la_subseq_log_probs * hat_fv_vals_subseq).unsqueeze(-1) # (subseq_len, 1)

            if self._grad_mode == 'default':
                influence.append((self.f_discount[:len(log_hat_f_p_hat_fv_subseq)] * log_hat_f_p_hat_fv_subseq).sum())
            elif self._grad_mode == 'weighted_average':
                weight = self.f_discount[:len(log_hat_f_p_hat_fv_subseq)]
                influence.append((weight * log_hat_f_p_hat_fv_subseq).sum() / weight.sum())
            else:
                raise ValueError(f'Unknown grad_mode: {self._grad_mode}')
            
        influence = torch.stack(influence, dim=0) # (batch_size,)
        actor_loss_2 = -1.0 / trainer.follower.beta * (benefit * influence).mean()
            
        # Compute the final actor loss
        actor_loss_1 = self.lambda_coef_1 * actor_loss_1
        actor_loss_2 = self.lambda_coef_2 * actor_loss_2
        actor_loss = actor_loss_1 + actor_loss_2

        # Proceed with the combined loss optimization

        if self._grad_info:
            # Compute gradients for actor_loss_1
            self._policy_optimizer.zero_grad()
            actor_loss_1.backward(retain_graph=True)
            actor_loss_1_grads = [
                p.grad.clone().detach() if p.grad is not None else torch.zeros_like(p)
                for p in self.policy.parameters()
            ]

            # Compute gradients for actor_loss_2
            self._policy_optimizer.zero_grad()
            actor_loss_2.backward(retain_graph=False)
            actor_loss_2_grads = [
                p.grad.clone().detach() if p.grad is not None else torch.zeros_like(p)
                for p in self.policy.parameters()
            ]

            # Compute norms and inner product
            grad1 = torch.cat([g.flatten() for g in actor_loss_1_grads])
            grad2 = torch.cat([g.flatten() for g in actor_loss_2_grads])
            grad_sum = grad1 + grad2
            actor_loss_1_grad_norm = grad1.norm()
            actor_loss_2_grad_norm = grad2.norm()
            actor_loss_grad_norm = grad_sum.norm()
            if actor_loss_1_grad_norm > 0 and actor_loss_grad_norm > 0:
                grad_cosine_similarity = (
                    torch.dot(grad1, grad_sum)) / (actor_loss_1_grad_norm * actor_loss_grad_norm)
            else:
                grad_cosine_similarity = torch.tensor(0.0)

            for p, g1, g2 in zip(self.policy.parameters(), actor_loss_1_grads, actor_loss_2_grads):
                if p.grad is None:
                    p.grad = (g1 + g2).clone()
                else:
                    p.grad.copy_(g1 + g2)
        else:
            self._policy_optimizer.zero_grad()
            actor_loss.backward()

        if self._max_policy_grad_norm is not None:
            # Clip the gradients of the policy network
            utils.clip_grad_norm_(self.policy.parameters(), self._max_policy_grad_norm)

        self._policy_optimizer.step()

        # Compute the policy entropy
        policy_entropy = la_actor_dist.entropy().mean().detach().item() if hasattr(la_actor_dist, 'entropy') else 0.0

        # Compute the optimal policy gap
        policy_opt_gap = trainer.env.leader_policy_opt_gap(self.policy)

        info = dict(
            actor_loss_1=actor_loss_1.detach().item(),
            actor_loss_2=actor_loss_2.detach().item(),
            actor_loss_2_comp=influence.mean().detach().item(),
            actor_loss_2_benefit=benefit.mean().detach().item(),
            hat_f_val=float(np.mean(hat_f_vals_log)) if hat_f_vals_log else 0.0,
            hat_f_adv= float(np.mean(hat_f_advs_log)) if hat_f_advs_log else 0.0,
            policy_entropy=policy_entropy,
            policy_opt_gap=policy_opt_gap
        )
        if self._grad_info:
            info.update(
                actor_loss_grad_norm=actor_loss_grad_norm.item(),
                actor_loss_1_grad_norm=actor_loss_1_grad_norm.item(),
                actor_loss_2_grad_norm=actor_loss_2_grad_norm.item(),
                grad_cosine_similarity=grad_cosine_similarity.item(),
            )

        return actor_loss.detach(), info
    
    def compute_realized_guidance_effect(self, trainer=None):
        """Compute the correlation coefficient between the benefit and the log ratio 
           of the probability difference in the follower's policy as the guide effect.
           Needs adjustment for discrete leader actions / stochastic policy.
        """

        if not hasattr(self, '_last_benefit') or len(self._last_benefit) == 0:
            return 0.0
        
        with torch.no_grad():
            benefit = torch.cat(self._last_benefit) # (N,)

            obs_list, follower_act_list, leader_act_list = [], [], []
            for sample_batch in self._last_samples:
                obs_list.append(sample_batch['observation'])
                follower_act_list.append(sample_batch['action'])
                leader_act_list.append(sample_batch['leader_action'])

            observations = np.concatenate(obs_list, axis=0) # (N,)
            leader_acts = np.concatenate(leader_act_list, axis=0) # (N,)
            follower_acts = as_torch(follower_act_list).view(-1, 1).squeeze(-1) # (N,)

            f_policy_input = self.env_spec.get_inputs_for('follower', 'policy', 
                                                          obs=observations, leader_act=leader_acts)
            
            last_f_dist, _ = self._last_follower_policy(f_policy_input)
            current_f_dist, _ = self._hat_f_policy(f_policy_input)
            
            # log_prob of follower_act under current and last follower policies
            log_prob_current_f = current_f_dist.log_prob(follower_acts) # (N,)
            log_prob_last_f = last_f_dist.log_prob(follower_acts)       # (N,)

            log_ratio_follower_policy = log_prob_current_f - log_prob_last_f # (N,)

            if log_ratio_follower_policy.sum() == 0:
                corr = 0.0
            else:
                corr = correlation_coefficient(benefit, log_ratio_follower_policy)
            
            self._last_benefit = []
            self._last_samples = []
            self._update_last_follower()

        return corr
            
    def log_statistics(self, trainer, prefix='Leader'):        
        tabular = trainer.leader_tabular
        with tabular.prefix(prefix + '/'):
            tabular.record('Policy/AveragePolicyLoss',
                            np.mean(self._episode_policy_losses) if self._episode_policy_losses else float('nan'))
            tabular.record('Policy/AveragePolicyLossComp_1',
                            np.mean(self._actor_loss_1) if self._actor_loss_1 else float('nan'))
            tabular.record('Policy/AveragePolicyLossComp_2',
                            np.mean(self._actor_loss_2) if self._actor_loss_2 else float('nan'))
            tabular.record('Policy/AveragePolicyLossComp_2_wo_Benefit',
                            np.mean(self._actor_loss_2_comp) if self._actor_loss_2_comp else float('nan'))
            tabular.record('Policy/AverageBenefit',
                            np.mean(self._actor_loss_2_benefit) if self._actor_loss_2_benefit else float('nan'))

            if self._grad_info:
                tabular.record('Policy/AveragePolicyGrad_L2Norm',
                                np.mean(self._actor_loss_grad_norm) if self._actor_loss_grad_norm else float('nan'))
                tabular.record('Policy/AveragePolicyGrad_1_L2Norm',
                                np.mean(self._actor_loss_1_grad_norm) if self._actor_loss_1_grad_norm else float('nan'))
                tabular.record('Policy/AveragePolicyGrad_2_L2Norm',
                                np.mean(self._actor_loss_2_grad_norm) if self._actor_loss_2_grad_norm else float('nan'))
                tabular.record('Policy/AveragePolicyGradCosineSimilarity',
                                np.mean(self._grad_cosine_similarity) if self._grad_cosine_similarity else float('nan'))
            
            tabular.record('Policy/AverageSubseqLength',
                            np.mean(self._average_subseq_len) if self._average_subseq_len else float('nan'))
            tabular.record('Policy/AverageHatFVal',
                            np.mean(self._average_hat_f_val) if self._average_hat_f_val else float('nan'))
            tabular.record('Policy/AverageHatFAdv',
                            np.mean(self._average_hat_f_adv) if self._average_hat_f_adv else float('nan'))
            tabular.record('Policy/AveragePolicyEntropy',
                            np.mean(self._policy_entropies))
            tabular.record('Policy/PolicyOptimalityGap',
                            np.mean(self._policy_opt_gaps))
            tabular.record('AverageRealizedGuidanceEffect',
                            np.mean(self._realized_guidance_effect) if self._realized_guidance_effect else float('nan'))
            tabular.record('MaxRealizedGuidanceEffect',
                            np.max(self._realized_guidance_effect) if self._realized_guidance_effect else float('nan'))
            tabular.record('MinRealizedGuidanceEffect',
                            np.min(self._realized_guidance_effect) if self._realized_guidance_effect else float('nan'))

            tabular.record('QFunction/AverageQFunctionLoss',
                            np.mean(self._episode_qf_losses) if self._episode_qf_losses else float('nan'))
            tabular.record('QFunction/AverageQ',
                            np.mean(self._epoch_qs) if self._epoch_qs else float('nan'))
            tabular.record('QFunction/MaxQ',
                            np.max(self._epoch_qs) if self._epoch_qs else float('nan'))
            tabular.record('QFunction/AverageAbsQ',
                            np.mean(np.abs(self._epoch_qs)) if self._epoch_qs else float('nan'))
            tabular.record('QFunction/AverageY',
                            np.mean(self._epoch_ys) if self._epoch_ys else float('nan'))
            tabular.record('QFunction/MaxY',
                            np.max(self._epoch_ys) if self._epoch_ys else float('nan'))
            tabular.record('QFunction/AverageAbsY',
                            np.mean(np.abs(self._epoch_ys)) if self._epoch_ys else float('nan'))
            tabular.record('TrainAverageTargetReturn',
                            np.mean(self._episode_rewards))
        
        # Reset lists
        self._episode_policy_losses = []
        self._actor_loss_1 = []
        self._actor_loss_2 = []
        self._actor_loss_2_comp = []
        self._actor_loss_2_benefit = []
        self._average_subseq_len = []
        self._episode_qf_losses = []
        self._epoch_qs = []
        self._epoch_ys = []
        self._episode_rewards = []
        self._realized_guidance_effect = []
        self._average_hat_f_val = []
        self._average_hat_f_adv = []
        self._policy_entropies = []
        self._policy_opt_gaps = []
        if self._grad_info:
            self._actor_loss_grad_norm = []
            self._actor_loss_1_grad_norm = []
            self._actor_loss_2_grad_norm = []
            self._grad_cosine_similarity = []

    def update_target(self, qf=True, policy=True):
        """Update parameters in the target policy and Q-value network."""
        if qf:
            for t_param, param in zip(self._target_qf.parameters(),
                                    self._qf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self._tau) +
                                param.data * self._tau)
        if policy:
            for t_param, param in zip(self._target_policy.parameters(),
                                    self.policy.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self._tau) +
                                param.data * self._tau)
            
    def _update_last_follower(self):
        """Update the last follower policy to compute the guide effect."""
        last = [self._last_follower_policy]
        current = [self._hat_f_policy]
        for l, c in zip(last, current):
            for l_param, c_param in zip(l.parameters(), c.parameters()):
                l_param.data.copy_(c_param.data)
                
    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        device = device or global_device()
        for net in self.networks:
            net.to(device)

    @property
    def networks(self):
        """List of networks in the model."""
        if self._wb_follower:
            return [self._policy, self._qf, self._target_policy, self._target_qf]
        else:
            return [self._policy, self._qf, self._target_policy, self._target_qf, 
                    self._hat_qf, self._hat_f_policy]
