"""This modules creates a MADDPG model in PyTorch."""
# yapf: disable
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils as utils
from garage import make_optimizer, _Default, StepType
from garage.np.algos import RLAlgorithm
from garage.torch import as_torch_dict, torch_to_np, global_device, as_torch
from garage.torch.policies.stochastic_policy import StochasticPolicy
from garage.experiment.deterministic import get_seed

from ..policies import LinearGaussianPolicy
from ..utils import reset_module_parameters, reset_optimizer, correlation_coefficient

# For type hints (not necessary)
from ..experiment import Trainer

# yapf: enable

class BCHG_Opt(RLAlgorithm):
    name = 'BCHG_Opt'

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
            batch_size_for_la_exp=64,
            discount_sampling=True,
            on_policy=True,
            target_policy_smoothing=False,
            reset_leader_qf=False,
            reset_leader_qf_optimizer=False,
            lambda_coef_1=1.0,
            lambda_coef_2=1.0,
            use_advantage=True,
            use_advantage_in_influence=True,
            use_closed_form_gradient=True,
            use_K_L2_regularization=True,
            K_L2_reg_coef=1.0,
            grad_mode='default',
            grad_info=False,
            no_guidance=False,
            ):
        # Hyperparameters
        self._buffer_batch_size = buffer_batch_size
        self._min_buffer_size = min_buffer_size  # Steps after which the leader starts updating
        self._init_steps = init_steps  # Steps until which the leader acts randomly (refered in trainer)
        self._tau = target_update_tau
        self._policy_weight_decay = policy_weight_decay  # Not yet used
        self._qf_weight_decay = qf_weight_decay  # Not yet used
        self._clip_pos_returns = clip_pos_returns  # Not yet used
        self._clip_return = clip_return  # Not yet used
        self._max_policy_grad_norm = max_policy_grad_norm
        self._actor_update_steps_n = actor_update_steps_n
        self._critic_update_steps_n = critic_update_steps_n
        self._batch_size_for_fa_exp = batch_size_for_fa_exp
        self._batch_size_for_la_exp = batch_size_for_la_exp
        self._discount_sampling = discount_sampling
        self._on_policy = on_policy
        self._target_policy_smoothing = target_policy_smoothing
        self._reset_leader_qf = reset_leader_qf
        self._reset_leader_qf_optimizer = reset_leader_qf_optimizer
        self._use_advantage = use_advantage
        self._use_advantage_in_influence = use_advantage_in_influence
        self._use_closed_form_gradient = use_closed_form_gradient
        self._use_K_L2_regularization = use_K_L2_regularization
        self._K_L2_reg_coef = K_L2_reg_coef
        self._grad_mode = grad_mode
        self._reward_scale = reward_scale
        
        self.lambda_coef_1 = lambda_coef_1
        self.lambda_coef_2 = lambda_coef_2

        # Experiment parameters
        self._wb_follower = wb_follower  # True: follower is white-box
        
        self._max_episode_length = env_spec.max_episode_length

        self.env_spec = env_spec
        self.replay_buffer = replay_buffer
        self.exploration_policy = exploration_policy
        self.discount = discount
        self.policy = policy
        self.exploration_policy = None
        self.fixed_policy = fixed_policy
        self.is_deterministic_policy = not isinstance(self.policy, StochasticPolicy)

        self._policy = self.policy
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
            self._hat_f_policy = None
        
        # For computing of realized guidance effect
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
        self._policy_opt_gaps = []
        self._realized_guidance_effect = []
        self._episode_rewards = []
        self._policy_entropy = []
        self._K_L2_norm = []
        self._W_L2_norm = []

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
        trainer.follower.train(trainer=trainer)

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

                    # Compute the guide_effect of the previous update
                    self._realized_guidance_effect.append(self.compute_realized_guidance_effect())

                    self.train_once(trainer=trainer)
                    trainer.enable_leader_logging = True
                    
                    # Save the follower's policy at the leader's　policy update timing
                    self._update_last_follower()

                # Optimize the follower
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
        # Update the follower estimators # MARK: あとで実装
        if not self._wb_follower:
            raise NotImplementedError('The follower estimation model is not implemented yet.')

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

            if self._use_closed_form_gradient:
                policy_loss, info = self.optimize_policy_with_closed_form_gradient(samples, subsequences, trainer)
            else:
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
            self._policy_opt_gaps.append(info['policy_opt_gap'])
            self._policy_entropy = info['policy_entropy']
            self._K_L2_norm = info['K_L2_norm']
            self._W_L2_norm = info['W_L2_norm']

        # Update target networks
        self.update_target()

    def optimize_critic(self, samples_data):
        """Perform algorithm optimizing.

        Args:
            samples_data (dict): Sampled data for optimization.
            trainer (Trainer): Current trainer.
            sampled_subsequences (dict): Sampled subsequences for optimization.
            update_actor (bool): Whether to update the actor (leader policy).

        Returns:
            action_loss: Loss of action predicted by the policy network.
            qval_loss: Loss of Q-value predicted by the Q-network.
            ys: y_s.
            qval: Q-value predicted by the Q-network.
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
        n_o_flat_tensor = as_torch(self.env_spec.observation_space.flatten_n(n_o))
        la_flat_tensor = as_torch(self.env_spec.leader_action_space.flatten_n(la))
        with torch.no_grad():
            l_target_p_input = n_o_flat_tensor
            n_la_dist, n_la_infos = self._target_policy(l_target_p_input)
            if self._target_policy_smoothing:
                n_la = n_la_dist.sample()
            else:
                n_la = n_la_dist.mean  # tensor
            f_target_p_input = self.env_spec.get_inputs_for('follower', 'policy', 
                                                            obs=n_o_flat_tensor, 
                                                            leader_act=n_la)
            n_fa_dist, n_fa_infos = self._hat_f_policy(f_target_p_input)
            if self._target_policy_smoothing:
                n_fa = n_fa_dist.sample()
            elif 'mean' in n_fa_infos:
                n_fa = n_fa_infos['mean']
            elif 'mode' in n_fa_infos:
                n_fa = n_fa_infos['mode']
            else:
                raise NotImplementedError("n_fa_infos must contain 'mean' or 'mode'.")
            l_target_q_input = self.env_spec.get_inputs_for('leader', 'qf', 
                                                            obs=n_o_flat_tensor, 
                                                            follower_act=n_fa)
            target_qval = self._target_qf(l_target_q_input, n_la)
            clip_range = (-self._clip_return, 
                          0. if self._clip_pos_returns else self._clip_return)
            target_y = r + (1.0 - d) * self.discount * target_qval
            target_y = torch.clamp(target_y, clip_range[0], clip_range[1])  # (batch_size, 1)

        # --- Critic (Leader's Q-function) Optimization --- #

        l_q_input_from_buffer = self.env_spec.get_inputs_for('leader', 'qf', 
                                                             obs=o, 
                                                             follower_act=fa)
        qval = self._qf(l_q_input_from_buffer, la_flat_tensor)  # (batch_size, 1)
        qf_loss = torch.nn.MSELoss()
        qval_loss = qf_loss(qval, target_y)
        self._qf_optimizer.zero_grad()
        qval_loss.backward()
        self._qf_optimizer.step()

        return qval_loss.detach(), target_y, qval.detach()

    def optimize_policy(self, samples_data, sampled_subsequences, trainer):
        """Perform algorithm optimizing.

        Args:
            samples_data (dict): Sampled data for optimization.
            trainer (Trainer): Current trainer.
            sampled_subsequences (dict): Sampled subsequences for optimization.
            update_actor (bool): Whether to update the actor (leader policy).

        Returns:
            action_loss: Loss of action predicted by the policy network.
            qval_loss: Loss of Q-value predicted by the Q-network.
            ys: y_s.
            qval: Q-value predicted by the Q-network.
        """
        # Note: samples_data is a dict of numpy arrays.
        o = samples_data['observation']
        r = samples_data['target_reward'].reshape(-1, 1)
        fa = samples_data['action']
        la = samples_data['leader_action']
        n_o = samples_data['next_observation']
        d = samples_data['terminal'].reshape(-1, 1)

        # --- Actor (Leader's Policy) Optimization --- #

        actor_loss = torch.tensor([0.0])
        info = dict()

        # --- First term --- #

        l_q_input_from_buffer = self.env_spec.get_inputs_for('leader', 'qf', 
                                                             obs=o, 
                                                             follower_act=fa)
        l_p_input_from_buffer = self.env_spec.get_inputs_for('leader', 'policy', 
                                                                obs=o)
        la_dist, _ = self.policy(l_p_input_from_buffer)
        la_tensor = as_torch(la)
        la_flat_tensor = as_torch(self.env_spec.leader_action_space.flatten_n(la))
        log_pi_la_samples = la_dist.log_prob(value=la_tensor)  # (batch_size,)
        with torch.no_grad():
            qval_of_samples = self._qf(l_q_input_from_buffer, la_flat_tensor)  # (batch_size, 1)
            if self._use_advantage:
                v_actor = torch.zeros_like(qval_of_samples)  # (batch_size, 1)
                for _ in range(self._batch_size_for_la_exp):
                    la_for_exp = la_dist.sample()
                    v_actor = v_actor + self._qf(l_q_input_from_buffer, 
                                                la_for_exp)
                v_actor = v_actor / self._batch_size_for_la_exp
                advantage = qval_of_samples - v_actor
            else:
                advantage = qval_of_samples
            advantage = advantage.squeeze(-1)  # (batch_size,)
        actor_loss_1 = -(log_pi_la_samples * advantage).mean()

        if self._no_guidance:
            actor_loss = self.lambda_coef_1 * actor_loss_1
            self._policy_optimizer.zero_grad()
            actor_loss.backward(retain_graph=self._grad_info)
            if self._use_K_L2_regularization:
                if hasattr(self.policy, 'K') and isinstance(self.policy.K, nn.Parameter):
                    grad_K_L2 = self._K_L2_reg_coef * 2.0 * self.policy.K  # (action_dim, obs_dim)
                    for name, p in self.policy.named_parameters():
                        if name == 'K':
                            if p.grad is None:
                                p.grad = grad_K_L2
                            else:
                                p.grad += grad_K_L2
                grad_K_L2 = self._K_L2_reg_coef * 2.0 * self.policy.K  # (action_dim, obs_dim)
                for name, p in self.policy.named_parameters():
                    if name == 'K':
                        if p.grad is None:
                            p.grad = grad_K_L2
                        else:
                            p.grad += grad_K_L2
            actor_loss_grad_norm = torch.tensor(0.0)
            if self._grad_info:
                actor_loss_grads = [
                    param.grad.clone().detach() for param in self.policy.parameters() 
                    if param.grad is not None
                    ]
                actor_loss_grad_norm = torch.sqrt(
                    sum(torch.sum(g ** 2) for g in actor_loss_grads)
                    ) if actor_loss_grads else torch.tensor(0.0)
            self._policy_optimizer.step()

            policy_entropy = la_dist.entropy().mean().detach().item() if hasattr(la_dist, 'entropy') else 0.0
            info = dict(
                actor_loss_1=actor_loss.detach().item(),
                actor_loss_2=0.0,
                actor_loss_2_comp=0.0,
                actor_loss_2_benefit=0.0,
                hat_f_val=0.0,
                policy_entropy=policy_entropy,
                K_L2_norm=torch.norm(self.policy.K).item(),
                W_L2_norm=torch.norm(self.policy.W).item()
            )
            if self._grad_info:
                info.update(
                    actor_loss_grad_norm=actor_loss_grad_norm.item(),
                    actor_loss_1_grad_norm=actor_loss_grad_norm.item(),
                    actor_loss_2_grad_norm=0.0,
                    grad_cosine_similarity=0.0
                )
            if hasattr(trainer.env, 'leader_policy_opt_gap'):
                policy_opt_gap = trainer.env.leader_policy_opt_gap(self.policy)
                info['policy_opt_gap'] = policy_opt_gap
            else:
                info['policy_opt_gap'] = 0.0

            return actor_loss.detach(), info

        # --- Second term --- #

        # Benefit calculation
        with torch.no_grad():
            la_flat_tensor = la_flat_tensor.detach()
            o_flat_tensor = l_p_input_from_buffer.detach()
            f_p_input_from_buffer = self.env_spec.get_inputs_for('follower', 'policy', 
                                                                    obs=o_flat_tensor, 
                                                                    leader_act=la_flat_tensor)
            fa_actor_dist, _ = self._hat_f_policy(f_p_input_from_buffer)
            l_q_exp_on_fa = torch.zeros_like(qval_of_samples)  # (batch_size, 1)
            for _ in range(self._batch_size_for_fa_exp):
                fa_for_exp = fa_actor_dist.sample().cpu().numpy()
                l_q_input_for_exp = self.env_spec.get_inputs_for('leader', 'qf', 
                                                                    obs=o_flat_tensor, 
                                                                    follower_act=fa_for_exp)
                l_q_exp_on_fa += self._qf(l_q_input_for_exp, la_flat_tensor)  # (batch_size, 1)
            baseline_for_benefit = l_q_exp_on_fa / self._batch_size_for_fa_exp
            benefit = qval_of_samples - baseline_for_benefit  # (batch_size, 1)
            benefit = benefit.squeeze(-1)  # (batch_size,)
            # for cumputing the guide_effect
            self._last_benefit.append(benefit.clone().detach())
            self._last_samples.append({'observation': o.copy(), 
                                        'action': fa.copy(), 
                                        'leader_action': la.copy()})

        # influence calculation
        influence = []
        if self.f_discount is None:
            self.f_discount = as_torch(
                [trainer.follower.discount ** i for i in range(self._max_episode_length)]
                ).unsqueeze(1)  # (max_episode_length, 1)
        hat_f_vals_log = []
        for subseq_o, subseq_la in zip(sampled_subsequences['observation'], 
                                        sampled_subsequences['leader_action']):
            l_p_input_subseq = self.env_spec.get_inputs_for('leader', 'policy', 
                                                            obs=subseq_o)
            la_subseq_dist, _ = self.policy(l_p_input_subseq)
            subseq_la = as_torch(subseq_la)
            log_pi_la_subseq = la_subseq_dist.log_prob(value=subseq_la).unsqueeze(-1)  # (subseq_len, 1)
            with torch.no_grad():
                subseq_o_flat_tensor = l_p_input_subseq
                subseq_f_q_input = self.env_spec.get_inputs_for('follower', 'qf',
                                                                obs=subseq_o_flat_tensor, 
                                                                leader_act=subseq_la)
                if self._wb_follower:
                    subseq_val = self._hat_f_vf(subseq_f_q_input)  # (subseq_len, 1)
                    hat_f_vals_log.append(subseq_val.mean().detach().item())
                    if self._use_advantage_in_influence:
                        baseline = torch.zeros_like(subseq_val)
                        for _ in range(self._batch_size_for_la_exp):
                            la_for_exp = la_subseq_dist.sample()
                            subseq_f_q_input_for_exp = self.env_spec.get_inputs_for('follower', 'qf',
                                                                                obs=subseq_o_flat_tensor, 
                                                                                leader_act=la_for_exp)
                            baseline += self._hat_f_vf(subseq_f_q_input_for_exp)
                        baseline /= self._batch_size_for_la_exp  # (subseq_len, 1)
                        subseq_val -= baseline  # (subseq_len, 1)
                else:
                    raise NotImplementedError('The follower Q-function is not implemented yet.')
                
            log_pi_la_val_subseq = log_pi_la_subseq * subseq_val  # (subseq_len, 1)

            if self._grad_mode == 'default':
                influence.append(
                    (self.f_discount[:len(log_pi_la_val_subseq)] * log_pi_la_val_subseq).sum()
                    )
            elif self._grad_mode == 'weighted_average':
                weight = self.f_discount[:len(log_pi_la_val_subseq)]
                influence.append((weight * log_pi_la_val_subseq).sum() / weight.sum())
            else:
                raise ValueError(f'Unknown grad_mode: {self._grad_mode}')
        influence = torch.stack(influence, dim=0)  # (batch_size,)
        actor_loss_2 = -1.0 / trainer.follower.beta * (benefit * influence).mean()

        # Compute the final actor loss
        actor_loss_1 = self.lambda_coef_1 * actor_loss_1
        actor_loss_2 = self.lambda_coef_2 * actor_loss_2
        actor_loss = actor_loss_1 + actor_loss_2

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
            grad1 = torch.cat(
                [g.flatten() for g in actor_loss_1_grads]
                ) if actor_loss_1_grads else torch.tensor([0.0])
            grad2 = torch.cat(
                [g.flatten() for g in actor_loss_2_grads]
                ) if actor_loss_2_grads else torch.tensor([0.0])
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

        if self._use_K_L2_regularization:
            if hasattr(self.policy, 'K') and isinstance(self.policy.K, nn.Parameter):
                grad_K_L2 = self._K_L2_reg_coef * 2.0 * self.policy.K  # (action_dim, obs_dim)
                for name, p in self.policy.named_parameters():
                    if name == 'K':
                        if p.grad is None:
                            p.grad = grad_K_L2
                        else:
                            p.grad += grad_K_L2

        self._policy_optimizer.step()

        info = dict(
            actor_loss_1=actor_loss_1.detach().item(),
            actor_loss_2=actor_loss_2.detach().item(),
            actor_loss_2_comp=influence.mean().detach().item(),
            actor_loss_2_benefit=benefit.mean().detach().item(),
            hat_f_val=float(np.mean(hat_f_vals_log)),
            policy_entropy=la_dist.entropy().mean().detach().item() if hasattr(la_dist, 'entropy') else 0.0,
            K_L2_norm=torch.norm(self.policy.K).item(),
            W_L2_norm=torch.norm(self.policy.W).item()
        )
        if self._grad_info:
            info.update(
                actor_loss_grad_norm=actor_loss_grad_norm.item(),
                actor_loss_1_grad_norm=actor_loss_1_grad_norm.item(),
                actor_loss_2_grad_norm=actor_loss_2_grad_norm.item(),
                grad_cosine_similarity=grad_cosine_similarity.item()
            )
        if hasattr(trainer.env, 'leader_policy_opt_gap'):
            policy_opt_gap = trainer.env.leader_policy_opt_gap(self.policy)
            info['policy_opt_gap'] = policy_opt_gap
        else:
            info['policy_opt_gap'] = 0.0

        return actor_loss.detach(), info
            
    def log_statistics(self, trainer, prefix='Leader'):
        tabular = trainer.leader_tabular
        with tabular.prefix(prefix + '/'):
            tabular.record('Policy/AveragePolicyLoss',
                            np.mean(self._episode_policy_losses))
            tabular.record('Polcy/AveragePolicyLossComp_1',
                            np.mean(self._actor_loss_1))
            tabular.record('Polcy/AveragePolicyLossComp_2',
                            np.mean(self._actor_loss_2))
            tabular.record('Polcy/AveragePolicyLossComp_2_wo_Benefit',
                            np.mean(self._actor_loss_2_comp))
            tabular.record('Polcy/AverageBenefit',
                            np.mean(self._actor_loss_2_benefit))
            
            if self._grad_info:
                tabular.record('Policy/AveragePolicyGrad_L2Norm',
                                np.mean(self._actor_loss_grad_norm))
                tabular.record('Policy/AveragePolicyGrad_1_L2Norm',
                                np.mean(self._actor_loss_1_grad_norm))
                tabular.record('Policy/AveragePolicyGrad_2_L2Norm',
                                np.mean(self._actor_loss_2_grad_norm))
                tabular.record('Policy/AveragePolicyGradCosineSimilarity',
                                np.mean(self._grad_cosine_similarity))
                
            tabular.record('Policy/AverageSubseqLength',
                            np.mean(self._average_subseq_len))
            tabular.record('Policy/AverageHatFQVal',
                            np.mean(self._average_hat_f_val))
            tabular.record('Policy/PolicyOptimalityGap', 
                            np.mean(self._policy_opt_gaps))
            tabular.record('Policy/PolicyEntropy',
                            np.mean(self._policy_entropy))
            tabular.record('Policy/K_L2_Norm',
                            np.mean(self._K_L2_norm))
            tabular.record('Policy/W_L2_Norm',
                            np.mean(self._W_L2_norm))
            tabular.record('AverageRealizedGuidanceEffect',
                           np.mean(self._realized_guidance_effect))
            tabular.record('MaxRealizedGuidanceEffect',
                           np.max(self._realized_guidance_effect))
            tabular.record('MinRealizedGuidanceEffect',
                           np.min(self._realized_guidance_effect))

            tabular.record('QFunction/AverageQFunctionLoss',
                            np.mean(self._episode_qf_losses))
            tabular.record('QFunction/AverageQ', 
                            np.mean(self._epoch_qs))
            tabular.record('QFunction/MaxQ', 
                            np.max(self._epoch_qs))
            tabular.record('QFunction/AverageAbsQ',
                            np.mean(np.abs(self._epoch_qs)))
            tabular.record('QFunction/AverageY', 
                            np.mean(self._epoch_ys))
            tabular.record('QFunction/MaxY', 
                            np.max(self._epoch_ys))
            tabular.record('QFunction/AverageAbsY',
                            np.mean(np.abs(self._epoch_ys)))
            tabular.record('TrainAverageTargetReturn', 
                            np.mean(self._episode_rewards))
            
            tabular.record('K_1', self.K[0,0].item())
            tabular.record('K_2', self.K[0,1].item())
            tabular.record('W', self.W[0,0].item())
        
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
        self._policy_opt_gaps = []
        self._realized_guidance_effect = []
        self._episode_rewards = []        
        self._average_hat_f_val = []
        self._policy_entropy = []
        self._K_L2_norm = []
        self._W_L2_norm = []

    
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
            follower_acts = as_torch(follower_act_list).view(-1, 1) # (N, 1)

            f_policy_input = self.env_spec.get_inputs_for('follower', 'policy', 
                                                          obs=observations, leader_act=leader_acts)
            
            if not hasattr(self, '_last_follower_policy') or self._last_follower_policy is None:
                 self._last_benefit = []
                 self._last_samples = []
                 return float('nan')
            
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

        return corr

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
        
    @property
    def K(self):
        """Leader's policy gain matrix."""
        if isinstance(self.policy, LinearGaussianPolicy):
            return self.policy.K
        else:
            raise ValueError('The policy is not LinearGaussianPolicy.')

    @property
    def W(self):
        """Square root of Leader's policy covariance matrix."""
        if isinstance(self.policy, LinearGaussianPolicy):
            return self.policy.W
        else:
            raise ValueError('The policy is not LinearGaussianPolicy.')

    def get_policy_log_prob_gradient(self, observations, actions):
        """
        Args:
            observations (torch.Tensor): (batch_size, obs_dim) batch of observations.
            actions (torch.Tensor): (batch_size, act_dim) batch of actions.
        """
        with torch.no_grad():
            W = self.policy.W
            K = self.policy.K
            if not W.shape == (1,1):
                raise NotImplementedError(
                    'Currently, only LinearGaussianPolicy with 1D action is supported.'
                    )
            Sigma = W @ W.T  # (act_dim, act_dim)
            L = torch.cholesky(Sigma)
            Sigma_inv = torch.cholesky_inverse(L)  # (act_dim, act_dim)
            grad_K = []
            grad_W = []
            for o, a in zip(observations, actions):
                o_vec = o.unsqueeze(-1)  # (obs_dim, 1)
                mu = self.K @ o_vec  # (act_dim, 1)
                dev = a.unsqueeze(-1) - mu  # (act_dim, 1)
                nabla_mu_log_pi = Sigma_inv @ dev  # (act_dim, 1)
                nabla_Sigma_log_pi = 0.5 * (Sigma_inv @ dev @ dev.T @ Sigma_inv - Sigma_inv)  # (act_dim, act_dim)
                nabla_K_log_pi = nabla_mu_log_pi @ o_vec.T  # (act_dim, obs_dim)
                nabla_W_log_pi = 2.0 * W * nabla_Sigma_log_pi  # (act_dim, act_dim)
                grad_K.append(nabla_K_log_pi)
                grad_W.append(nabla_W_log_pi)
            grad_K = torch.stack(grad_K, dim=0)  # (batch_size, act_dim, obs_dim)
            grad_W = torch.stack(grad_W, dim=0)  # (batch_size, act_dim, act_dim)
            return {'K': grad_K, 'W': grad_W}
        
    def optimize_policy_with_closed_form_gradient(self, samples_data, sampled_subsequences, trainer):
        """Perform algorithm optimizing.

        Args:
            samples_data (dict): Sampled data for optimization.
            trainer (Trainer): Current trainer.
            sampled_subsequences (dict): Sampled subsequences for optimization.
            update_actor (bool): Whether to update the actor (leader policy).

        Returns:
            action_loss: Loss of action predicted by the policy network.
            qval_loss: Loss of Q-value predicted by the Q-network.
            ys: y_s.
            qval: Q-value predicted by the Q-network.
        """
        # Note: samples_data is a dict of numpy arrays.
        o = samples_data['observation']
        r = samples_data['target_reward'].reshape(-1, 1)
        fa = samples_data['action']
        la = samples_data['leader_action']
        n_o = samples_data['next_observation']
        d = samples_data['terminal'].reshape(-1, 1)

        # --- Actor (Leader's Policy) Optimization --- #

        actor_loss_1 = torch.tensor([0.0])
        info = dict()
        gradient = {'K': torch.zeros_like(self.policy.K), 'W': torch.zeros_like(self.policy.W)}

        # --- First term --- #

        l_q_input_from_buffer = self.env_spec.get_inputs_for('leader', 'qf', 
                                                             obs=o, 
                                                             follower_act=fa)
        l_p_input_from_buffer = self.env_spec.get_inputs_for('leader', 'policy', 
                                                                obs=o)
        la_dist, _ = self.policy(l_p_input_from_buffer)
        la_flat_tensor = as_torch(self.env_spec.leader_action_space.flatten_n(la))
        nabla_log_pi_la_samples = self.get_policy_log_prob_gradient(
            l_p_input_from_buffer, la_flat_tensor
            )  # 'K': (batch_size, act_dim, obs_dim), 'W': (batch_size, act_dim, act_dim)
        with torch.no_grad():
            qval_of_samples = self._qf(l_q_input_from_buffer, la_flat_tensor)  # (batch_size, 1)
            if self._use_advantage:
                v_actor = torch.zeros_like(qval_of_samples)  # (batch_size, 1)
                for _ in range(self._batch_size_for_la_exp):
                    la_for_exp = la_dist.sample()
                    v_actor = v_actor + self._qf(l_q_input_from_buffer, 
                                                la_for_exp)
                v_actor = v_actor / self._batch_size_for_la_exp
                advantage = qval_of_samples - v_actor  # (batch_size, 1)
            else:
                advantage = qval_of_samples  # (batch_size, 1)
            advantage = advantage.unsqueeze(-1)  # (batch_size, 1, 1)
            grad_K_1 = (nabla_log_pi_la_samples['K'] * advantage).mean(dim=0)  # (act_dim, obs_dim)
            grad_W_1 = (nabla_log_pi_la_samples['W'] * advantage).mean(dim=0)  # (act_dim, act_dim)
            gradient['K'] += self.lambda_coef_1 * grad_K_1
            gradient['W'] += self.lambda_coef_1 * grad_W_1

        if self._no_guidance:
            self._policy_optimizer.zero_grad()
            for name, p in self.policy.named_parameters():
                if p.grad is None:
                    p.grad = (gradient[name]).clone()
                else:
                    p.grad.copy_(gradient[name])

            actor_loss_grad_norm = torch.tensor(0.0)
            if self._grad_info:
                actor_loss_grads = [
                    param.grad.clone().detach() for param in self.policy.parameters() 
                    if param.grad is not None
                    ]
                actor_loss_grad_norm = torch.sqrt(
                    sum(torch.sum(g ** 2) for g in actor_loss_grads)
                    ) if actor_loss_grads else torch.tensor(0.0)
                
            if self._use_K_L2_regularization:
                if hasattr(self.policy, 'K') and isinstance(self.policy.K, nn.Parameter):
                    grad_K_L2 = self._K_L2_reg_coef * 2.0 * self.policy.K  # (action_dim, obs_dim)
                    for name, p in self.policy.named_parameters():
                        if name == 'K':
                            if p.grad is None:
                                p.grad = grad_K_L2
                            else:
                                p.grad += grad_K_L2
                
            self._policy_optimizer.step()

            policy_entropy = la_dist.entropy().mean().detach().item() if hasattr(la_dist, 'entropy') else 0.0
            info = dict(
                actor_loss_1=0.0,
                actor_loss_2=0.0,
                actor_loss_2_comp=0.0,
                actor_loss_2_benefit=0.0,
                hat_f_val=0.0,
                policy_entropy=policy_entropy,
                K_L2_norm=torch.norm(self.policy.K).item(),
                W_L2_norm=torch.norm(self.policy.W).item()
            )
            if self._grad_info:
                info.update(
                    actor_loss_grad_norm=actor_loss_grad_norm.item(),
                    actor_loss_1_grad_norm=actor_loss_grad_norm.item(),
                    actor_loss_2_grad_norm=0.0,
                    grad_cosine_similarity=0.0
                )
            if hasattr(trainer.env, 'leader_policy_opt_gap'):
                policy_opt_gap = trainer.env.leader_policy_opt_gap(self.policy)
                info['policy_opt_gap'] = policy_opt_gap
            else:
                info['policy_opt_gap'] = 0.0

            return actor_loss_1.detach(), info

        # --- Second term --- #

        # Benefit calculation
        with torch.no_grad():
            la_flat_tensor = la_flat_tensor.detach()
            o_flat_tensor = l_p_input_from_buffer.detach()
            f_p_input_from_buffer = self.env_spec.get_inputs_for('follower', 'policy', 
                                                                    obs=o_flat_tensor, 
                                                                    leader_act=la_flat_tensor)
            fa_actor_dist, _ = self._hat_f_policy(f_p_input_from_buffer)
            l_q_exp_on_fa = torch.zeros_like(qval_of_samples)  # (batch_size, 1)
            for _ in range(self._batch_size_for_fa_exp):
                fa_for_exp = fa_actor_dist.sample().cpu().numpy()
                l_q_input_for_exp = self.env_spec.get_inputs_for('leader', 'qf', 
                                                                    obs=o_flat_tensor, 
                                                                    follower_act=fa_for_exp)
                l_q_exp_on_fa += self._qf(l_q_input_for_exp, la_flat_tensor)  # (batch_size, 1)
            baseline_for_benefit = l_q_exp_on_fa / self._batch_size_for_fa_exp
            benefit = qval_of_samples - baseline_for_benefit  # (batch_size, 1)
            benefit = benefit.unsqueeze(-1)  # (batch_size, 1, 1)
            # for cumputing the guide_effect
            self._last_benefit.append(benefit.clone().detach())
            self._last_samples.append({'observation': o.copy(), 
                                        'action': fa.copy(), 
                                        'leader_action': la.copy()})

            # influence calculation
            influence_K = []
            influence_W = []
            if self.f_discount is None:
                self.f_discount = as_torch(
                    [trainer.follower.discount ** i for i in range(self._max_episode_length)]
                    ).unsqueeze(-1).unsqueeze(-1)  # (max_episode_length, 1, 1)
            hat_f_vals_log = []
            for subseq_o, subseq_la in zip(sampled_subsequences['observation'], 
                                            sampled_subsequences['leader_action']):
                l_p_input_subseq = self.env_spec.get_inputs_for('leader', 'policy', 
                                                                obs=subseq_o)
                subseq_la_flat_tensor = as_torch(self.env_spec.leader_action_space.flatten_n(subseq_la))
                nabla_log_pi_la_subseq = self.get_policy_log_prob_gradient(
                    l_p_input_subseq, subseq_la_flat_tensor
                )  # 'K': (subseq_len, act_dim, obs_dim), 'W': (subseq_len, act_dim, act_dim)

                la_subseq_dist, _ = self.policy(l_p_input_subseq)
                with torch.no_grad():
                    subseq_o_flat_tensor = l_p_input_subseq
                    subseq_f_q_input = self.env_spec.get_inputs_for('follower', 'qf',
                                                                    obs=subseq_o_flat_tensor, 
                                                                    leader_act=subseq_la_flat_tensor)
                    if self._wb_follower:
                        subseq_val = self._hat_f_vf(subseq_f_q_input)  # (subseq_len, 1)
                        hat_f_vals_log.append(subseq_val.mean().detach().item())
                        if self._use_advantage_in_influence:
                            baseline = torch.zeros_like(subseq_val)
                            for _ in range(self._batch_size_for_la_exp):
                                la_for_exp = la_subseq_dist.sample()
                                subseq_f_q_input_for_exp = self.env_spec.get_inputs_for('follower', 'qf',
                                                                                    obs=subseq_o_flat_tensor, 
                                                                                    leader_act=la_for_exp)
                                baseline += self._hat_f_vf(subseq_f_q_input_for_exp)
                            baseline /= self._batch_size_for_la_exp  # (subseq_len, 1)
                            subseq_val -= baseline  # (subseq_len, 1)
                    else:
                        raise NotImplementedError('The follower Q-function is not implemented yet.')

                grad_subseq_K = nabla_log_pi_la_subseq['K'] * subseq_val.unsqueeze(-1)  # (subseq_len, act_dim, obs_dim)
                grad_subseq_W = nabla_log_pi_la_subseq['W'] * subseq_val.unsqueeze(-1)  # (subseq_len, act_dim, act_dim)
                subseq_length = grad_subseq_K.shape[0]

                if self._grad_mode == 'default':
                    influence_K.append(
                        (self.f_discount[:subseq_length] * grad_subseq_K).sum(dim=0)  # (act_dim, obs_dim)
                        )
                    influence_W.append(
                        (self.f_discount[:subseq_length] * grad_subseq_W).sum(dim=0)  # (act_dim, act_dim)
                        )
                elif self._grad_mode == 'weighted_average':
                    weight = self.f_discount[:subseq_length]
                    influence_K.append((weight * grad_subseq_K).sum(dim=0) / weight.sum())
                    influence_W.append((weight * grad_subseq_W).sum(dim=0) / weight.sum())
                else:
                    raise ValueError(f'Unknown grad_mode: {self._grad_mode}')
            influence_K = torch.stack(influence_K, dim=0)  # (batch_size, act_dim, obs_dim)
            influence_W = torch.stack(influence_W, dim=0)  # (batch_size, act_dim, act_dim)
            grad_K_2 = 1.0 / trainer.follower.beta * (benefit * influence_K).mean(dim=0)  # (act_dim, obs_dim)
            grad_W_2 = 1.0 / trainer.follower.beta * (benefit * influence_W).mean(dim=0)  # (act_dim, act_dim)
            gradient['K'] += self.lambda_coef_2 * grad_K_2
            gradient['W'] += self.lambda_coef_2 * grad_W_2

        if self._grad_info:
            # Compute gradients for actor_loss_1
            actor_loss_1_grads = [grad_K_1.detach().clone(), grad_W_1.detach().clone()]
            # Compute gradients for actor_loss_2
            actor_loss_2_grads = [grad_K_2.detach().clone(), grad_W_2.detach().clone()]
            # Compute norms and inner product
            grad1 = torch.cat(
                [g.flatten() for g in actor_loss_1_grads]
                ) if actor_loss_1_grads else torch.tensor([0.0])
            grad2 = torch.cat(
                [g.flatten() for g in actor_loss_2_grads]
                ) if actor_loss_2_grads else torch.tensor([0.0])
            grad_sum = grad1 + grad2
            actor_loss_1_grad_norm = grad1.norm()
            actor_loss_2_grad_norm = grad2.norm()
            actor_loss_grad_norm = grad_sum.norm()
            if actor_loss_1_grad_norm > 0 and actor_loss_grad_norm > 0:
                grad_cosine_similarity = (
                    torch.dot(grad1, grad_sum)) / (actor_loss_1_grad_norm * actor_loss_grad_norm)
            else:
                grad_cosine_similarity = torch.tensor(0.0)

        self._policy_optimizer.zero_grad()
        for name, p in self.policy.named_parameters():
            if p.grad is None:
                p.grad = (gradient[name]).clone()
            else:
                p.grad.copy_(gradient[name])
            
        if self._max_policy_grad_norm is not None:
            # Clip the gradients of the policy network
            utils.clip_grad_norm_(self.policy.parameters(), self._max_policy_grad_norm)

        if self._use_K_L2_regularization:
            if hasattr(self.policy, 'K') and isinstance(self.policy.K, nn.Parameter):
                grad_K_L2 = self._K_L2_reg_coef * 2.0 * self.policy.K  # (action_dim, obs_dim)
                for name, p in self.policy.named_parameters():
                    if name == 'K':
                        if p.grad is None:
                            p.grad = grad_K_L2
                        else:
                            p.grad += grad_K_L2

        self._policy_optimizer.step()

        info = dict(
            actor_loss_1=0.0,
            actor_loss_2=0.0,
            actor_loss_2_comp=0.0,
            actor_loss_2_benefit=benefit.mean().detach().item(),
            hat_f_val=float(np.mean(hat_f_vals_log)),
            policy_entropy=la_dist.entropy().mean().detach().item() if hasattr(la_dist, 'entropy') else 0.0,
            K_L2_norm=torch.norm(self.policy.K).item(),
            W_L2_norm=torch.norm(self.policy.W).item()
        )
        if self._grad_info:
            info.update(
                actor_loss_grad_norm=actor_loss_grad_norm.item(),
                actor_loss_1_grad_norm=actor_loss_1_grad_norm.item(),
                actor_loss_2_grad_norm=actor_loss_2_grad_norm.item(),
                grad_cosine_similarity=grad_cosine_similarity.item()
            )
        if hasattr(trainer.env, 'leader_policy_opt_gap'):
            policy_opt_gap = trainer.env.leader_policy_opt_gap(self.policy)
            info['policy_opt_gap'] = policy_opt_gap
        else:
            info['policy_opt_gap'] = 0.0

        return actor_loss_1.detach(), info
