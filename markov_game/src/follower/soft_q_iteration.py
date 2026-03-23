"""
Soft Q-Iteration (entropy-regularized value iteration) for discrete MDPs.
Reference: "Reinforcement Learning with Deep Energy-Based Policies" (Haarnoja et al., 2017)
"""
import numpy as np
import torch
from akro import Discrete
from collections import deque
from garage.np.algos import RLAlgorithm
from garage.torch import global_device

from dowel import tabular

from ..policies import TabularCategoricalPolicy

class SoftQIteration(RLAlgorithm):
    """
    Soft Q-Iteration for discrete state/action spaces with deterministic transitions.
    """
    name = 'SoftQIteration'
    
    def __init__(
            self,
            env,
            *,
            discount=0.99,
            temperature: float = 1.0,
            max_iterations=1000,
            tol=1e-6,):

        if hasattr(env, 'transition_fn'):
            self.transition_fn = env.transition_fn
        else:
            raise NotImplementedError("env must have a transition_fn.")
        if hasattr(env, 'reward_fn'):
            self.reward_fn = env.reward_fn
        else:
            raise NotImplementedError("env must have a reward_fn.")
        
        self._discount = discount
        self._temperature = float(temperature)
        self._max_iterations = max_iterations
        self._tol = tol

        self.env_spec = env.spec  # must be GlobalEnvSpec
        
        if not hasattr(self.env_spec, 'leader_action_space'):
            raise NotImplementedError("env_spec must be GlobalEnvSpec.")
        if not isinstance(self.env_spec.observation_space, Discrete):
            raise NotImplementedError("Unsupported observation space type: {}".format(type(self.env_spec.observation_space)))
        if not isinstance(self.env_spec.action_space, Discrete):
            raise NotImplementedError("Unsupported action space type: {}".format(type(self.env_spec.action_space)))
        if not isinstance(self.env_spec.leader_action_space, Discrete):
            raise NotImplementedError("Unsupported leader action space type: {}".format(type(self.env_spec.leader_action_space)))
        
        self.num_states = self.env_spec.observation_space.n
        self.num_actions = self.env_spec.action_space.n
        self.num_leader_actions = self.env_spec.leader_action_space.n
        self.Q = torch.zeros(self.num_states, self.num_leader_actions, self.num_actions)  # Q(s, a_l, a_f)
        # V(s, a_l) = beta * logsumexp(Q(s, a_l, a_f) / beta, dim=2)
        self.V = torch.zeros(self.num_states, self.num_leader_actions)  # V(s, a_l)
        self.policy = TabularCategoricalPolicy(
            env_spec=self.env_spec, policy_matrix=self.policy_matrix()
        )

        self.episode_rewards = deque(maxlen=30)  # retains episodes sampled in the leader's iteration

        self.to(global_device())

    def train(self, trainer):
        for i in range(self._max_iterations):
            max_diff, info = self.train_once(trainer)
            info['converged'] = False
            if max_diff < self._tol:
                info['converged'] = True
                break
        info['iteration'] = i + 1
        self.V = self.soft_value_operator(self.Q)
        self.update_policy()
        return max_diff, info
    
    def train_once(self, trainer):
        # 1 step soft Q-iteration
        with torch.no_grad():
            Q_prev = self.Q.clone()
            Q_new = torch.zeros_like(self.Q)
            for s in range(self.num_states):
                for a_l in range(self.num_leader_actions):
                    for a_f in range(self.num_actions):
                        next_state = self.transition_fn(s, a_l, a_f)
                        next_a_l_probs = trainer.leader.policy(
                            self.env_spec.get_input_for('leader', 'policy', obs=next_state)
                        )[1]['probs']
                        reward = self.reward_fn(s, a_l, a_f)
                        next_val = self.soft_value_operator(torch.matmul(next_a_l_probs, Q_prev[next_state]))
                        Q_new[s, a_l, a_f] = reward + self._discount * next_val
            self.Q = Q_new
            max_diff = torch.max(torch.abs(self.Q - Q_prev)).item()
        return max_diff, {}

    def soft_value_operator(self, Q):
        beta = self._temperature
        return beta * torch.logsumexp(Q / beta, dim=-1)

    def update_policy(self):
        self.policy.set_policy_matrix(self.policy_matrix())

    def policy_matrix(self):
        beta = self._temperature
        return torch.softmax(self.Q / beta, dim=-1)

    def get_value(self):
        return self.V.cpu().numpy()

    def get_q(self):
        return self.Q.cpu().numpy()

    def _log_statistics(self, max_diff, info, prefix='Follower'):
        with torch.no_grad():
            with tabular.prefix(prefix + '/'):
                tabular.record('MaxDiff', max_diff)
                tabular.record('MeanValue', self.V.mean().item())
                tabular.record('MaxValue', self.V.max().item())
                tabular.record('MinValue', self.V.min().item())
                tabular.record('MeanQ', self.Q.mean().item())
                tabular.record('MaxQ', self.Q.max().item())
                tabular.record('MinQ', self.Q.min().item())
                if 'converged' in info:
                    tabular.record('Converged', info['converged'])
                    tabular.record('Iterations', info['iteration'])
                tabular.record('Policy/Entropy', self.policy.get_entropy().mean().item())
                tabular.record('TrainAverageReturn',
                               np.mean(self.episode_rewards))

    @property
    def networks(self):
        return []  # Empty because this is tabular

    def to(self, device=None):
        if device is None:
            device = global_device()
        self.Q = self.Q.to(device)
        self.V = self.V.to(device)
        self.policy.policy_matrix = self.policy.policy_matrix.to(device)
