import numpy as np
import torch
from collections import deque
from garage.torch import global_device
from garage.torch.policies.stochastic_policy import StochasticPolicy

class FollowerWrapper:    
    def __init__(self, 
                 algo,
                 init_steps=None,
                 fixed_policy=False,
                 **kwargs):
        self.algo = algo(**kwargs)
        self.algo_name = algo.__name__
        self.fixed_policy = fixed_policy
        self.is_deterministic_policy = not isinstance(self.algo.policy, StochasticPolicy)
        self._init_steps = init_steps if init_steps is not None else -1

        self.stats = {}

    def make_q_function(self):
        if self.algo_name in ['SAC']:
            def q_function(observation: torch.Tensor, action: torch.Tensor):
                networks = self.algo.networks
                qf1, qf2 = networks[1], networks[2]
                # Return q value used for computing follower's policy gradient
                q_val = torch.min(qf1(observation, action),
                                  qf2(observation, action))
                return q_val  # (batch_size, 1)
        elif self.algo_name in ['SACDiscrete']:
            def q_function(observation: torch.Tensor, action: torch.Tensor):
                networks = self.algo.networks
                qf1, qf2 = networks[1], networks[2]
                q_val = torch.min(qf1(observation), qf2(observation))  # (batch_size, num_actions)
                action = torch.argmax(action, dim=-1).unsqueeze(-1)  # (batch_size, 1)
                q_val = q_val.gather(1, action)  # (batch_size, 1)
                return q_val  # (batch_size, 1)
        elif self.algo_name in ['SoftQIteration', 'SoftQIterationSubopt']:
            def q_function(observation: torch.Tensor, action: torch.Tensor = None):
                state = observation[:, :self.algo.num_states]
                state = torch.argmax(state, dim=-1).long()
                leader_action = observation[:, self.algo.num_states:]
                leader_action = torch.argmax(leader_action, dim=-1).long()
                q_val = []
                if action is None:
                    for s, la in zip(state, leader_action):
                        q_val.append(self.algo.Q[s, la])
                    q_val = torch.stack(q_val, dim=0)  # (batch_size, num_actions)
                else:
                    action = action.long()
                    for s, la, fa in zip(state, leader_action, action):
                        q_val.append(self.algo.Q[s, la, fa])
                    q_val = torch.stack(q_val, dim=0)
                return q_val  # (batch_size, 1)
        elif self.algo_name in ['MaxEntLQR']:
            # def q_function(observation: torch.Tensor, action: torch.Tensor):
            #     q_val = []
            #     for obs, act in zip(observation, action):
            #         q_val.append(self.algo.action_value(obs, act))
            #     q_val = torch.stack(q_val, dim=0).unsqueeze(-1)
            #     return q_val  # (batch_size, 1)
            def q_function(observation: torch.Tensor, action: torch.Tensor):
                return -1.0 * self.algo.action_value_batch(observation, action)  # (batch_size, 1)
        else:
            raise NotImplementedError(f"Algorithm {self.algo_name} not supported.")
        return q_function
    
    def make_value_function(self):
        if self.algo_name in ['SAC']:  # NOT recoomended to use
            def value_function(observation: torch.Tensor, n_samples: int = 32):
                networks = self.algo.networks
                qf1 = networks[1]
                action = np.zeros((n_samples, *self.algo.env_spec.action_space.shape)) # (n_samples, action_dim)
                for i in range(n_samples):
                    action[i] = self.algo.env_spec.action_space.sample()
                action = torch.as_tensor(action, dtype=torch.float32).to(global_device())
                repeated_action = action.repeat(observation.shape[0], 1)  # (batch_size*n_samples, action_dim)
                repeated_obs = []
                for i in range(observation.shape[0]):
                    repeated_obs.append(observation[i].repeat(n_samples, 1))
                repeated_obs = torch.cat(repeated_obs, dim=0)  # (batch_size*n_samples, obs_dim)
                qval = qf1(repeated_obs, repeated_action)  # (batch_size*n_samples, 1)
                val = []
                for i in range(observation.shape[0]):
                    start = i * n_samples
                    end = (i + 1) * n_samples
                    v = self.beta * torch.logsumexp(qval[start:end] / self.beta, dim=0)  # (1,)
                    val.append(v)
                val = torch.stack(val, dim=0)  # (batch_size, 1)
                return val  # (batch_size, 1)
        elif self.algo_name in ['SACDiscrete']:
            def value_function(observation: torch.Tensor):
                networks = self.algo.networks
                qf1, qf2 = networks[1], networks[2]
                q_val = torch.min(qf1(observation), qf2(observation))  # (batch_size, num_actions)
                val = self.beta * torch.logsumexp(q_val / self.beta, dim=-1)  # (batch_size,)
                val = val.unsqueeze(-1)  # (batch_size, 1)
                return val  # (batch_size, 1)
        elif self.algo_name in ['SoftQIteration', 'SoftQIterationSubopt']:
            def value_function(observation: torch.Tensor):
                state = observation[:, :self.algo.num_states]
                state = torch.argmax(state, dim=-1).long()
                leader_action = observation[:, self.algo.num_states:]
                leader_action = torch.argmax(leader_action, dim=-1).long()
                val = self.algo.V[state, leader_action]
                return val.unsqueeze(-1)  # (batch_size, 1)
        elif self.algo_name in ['MaxEntLQR']:
            # def value_function(observation: torch.Tensor):
            #     val = []
            #     for obs in observation:
            #         val.append(self.algo.state_value(obs))
            #     val = torch.stack(val, dim=0).unsqueeze(-1)  # (batch_size, 1)
            #     return val  # (batch_size, 1)
            def value_function(observation: torch.Tensor):
                return -1.0 * self.algo.state_value_batch(observation)  # (batch_size, 1)
        else:
            raise NotImplementedError(f"Algorithm {self.algo_name} not supported.")
        return value_function

    def log_statistics(self, *args, **kwargs):
        if self.algo_name in ['SAC', 'SACDiscrete', 'SoftQIteration', 'SoftQIterationSubopt', 'MaxEntLQR']:
            self.algo.episode_rewards = np.array(self.stats['episode_rewards'])
            self.algo._log_statistics(*args, **kwargs)
        else:
            raise NotImplementedError(f"Algorithm {self.algo_name} not supported.")

    def __setstate__(self, state):
        """Set state."""
        self.__dict__.update(state)
        self.to()

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped algo."""
        if name in self.__dict__:
            return getattr(self, name)
        return getattr(self.algo, name)

    @property
    def policy(self):
        if self.algo_name in ['SAC', 'SACDiscrete', 'SoftQIteration', 'SoftQIterationSubopt', 'MaxEntLQR']:
            return self.algo.policy
        else:
            raise NotImplementedError(f"Algorithm {self.algo_name} not supported.")
    
    @property
    def policy_optimizer(self):
        return self.algo._policy_optimizer
    
    @property
    def beta(self) -> float:
        if self.algo_name in ['SAC', 'SACDiscrete']:
            return self.algo._log_alpha.exp().item()
        elif self.algo_name in ['SoftQIteration', 'SoftQIterationSubopt']:
            return self.algo._temperature
        elif self.algo_name == 'MaxEntLQR':
            return self.algo._beta
        else:
            raise NotImplementedError(f"Algorithm {self.algo_name} not supported.")
    
    @property
    def discount(self):
        return self.algo._discount
    
    
    def q_function(self, observation: torch.Tensor, action: torch.Tensor = None):# MARK: あとで消す
        if self.algo_name in ['SAC']:
            networks = self.algo.networks
            qf1, qf2 = networks[1], networks[2]
            # Return q value used for computing follower's policy gradient
            q_val = torch.min(qf1(observation, action),
                              qf2(observation, action))
        elif self.algo_name in ['SACDiscrete']:
            networks = self.algo.networks
            qf1, qf2 = networks[1], networks[2]
            q_val = torch.min(qf1(observation), qf2(observation))  # (batch_size, num_actions)
            action = torch.argmax(action, dim=-1).unsqueeze(-1)  # (batch_size, 1)
            q_val = q_val.gather(1, action).squeeze(-1)  # (batch_size,)
        elif self.algo_name in ['SoftQIteration', 'SoftQIterationSubopt']:
            state = observation[:, :self.algo.num_states]
            state = torch.argmax(state, dim=-1).long()
            leader_action = observation[:, self.algo.num_states:]
            leader_action = torch.argmax(leader_action, dim=-1).long()
            q_val = []
            if action is None:
                for s, la in zip(state, leader_action):
                    q_val.append(self.algo.Q[s, la])
            else:
                action = action.long()
                for s, la, fa in zip(state, leader_action, action):
                    q_val.append(self.algo.Q[s, la, fa])
            q_val = torch.stack(q_val, dim=0)
        else:
            raise NotImplementedError(f"Algorithm {self.algo_name} not supported.")
        return q_val
    
    def value_function(self, observation: torch.Tensor):# MARK: あとで消す
        if self.algo_name in ['SACDiscrete']:
            networks = self.algo.networks
            qf1, qf2 = networks[1], networks[2]
            q_val = torch.min(qf1(observation), qf2(observation))  # (batch_size, num_actions)
            val = self.beta * torch.logsumexp(q_val / self.beta, dim=-1)  # (batch_size,)
        elif self.algo_name in ['SoftQIteration', 'SoftQIterationSubopt']:
            state = observation[:, :self.algo.num_states]
            state = torch.argmax(state, dim=-1).long()
            leader_action = observation[:, self.algo.num_states:]
            leader_action = torch.argmax(leader_action, dim=-1).long()
            val = self.algo.V[state, leader_action]
        else:
            raise NotImplementedError(f"Algorithm {self.algo_name} not supported.")
        return val

