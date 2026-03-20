"""Joint Policy of leader's and follower's policies."""
import warnings
import numpy as np
import torch
from garage.torch.policies import Policy
from garage.torch.policies.stochastic_policy import StochasticPolicy
from garage.torch import global_device

from .categorical_mlp_policy import CategoricalMLPPolicy
from .tabular_categorical_policy import TabularCategoricalPolicy

class JointPolicy(Policy):
    """Joint Policy of leader's and follower's policies.

    Args:
        env_spec (EnvSpec): Environment specification.
        leader_policy (garage.Policy): Leader's policy.
        follower_policy (garage.Policy): Follower's policy.
        name (str): Policy name, also the name of the :class:`Module`.
        noise_sigma (float): Standard deviation of Gaussian noise added to
                             leader's and follower's actions if they are
                             deterministic policies. If None, no noise is added.
    """
    
    def __init__(self,
                 env_spec,
                 leader_policy, 
                 follower_policy, 
                 name='JointPolicy',
                 noise_sigma_l=None,
                 noise_sigma_f=None
                 ):
        super().__init__(env_spec, name)
        self._l_policy = leader_policy
        self._f_policy = follower_policy
        self._l_env_spec = self._env_spec.leader_policy_env_spec
        self._f_env_spec = self._env_spec.follower_policy_env_spec

        self.noise_sigma_l = noise_sigma_l
        self.noise_sigma_f = noise_sigma_f

    @property
    def env_spec(self):
        return self._env_spec

    def get_action(self, observation, 
                   deterministic_leader=False, deterministic_follower=False,
                   explorate_leader=False, explorate_follower=False, 
                   zero_leader=False, zero_follower=False):
        """Get unflattened action sampled from the policy.
        Args:
            observation (np.ndarray): Observation from the environment.
        Returns:
            Tuple[np.ndarray, dict[str,np.ndarray]]: Action and extra agent info.
        """
        with torch.no_grad():
            leader_follower_actions, agent_infos = self.get_actions(
                [observation], 
                deterministic_leader,
                deterministic_follower,
                explorate_leader,
                explorate_follower,
                zero_leader,
                zero_follower
                )
            actions = [a[0] for a in leader_follower_actions]
            return actions, {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations, 
                    deterministic_leader=False, deterministic_follower=False,
                    explorate_leader=False, explorate_follower=False, 
                    zero_leader=False, zero_follower=False):
        """Get unflattened actions given observations.
        Args:
            observations (np.ndarray): Minibatch of the observations from the environment.
                                       (observations.shape[0] = batch_size)
        Returns:
            Tuple[np.ndarray, dict[str,np.ndarray]]: Actions and extra agent infos.
        """
        la, la_infos = self._l_policy.get_actions(observations)

        if zero_leader:
            if isinstance(la, (np.ndarray, np.integer)):
                la = np.zeros_like(la)
            else:
                if isinstance(la, torch.Tensor):
                    la = torch.zeros_like(la)
                else:
                    raise ValueError(
                        'The action of the leader should be either a numpy array or a torch tensor.'
                    )
        elif explorate_leader:
            for i in range(len(la)):
                la[i] = self._l_env_spec.action_space.sample()
        elif isinstance(self._l_policy, CategoricalMLPPolicy):
            if deterministic_leader:
                la = la_infos['mode']
        elif isinstance(self._l_policy, StochasticPolicy):
            if deterministic_leader:
                la = la_infos['mean']
        elif isinstance(self._l_policy, Policy):  # Deterministic policy
            if not deterministic_leader:
                la_infos['mean'] = np.copy(la)
                for itr, _ in enumerate(la):
                    la[itr] = np.clip(
                        la[itr] + np.random.normal(size=la[itr].shape) * self.noise_sigma_l,
                        self._l_env_spec.action_space.low, self._l_env_spec.action_space.high)
        else:
            raise NotImplementedError            

        f_observation = self._env_spec.get_inputs_for('follower', 'policy',
                                                     obs=observations,
                                                     leader_act=la)
        fa, fa_infos = self._f_policy.get_actions(f_observation)
        
        if zero_follower:
            if isinstance(fa, (np.ndarray, np.integer)):
                fa = np.zeros_like(fa)
            else:
                if isinstance(fa, torch.Tensor):
                    fa = torch.zeros_like(fa)
                else:
                    raise ValueError(
                        'The action of the follower should be either a numpy array or a torch tensor.'
                    )
        elif explorate_follower:
            for i in range(len(fa)):
                fa[i] = self._f_env_spec.action_space.sample()
        elif (isinstance(self._f_policy, CategoricalMLPPolicy) 
              or isinstance(self._f_policy, TabularCategoricalPolicy)):
            if deterministic_follower:
                fa = fa_infos['mode']
        elif isinstance(self._f_policy, StochasticPolicy):
            if deterministic_follower:
                fa = fa_infos['mean']
        elif isinstance(self._f_policy, Policy):  # Deterministic policy
            if not deterministic_follower:
                fa_infos['mean'] = np.copy(fa)
                for itr, _ in enumerate(fa):
                    fa[itr] = np.clip(
                        fa[itr] + np.random.normal(size=fa[itr].shape) * self.noise_sigma_f,
                        self._f_env_spec.action_space.low, self._f_env_spec.action_space.high)
        else:
            raise NotImplementedError
            
        infos = {}
        for k,v in la_infos.items():
            infos['leader_'+k] = v
        for k,v in fa_infos.items():
            infos[k] = v
        return [la, fa], infos
    
    def reset(self, do_resets=None):
        """Reset the policy."""
        self._l_policy.reset(do_resets)
        self._f_policy.reset(do_resets)

    def set_param_values(self, state_dict):
        self._l_policy.set_param_values(state_dict['leader'])
        self._f_policy.set_param_values(state_dict['follower'])

    def get_param_values(self):
        return {'leader': self._l_policy.get_param_values(), 
                'follower': self._f_policy.get_param_values()}
        
