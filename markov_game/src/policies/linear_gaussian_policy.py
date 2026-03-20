import numpy as np
import torch
import torch.nn as nn
from garage.torch.policies.stochastic_policy import StochasticPolicy

class LinearGaussianPolicy(StochasticPolicy):
    def __init__(self, 
                 env_spec, 
                 init_mean_K=None,
                 init_mean_W=None,
                 init_std_K=None,
                 init_std_W=None,
                 init_K=None,
                 init_W=None,
                 fixed_K=None,
                 fixed_W=None,
                 name='LinearGaussianPolicy'):
        """
        action ~ N(K s, W * W^T), where s is the state
        Args:
            env_spec (EnvSpec): Environment specification.
            name (str): Policy name, also the name of the module.
        Returns: None
        """
        super().__init__(env_spec=env_spec, name=name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        if fixed_K is not None:
            # Fixed K (not learnable)
            self.K = torch.as_tensor(fixed_K, dtype=torch.float32)
            if not self.K.shape == (self._action_dim, self._obs_dim):
                raise ValueError(f"fixed_K should have shape ({self._action_dim}, {self._obs_dim})")
        else:
            if init_K is not None:
                # Predefined initial K
                self.K = nn.Parameter(torch.as_tensor(init_K, dtype=torch.float32))
                if not self.K.shape == (self._action_dim, self._obs_dim):
                    raise ValueError(f"init_K should have shape ({self._action_dim}, {self._obs_dim})")
            else:
                # Random initial K
                init_mean_K = (torch.zeros((self._action_dim, self._obs_dim)) 
                               if init_mean_K is None 
                               else torch.as_tensor(init_mean_K, dtype=torch.float32))
                init_std_K = (torch.ones((self._action_dim, self._obs_dim)) 
                              if init_std_K is None 
                              else torch.as_tensor(init_std_K, dtype=torch.float32))
                if not init_mean_K.shape == init_std_K.shape == (self._action_dim, self._obs_dim):
                    raise ValueError(f"init_mean_K and init_std_K should have shape ({self._action_dim}, {self._obs_dim})")
                self.K = nn.Parameter(torch.normal(mean=init_mean_K, std=init_std_K))
        if fixed_W is not None:
            # Fixed W (not learnable)
            self.W = torch.as_tensor(fixed_W, dtype=torch.float32)
            if not self.W.shape == (self._action_dim, self._action_dim):
                raise ValueError(f"fixed_W should have shape ({self._action_dim}, {self._action_dim})")
        else:
            if init_W is not None:
                # Predefined initial W
                self.W = nn.Parameter(torch.as_tensor(init_W, dtype=torch.float32))
                if not self.W.shape == (self._action_dim, self._action_dim):
                    raise ValueError(f"init_W should have shape ({self._action_dim}, {self._action_dim})")
            else:
                # Random initial W
                init_mean_W = (torch.zeros((self._action_dim, self._action_dim)) 
                               if init_mean_W is None 
                               else torch.as_tensor(init_mean_W, dtype=torch.float32))
                init_std_W = (torch.ones((self._action_dim, self._action_dim)) 
                              if init_std_W is None 
                              else torch.as_tensor(init_std_W, dtype=torch.float32))
                if not init_mean_W.shape == init_std_W.shape == (self._action_dim, self._action_dim):
                    raise ValueError(f"init_mean_W and init_std_W should have shape ({self._action_dim}, {self._action_dim})")
                self.W = nn.Parameter(torch.normal(mean=init_mean_W, std=init_std_W))

    def set_parameters(self, K, W):
        """
        Args:
            K (np.ndarray or torch.Tensor): shape = (action_dim, state_dim)
            W (np.ndarray or torch.Tensor): shape = (action_dim, action_dim)
        Returns: None
        """
        self.K = nn.Parameter(torch.as_tensor(K))
        self.W = nn.Parameter(torch.as_tensor(W))

    def forward(self, observations):
        """
        Args:
            observations (torch.Tensor): shape = (batch_size, flat_state_dim)
        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors
        """
        batch_mean = observations @ self.K.T  # shape = (batch_size, action_dim)
        Covariance_matrix = self.W @ self.W.T
        batch_Sigma = Covariance_matrix.expand(
            observations.shape[0], self._action_dim, self._action_dim
            )  # shape = (batch_size, action_dim, action_dim)
        dist = torch.distributions.MultivariateNormal(
            loc=batch_mean, covariance_matrix=batch_Sigma
            )
        ret_mean = dist.mean.cpu()
        return dist, {'mean': ret_mean}
    
    def sample_disturbance(self):
        W = self.W.detach().cpu().numpy()
        return W @ np.random.randn(W.shape[1])

    def to(self, device):
        if not isinstance(self.K, torch.nn.Parameter):
            self.K = self.K.to(device)
        if not isinstance(self.W, torch.nn.Parameter):
            self.W = self.W.to(device)
        return super().to(device)

    @property
    def sqrtSigma(self):
        return np.linalg.cholesky(self.Sigma.detach().cpu().numpy())
    
    @property
    def obs_dim(self):
        """Return the observation dimension."""
        return self._obs_dim

    @property
    def action_dim(self):
        """Return the action dimension."""
        return self._action_dim
    
