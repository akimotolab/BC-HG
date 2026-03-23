"""
Soft Q-Iteration (entropy-regularized value iteration) for discrete MDPs.
Reference: "Reinforcement Learning with Deep Energy-Based Policies" (Haarnoja et al., 2017)
"""
import numpy as np
import torch
from torch import inverse, slogdet, cholesky
from akro import Discrete
from collections import deque
from garage.np.algos import RLAlgorithm
from garage.torch import global_device, as_torch
from garage.torch.policies.stochastic_policy import StochasticPolicy

from dowel import tabular

# For type hints (not necessary)
from ..experiment import Trainer

class MaxEntLQR(RLAlgorithm):
    name = "MaxEntLQR"
    
    def __init__(
            self,
            env,
            *,
            discount=0.95,
            beta=1.0,
            max_iterations=5000,
            tol=1e-10):
        
        self.env_spec = env.spec  # must be GlobalEnvSpec
        self._discount = discount
        self._beta = beta
        self._max_iterations = max_iterations
        self._tol = tol
        self._obs_dim = self.env_spec.observation_space.flat_dim
        self._action_dim = self.env_spec.action_space.flat_dim
        self._leader_action_dim = self.env_spec.leader_action_space.flat_dim
        
        if all([hasattr(env, Mat) for Mat in ['A', 'B', 'C', 'Q', 'R']]):
            self.A = torch.from_numpy(env.A).float()  # transition matrix
            self.B = torch.from_numpy(env.B).float()  # transition matrix
            self.C = torch.from_numpy(env.C).float()  # leader influence matrix
            self.Q = torch.from_numpy(env.Q).float()  # state cost matrix
            self.R = torch.from_numpy(env.R).float()  # action cost matrix
        else:
            raise NotImplementedError("env must have A, B, C, Q, R matrices.")
        
        self.policy = MaxEntLQRPolicy(
            env_spec=self.env_spec,
            Kx=torch.zeros((self._action_dim, self._obs_dim), dtype=torch.float32),
            Ka=torch.zeros((self._action_dim, self._obs_dim), dtype=torch.float32),
            C=self.C,
            K=torch.zeros((self._leader_action_dim, self._obs_dim), dtype=torch.float32),
            sqrtSigma=torch.eye(self._action_dim, dtype=torch.float32),
        )
        self.episode_rewards = deque(maxlen=30)  # retains episodes sampled in the leader's iteration

    def train(self, trainer: Trainer):
        """
        Solve KL-regularized LQR with observable additive noise:
            x_{t+1} = A x_t + B u_t + a_t,   a_t ~ N(0,W * W^T), observed before u_t.
        Value: V(x,a) = [x; a]^T P [x; a] + v, with P partitioned.
        Returns P_blocks, v, Kx, Ka, Sigma.
        """
        with torch.no_grad():
            """
            To compatible with the formulation of state transitions,
                x_{t+1} = A x_t + B u_t + C (K x_t + a_t)
                        = (A + C K) x_t + B u_t + C a_t,
            where K x_t + a_t is the leader's action, with a_t ~ N(0, W * W^T),
            substitute:
                A <- A + C K
                W <- C W (C a_t ~ N(0, (C W)(C W)^T))
            """
            K = trainer.leader.K.data.clone().detach()  # (leader_action_dim, state_dim)
            W = trainer.leader.W.data.clone().detach()  # (leader_action_dim, leader_action_dim)
            A = self.A + self.C @ K
            W = self.C @ W
            B, Q, R = self.B, self.Q, self.R
            gamma, beta = self._discount, self._beta
            n = A.shape[0]
            m = B.shape[1]

            # initialize
            P_xx = Q.clone().detach()
            v = 0.0
            info = {}
            for i in range(self._max_iterations):
                S = R + gamma * B.T @ P_xx @ B  # m x m, positive definite
                S_inv = inverse(S)

                # Riccati-like update for P_xx
                P_xx_new = Q + gamma * A.T @ P_xx @ A \
                        - (gamma**2) * A.T @ P_xx @ B @ S_inv @ B.T @ P_xx @ A

                # compute other blocks from P_xx_new (we use updated P_xx for consistency)
                P_xx_tmp = P_xx_new
                S_tmp = R + gamma * B.T @ P_xx_tmp @ B
                S_tmp_inv = inverse(S_tmp)

                # P_xa = gamma * A.T @ P_xx_tmp - (gamma**2) * A.T @ P_xx_tmp @ B @ S_tmp_inv @ B.T @ P_xx_tmp
                P_aa = gamma * P_xx_tmp - (gamma**2) * P_xx_tmp @ B @ S_tmp_inv @ B.T @ P_xx_tmp

                # constant term update
                # v_new: (1 - gamma) v = -beta m/2 ln(pi beta) + beta/2 ln det S + gamma tr(P_aa W)
                sign, logdetS = slogdet(S_tmp)  # stable log-det
                if sign <= 0:
                    raise RuntimeError("S_tmp not positive definite in iteration.")
                v_new = (
                    - beta * (m / 2.0) * torch.log(torch.tensor(np.pi * beta, dtype=P_aa.dtype, device=P_aa.device))
                    + 0.5 * beta * logdetS
                    + gamma * torch.trace(P_aa @ W @ W.T)
                ) / (1.0 - gamma)

                # check convergence (on P_xx and v)
                Riccati_error = torch.norm(P_xx_new - P_xx, p='fro')
                value_error = torch.abs(v_new - v)
                if Riccati_error < self._tol and value_error < self._tol:
                    info['converged'] = True
                    P_xx = P_xx_new
                    v = v_new
                    break

                P_xx = P_xx_new
                v = v_new
            else:
                info['converged'] = False
            info['iteration'] = i + 1

            # final S and inverses
            S = R + gamma * B.T @ P_xx @ B
            S_inv = inverse(S)

            # compute final block matrices
            P_xa = gamma * A.T @ P_xx - (gamma**2) * A.T @ P_xx @ B @ S_inv @ B.T @ P_xx
            P_aa = gamma * P_xx - (gamma**2) * P_xx @ B @ S_inv @ B.T @ P_xx

            # policy gains and covariance
            Kx = gamma * S_inv @ B.T @ P_xx @ A
            Ka = gamma * S_inv @ B.T @ P_xx
            Sigma = (beta / 2.0) * S_inv

            # package P as block matrix for user
            P = torch.cat([
                torch.cat([P_xx, P_xa], dim=1),
                torch.cat([P_xa.T, P_aa], dim=1)
            ], dim=0)
        
            # All the attributes below are tensors on the global device
            self.A_CK = A  # (state_dim, state_dim)
            self.W = W     # (state_dim, leader_action_dim)
            self.P = P
            self.P_xx = P_xx  # (state_dim, state_dim)
            self.P_xa = P_xa  # (state_dim, state_dim)
            self.P_aa = P_aa  # (state_dim, state_dim)
            self.v = v  # scalar
            self.K = K  # (leader_action_dim, state_dim)
            self.Kx = Kx  # (action_dim, state_dim) 
            self.Ka = Ka  # (action_dim, state_dim)
            self.Sigma = Sigma  # (action_dim, action_dim)
            self.sqrtSigma = cholesky(Sigma)
            self.policy.update_policy(
                Kx=self.Kx,
                Ka=self.Ka,
                C=self.C,
                K=self.K,
                sqrtSigma=self.sqrtSigma,
            )
            
        return Riccati_error, value_error, info

    def _cost(self, x: torch.Tensor, u: torch.Tensor):
        return x.T @ self.Q @ x + u.T @ self.R @ u

    def _next_state(self, x: torch.Tensor, a: torch.Tensor, u: torch.Tensor):
        # a is the leader's action noise (a ~ N(0, W * W^T))
        return self.A_CK @ x + self.B @ u + self.C @ a

    def state_value(self, obs: torch.Tensor):  # Note: NEGATIVE Value
        x = obs[:self._obs_dim]   # state
        a = obs[self._obs_dim:]   # leader's action
        a_noise = a - self.K @ x  # leader's action noise (a ~ N(0, W * W^T))
        Ca = self.C @ a_noise     # leader's action noise (a ~ N(0, CW * (CW)^T))
        return x.T @ self.P_xx @ x + 2 * x.T @ self.P_xa @ Ca + Ca.T @ self.P_aa @ Ca + self.v
    
    def state_value_batch(self, obs: torch.Tensor):  # Note: NEGATIVE Value
        """
        Args:
            obs (torch.Tensor): shape = (batch_size, state_dim + leader_action_dim)
        Returns:
            torch.Tensor: shape = (batch_size,) - state values for each observation in the batch
        """
        x = obs[:, :self._obs_dim]   # (batch_size, state_dim)
        a = obs[:, self._obs_dim:]   # (batch_size, leader_action_dim)
        a_noise = a - x @ self.K.T   # (batch_size, leader_action_dim)
        Ca = a_noise @ self.C.T      # (batch_size, state_dim)
        # Batch quadratic form computation
        term1 = torch.sum(x * (x @ self.P_xx), dim=1)  # x.T @ P_xx @ x for each batch
        term2 = 2 * torch.sum(x * (Ca @ self.P_xa.T), dim=1)  # 2 * x.T @ P_xa @ Ca for each batch  
        term3 = torch.sum(Ca * (Ca @ self.P_aa), dim=1)  # Ca.T @ P_aa @ Ca for each batch
        
        return (term1 + term2 + term3 + self.v).unsqueeze(-1)  # (batch_size, 1)

    def action_value(self, obs: torch.Tensor, u: torch.Tensor):  # Note: NEGATIVE Q-value
        x = obs[:self._obs_dim]   # state
        a = obs[self._obs_dim:]   # leader's action
        a_noise = a - self.K @ x  # leader's action noise (a ~ N(0, W * W^T))
        Ca = self.C @ a_noise     # leader's action noise (a ~ N(0, CW * (CW)^T))
        xx = self._next_state(x, Ca, u)
        next_expected_value = xx.T @ self.P_xx @ xx + (self.P_aa * (self.W @ self.W.T)).sum() + self.v
        return self._cost(x, u) + self.discount * next_expected_value

    def action_value_batch(self, obs: torch.Tensor, u: torch.Tensor):  # Note: NEGATIVE Q-value
        """
        Args:
            obs (torch.Tensor): shape = (batch_size, state_dim + leader_action_dim)
            u (torch.Tensor): shape = (batch_size, action_dim)
        Returns:
            torch.Tensor: shape = (batch_size,) - action values for each observation-action pair in the batch
        """
        x = obs[:, :self._obs_dim]   # (batch_size, state_dim)
        a = obs[:, self._obs_dim:]   # (batch_size, leader_action_dim)
        a_noise = a - x @ self.K.T   # (batch_size, leader_action_dim)
        Ca = a_noise @ self.C.T      # (batch_size, state_dim)
        
        # Next state computation for batch
        xx = x @ self.A_CK.T + u @ self.B.T + Ca @ self.C.T  # (batch_size, state_dim)
        # Next expected value for batch
        next_expected_value = (
            torch.sum(xx * (xx @ self.P_xx), dim=1) +  # xx.T @ P_xx @ xx for each batch
            torch.trace(self.P_aa @ (self.W @ self.W.T)) +  # constant term (same for all batches)
            self.v  # scalar constant
        )
        # Cost computation for batch
        state_cost = torch.sum(x * (x @ self.Q), dim=1)  # x.T @ Q @ x for each batch
        action_cost = torch.sum(u * (u @ self.R), dim=1)  # u.T @ R @ u for each batch
        current_cost = state_cost + action_cost
        
        return (current_cost + self._discount * next_expected_value).unsqueeze(-1)  # (batch_size, 1)
    
    def policy_entropy(self):
        m = self.env_spec.action_space.flat_dim
        sign, logdetSigma = np.linalg.slogdet(self.Sigma.cpu().numpy())
        if sign <= 0:
            raise RuntimeError("Sigma not positive definite.")
        return 0.5 * (m * (1.0 + np.log(2.0 * np.pi)) + logdetSigma)

    def _log_statistics(self,Riccati_error, value_error, info, prefix='Follower'):
        with torch.no_grad():
            with tabular.prefix(prefix + '/'):
                tabular.record('RiccatiError', Riccati_error)
                tabular.record('ValueError', value_error)
                tabular.record('Iterations', info['iteration'])
                tabular.record('Converged', info['converged'])
                tabular.record('Policy/Entropy', self.policy_entropy())
                tabular.record('TrainAverageReturn',
                               np.mean(self.episode_rewards))

    def to(self, device=None):
        if device is None:
            device = global_device()
        self.A = self.A.to(device)
        self.B = self.B.to(device)
        self.C = self.C.to(device)
        self.Q = self.Q.to(device)
        self.R = self.R.to(device)
        self.policy.to(device)


class MaxEntLQRPolicy(StochasticPolicy):
    def __init__(self, env_spec, Kx, Ka, C, K, sqrtSigma, name='MaxEntLQRPolicy'):
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
        self._leader_action_dim = env_spec.leader_action_space.flat_dim
        # must be tensor:
        self.Kx = Kx  # (action_dim, obs_dim)
        self.Ka = Ka  # (action_dim, obs_dim)
        self.C = C    # (obs_dim, leader_action_dim)
        self.K = K    # (leader_action_dim, obs_dim)
        self.sqrtSigma = sqrtSigma

    def update_policy(self, Kx, Ka, C, K, sqrtSigma):
        self.Kx = Kx
        self.Ka = Ka
        self.C = C
        self.K = K
        self.sqrtSigma = sqrtSigma

    def forward(self, observations):
        """
        Args:
            observations (torch.Tensor): shape = (batch_size, flat_state_dim+flat_leader_action_dim)
        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors
        """
        observations = observations.float()
        obs = observations[:, :self._obs_dim]  # (batch_size, state_dim)
        leader_action = observations[:, self._obs_dim:]  # leader's action, (batch_size, leader_action_dim)
        a = leader_action - obs @ self.K.T  # leader's action noise, (batch_size, leader_action_dim)
        batch_mean = (- self.Kx @ obs.T - self.Ka @ self.C @ a.T).T  # (batch_size, action_dim)
        Covariance_matrix = self.sqrtSigma @ self.sqrtSigma.T
        batch_Sigma = Covariance_matrix.expand(
            observations.shape[0], self._action_dim, self._action_dim
            )  # shape = (batch_size, action_dim, action_dim)
        dist = torch.distributions.MultivariateNormal(
            loc=batch_mean, covariance_matrix=batch_Sigma
            )
        ret_mean = dist.mean.cpu()
        return dist, {'mean': ret_mean}
    
    def sample_disturbance(self):
        W = self.sqrtSigma.detach().cpu().numpy()
        return W @ np.random.randn(W.shape[1])
    
    @property
    def obs_dim(self):
        """Return the observation dimension."""
        return self._obs_dim

    @property
    def action_dim(self):
        """Return the action dimension."""
        return self._action_dim
    
    def to(self, device=None):
        if device is None:
            device = global_device()
        self.Kx = self.Kx.to(device)
        self.Ka = self.Ka.to(device)
        self.C = self.C.to(device)
        if isinstance(self.K, torch.nn.Parameter):
            K = self.K.data.clone().detach()
            delattr(self, 'K')
            self.K = K
        self.K = self.K.to(device)
        self.sqrtSigma = self.sqrtSigma.to(device)