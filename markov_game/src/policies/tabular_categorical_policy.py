import numpy as np
import torch
from garage.torch.policies.stochastic_policy import StochasticPolicy

class TabularCategoricalPolicy(StochasticPolicy):
    def __init__(self, env_spec, policy_matrix, name='DiscreteQFSoftmaxPolicy'):
        """
        Args:
            policy_matrix (np.ndarray or torch.Tensor): shape = (num_states, num_leader_actions, num_actions)
                Action probability distribution for each state and leader action.
            name (str): Policy name.
        """
        # Initialize StochasticPolicy (env_spec can be None; extend if needed)
        super().__init__(env_spec=env_spec, name=name)
        self.num_states = env_spec.observation_space.n
        self.num_actions = env_spec.action_space.n
        self.num_leader_actions = env_spec.leader_action_space.n
        self.policy_matrix = torch.zeros(
            self.num_states, self.num_leader_actions, self.num_actions
        )

        self.set_policy_matrix(policy_matrix)

    def set_policy_matrix(self, policy_matrix):
        """
        Update the policy matrix.
        Args:
            policy_matrix (np.ndarray or torch.Tensor): New policy matrix.
        """
        self.policy_matrix = policy_matrix

    def get_entropy(self):
        """Calculate entropy of the policy for each state.
        
        Returns:
            torch.Tensor: Entropy values with shape (num_states, num_leader_actions)
        """
        policy_probs = self.policy_matrix
        return -torch.sum(policy_probs * torch.log(policy_probs + 1e-10), dim=-1)

    def forward(self, observations):
        """
        Args:
            observations (torch.Tensor): shape = (batch_size, obs_dim+leader_action_dim)
                Concatenation of [state_onehot_vec, leader_action_onehot_vec].
        Returns:
            dist (torch.distributions.Categorical): Batch categorical distribution.
            info (dict): {'mode': ..., 'probs': ...}
        """
        if observations[0].shape[0] == self.num_states + self.num_leader_actions:
            # Extract state and leader action
            state = observations[:, :self.num_states]
            state = torch.argmax(state, dim=-1).long()  # Convert one-hot state vector to index
            leader_action = observations[:, self.num_states:]
            leader_action = torch.argmax(leader_action, dim=-1).long()  # Convert one-hot leader action vector to index
        elif observations[0].shape[0] == 2:
            state = observations[:, 0].long()  # Get state as index
            leader_action = observations[:, 1].long()
        else:
            raise ValueError("Observations must be in the format of [state, leader_action] or one-hot vectors.")
        # Get the probability table of shape (batch_size, num_actions)
        probs = self.policy_matrix[state, leader_action]  # (batch_size, num_actions)
        # Convert to float if needed
        probs = probs.float() if hasattr(probs, 'float') else torch.from_numpy(probs).float()
        dist = torch.distributions.Categorical(probs=probs)
        mode = torch.argmax(probs, dim=-1)
        return dist, {'mode': mode, 'probs': probs}

    @property
    def obs_dim(self):
        # Not particularly meaningful for tabular form, but set to 2 (state, leader_action)
        return 2

    @property
    def action_dim(self):
        # Last dimension of policy_matrix
        return self.policy_matrix.shape[-1]