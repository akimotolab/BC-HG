import numpy as np
import torch
from garage.torch.policies.stochastic_policy import StochasticPolicy

class TabularCategoricalPolicy(StochasticPolicy):
    def __init__(self, env_spec, policy_matrix, name='DiscreteQFSoftmaxPolicy'):
        """
        Args:
            policy_matrix (np.ndarray or torch.Tensor): shape = (num_states, num_leader_actions, num_actions)
                各状態・リーダー行動ごとのアクション確率分布
            name (str): ポリシー名
        """
        # StochasticPolicyの初期化（env_specはNoneでOK、必要なら拡張）
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
        ポリシーマトリックスを更新するメソッド
        Args:
            policy_matrix (np.ndarray or torch.Tensor): 新しいポリシーマトリックス
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
                [state_onehot_vec, leader_action_onehor_vec]のconcatenation
        Returns:
            dist (torch.distributions.Categorical): バッチ分のCategorical分布
            info (dict): {'mode': ..., 'probs': ...}
        """
        if observations[0].shape[0] == self.num_states + self.num_leader_actions:
            # 状態・リーダー行動を抽出
            state = observations[:, :self.num_states]
            state = torch.argmax(state, dim=-1).long()  # 状態のone-hotベクトルをインデックスに変換
            leader_action = observations[:, self.num_states:]
            leader_action = torch.argmax(leader_action, dim=-1).long()  # リーダー行動のone-hotベクトルをインデックスに変換
        elif observations[0].shape[0] == 2:
            state = observations[:, 0].long()  # 状態をインデックスとして取得
            leader_action = observations[:, 1].long()
        else:
            raise ValueError("Observations must be in the format of [state, leader_action] or one-hot vectors.")
        # (batch_size, num_actions) の確率テーブルを取得
        probs = self.policy_matrix[state, leader_action]  # (batch_size, num_actions)
        # 必要ならfloat型に変換
        probs = probs.float() if hasattr(probs, 'float') else torch.from_numpy(probs).float()
        dist = torch.distributions.Categorical(probs=probs)
        mode = torch.argmax(probs, dim=-1)
        return dist, {'mode': mode, 'probs': probs}

    @property
    def obs_dim(self):
        # テーブル型なので特に意味はないが、2（state, leader_action）とする
        return 2

    @property
    def action_dim(self):
        # policy_matrixの最後の次元
        return self.policy_matrix.shape[-1]