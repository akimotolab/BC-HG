"""
Soft Q-Iteration with sub-optimal Q-table mixing.
"""
import torch
from dowel import tabular

from .soft_q_iteration import SoftQIteration


class SoftQIteration_Subopt(SoftQIteration):
    """
    Soft Q-Iteration that produces a sub-optimal policy by mixing
    pre-convergence and post-convergence Q tables, then deriving
    the policy from the mixed Q table.
    """
    name = 'SoftQIteration_Subopt'

    def __init__(
            self,
            *args,
            stop_q_iteration: int = None,
            reset_q: bool = False,
            **kwargs
            ):
        """
        Args:
            env: Environment with transition_fn and reward_fn.
            discount: Discount factor.
            temperature: Temperature for soft value computation.
            max_iterations: Maximum number of iterations.
            tol: Convergence tolerance.
            stop_q_iteration: Iteration at which to stop Q-iteration and use the current Q table.
                0 = fully pre-convergence (initial) Q table,
                max_iterations = fully converged (optimal) Q table.
            reset_q: Whether to reset the Q table at the start of each train() call.
        """
        super().__init__(*args, **kwargs)
        if stop_q_iteration is not None and not (0 <= stop_q_iteration <= self._max_iterations):
            raise ValueError("stop_q_iteration must be in [0, max_iterations]")
        self._stop_q_iteration = stop_q_iteration
        self._reset_q = reset_q

    def train(self, trainer):
        # Reset Q table if specified (e.g., for random initialization experiments)
        if self._reset_q:
            self._rand_init_q_table()

        # Save the pre-convergence Q table
        pre_convergence_Q = self.Q.clone()

        immediate_Q = None
        if self._stop_q_iteration == 0:
            immediate_Q = self.Q.clone()

        # Run train_once() until convergence
        for i in range(self._max_iterations):
            max_diff, info = self.train_once(trainer)
            # capture Q at the requested stop iteration (1-indexed semantics)
            if i == (self._stop_q_iteration - 1):
                immediate_Q = self.Q.clone()
            info['converged'] = False
            if max_diff < self._tol:
                info['converged'] = True
                break
        info['iteration'] = i + 1

        # Get the post-convergence (optimal) Q table
        post_convergence_Q = self.Q
        info['initial_Q'] = pre_convergence_Q
        info['final_Q'] = post_convergence_Q

        if immediate_Q is None:
            immediate_Q = post_convergence_Q.clone()

        # Update policy with immediate Q table
        self.Q = immediate_Q
        self.V = self.soft_value_operator(self.Q)
        self.update_policy()

        return max_diff, info
    
    def policy_matrix(self, Q_table=None):
        beta = self._temperature
        Q = Q_table if Q_table is not None else self.Q
        return torch.softmax(Q / beta, dim=-1)

    def optimality_gap(self, q, final_q, initial_q=None):
        """Compute the normalized optimality gap between the mixed policy table and the final policy table."""
        with torch.no_grad():
            policy = self.policy_matrix(q)  # shape: (n_states, n_leader_actions, n_follower_actions)
            final_policy = self.policy_matrix(final_q)  # shape: (n_states, n_leader_actions, n_follower_actions)
            initial_policy = self.policy_matrix(initial_q) if initial_q is not None else None
            # Compute the average KL divergence across all states and leader actions
            kl_divergence = torch.sum(
                policy * (torch.log(policy + 1e-10) - torch.log(final_policy + 1e-10)),
                dim=-1
            ).mean().item()
            kl_divergence_normalized = kl_divergence
            if initial_policy is not None:
                kl_initial = torch.sum(
                    initial_policy * (torch.log(initial_policy + 1e-10) - torch.log(final_policy + 1e-10)),
                    dim=-1
                ).mean().item()
                # Normalize by the KL divergence of the initial policy to the final policy
                kl_divergence_normalized = kl_divergence / (kl_initial + 1e-10)
            return kl_divergence_normalized, kl_divergence

    def _log_statistics(self, max_diff, info, prefix='Follower'):
        super()._log_statistics(max_diff, info, prefix)
        with torch.no_grad():
            with tabular.prefix(prefix + '/'):
                gap_normalized, gap_unnormalized = self.optimality_gap(self.Q, info['final_Q'], info['initial_Q'])
                tabular.record('OptimalityGap', gap_unnormalized)
                tabular.record('OptimalityGapNormalized', gap_normalized)

    def _rand_init_q_table(self):
        self.Q = torch.randn_like(self.Q)
        self.V = self.soft_value_operator(self.Q)
