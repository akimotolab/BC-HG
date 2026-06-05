from typing import Tuple, Optional, Callable
import jax
from flax import struct
from jax import numpy as jnp
from typing import NamedTuple

from src.environments.ConfigurableFourRooms import ConfigurableFourRooms, EnvState
from src.environments.utils import sample_array


def get_reward_matrix(
    env,
    env_params,
    external_reward: Optional[
        Callable[[EnvState, jnp.ndarray, NamedTuple], jnp.ndarray]
    ] = None,
) -> jnp.ndarray:
    """
    Get reward matrix
    :param env: Environment
    :param env_params: Environment parameters
    :param external_reward: External reward function r(state, action, env_params) -> reward
    :return: Reward matrix, Shape: (len(env.available_goals), n_states, n_actions, |params| if reward_grad)
    """

    def get_reward(goal, pos, action):
        """
        Return the reward for the given goal, pos, and action.
        Returned value can be either scalar or a vector value depending on the reward function
        """
        state = EnvState(pos=pos, goal=goal, time=0)
        if external_reward:
            return external_reward(state, action, env_params)
        else:
            return env.get_reward(state, action, env_params)

    get_reward_vmap = get_reward
    for i in range(3):
        in_axes = [None, None, None]
        in_axes[2 - i] = 0
        get_reward_vmap = jax.vmap(get_reward_vmap, in_axes=in_axes)
    reward_matrix = get_reward_vmap(
        env.available_goals, env.coords, jnp.arange(env.num_actions)
    )  # Shape: (len(env.available_goals), n_states, n_actions, |params| if reward_grad)
    return reward_matrix


def general_value_iteration(
    env: ConfigurableFourRooms,
    env_params: struct.dataclass,
    gamma: float,
    n_policy_iter: int,
    n_value_iter: int,
    policy: Optional[jnp.ndarray] = None,
    regularization: Optional[str] = None,
    reg_lambda: Optional[float] = jnp.nan,
    return_q_value: Optional[bool] = False,
    external_reward: Optional[
        Callable[[EnvState, jnp.ndarray, NamedTuple], jnp.ndarray]
    ] = None,
    arr_init: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    General function for both value iteration and value prediction
    - Regularization is only supported for Q-value iteration and for entropy regularization
    :param env: Environment
    :param env_params: Environment parameters
    :param gamma: Discount factor
    :param n_policy_iter: Number of policy iterations
    :param n_value_iter: Number of value iterations
    :param policy: Policy, Shape: (len(env.available_goals), n_states, n_actions)
        If None, value iteration is performed
        If given, value prediction is performed
    :param regularization: Name of the regularization
    :param reg_lambda: Regularization parameter
    :param return_q_value: If True, Q-value iteration is performed
    :param external_reward: External reward function r(state, action, env_params) -> reward
    :param arr_init: Initial value/Q array. If None, initialized as zeros.
        Shape: (len(env.available_goals), n_states) for value or
                (len(env.available_goals), n_states, n_actions, |params| if vectorized_reward) for Q-value
    :return:
        - Value function, Shape: (len(env.available_goals), n_states)
        - Errors, Shape: (len(env.available_goals), n_states)
    """
    if regularization:
        assert reg_lambda is not None, "Regularization parameter is not provided"
        assert return_q_value, "Regularization is only supported for Q-value iteration"
    if regularization == "KL_divergence":
        return_q_value = True
        policy = "softmax" if not isinstance(policy, jnp.ndarray) else policy
    if isinstance(policy, jnp.ndarray) and policy.ndim != 3:
        raise ValueError(
            "policy must be 3D with shape "
            "(len(env.available_goals), n_states, n_actions), "
            f"but got shape {policy.shape}"
        )
    # Setup variables
    reward_matrix = get_reward_matrix(
        env,
        env_params,
        external_reward=external_reward,
    )  # Shape: (len(env.available_goals), n_states, n_actions, |params| if vectorized_reward)
    vectorized_reward = len(reward_matrix.shape) == 4

    transition_probability_matrix = env.get_transition_probability_matrix(
        env_params
    )  # Shape: (len(env.available_goals), n_states, n_actions, n_states)
    is_terminal_state = jnp.expand_dims(
        env.terminal_states, -1
    )  # Shape: (len(env.available_goals), n_states, 1)
    if vectorized_reward:
        transition_probability_matrix = jnp.expand_dims(
            transition_probability_matrix, -1
        )  # Shape: (len(env.available_goals), n_states, n_actions, n_states, 1)
        is_terminal_state = jnp.expand_dims(
            is_terminal_state, -1
        )  # Shape: (len(env.available_goals), n_states, 1, 1)

    # Update Loop
    if arr_init is None:
        if not return_q_value:
            arr_init = jnp.zeros(
                (reward_matrix.shape[0], reward_matrix.shape[1])
            )  # Shape: (len(env.available_goals), n_states)
        else:
            arr_init = jnp.zeros_like(
                reward_matrix
            )  # Shape: (len(env.available_goals), n_states, n_actions, |params| if vectorized_reward)

    def policy_update(value_estimate_policy_carry, unused):
        if return_q_value:
            if policy == "softmax":
                iteration_policy = jax.nn.softmax(
                    value_estimate_policy_carry / reg_lambda, axis=-1
                )  # Shape: (len(env.available_goals), n_states, n_actions)
                # print("Using softmax policy update")
            elif isinstance(policy, jnp.ndarray):
                if vectorized_reward:
                    iteration_policy = jnp.expand_dims(
                        policy, -1
                    )  # Shape: (len(env.available_goals), n_states, n_actions, 1)
                else:
                    iteration_policy = policy
            elif policy is None:
                iteration_policy = None
            else:
                raise NotImplementedError
        else:
            iteration_policy = policy

        def value_update(value_estimate_carry, unused):
            # Update the state-value estimate
            if return_q_value:
                v = jnp.sum(
                    iteration_policy * value_estimate_carry, axis=2
                )  # Shape: (len(env.available_goals), n_states, |params| if return_grad)
            else:
                v = value_estimate_carry  # Shape: (len(env.available_goals), n_states, |params| if return_grad)
            discounted_next_value = jnp.sum(
                gamma * transition_probability_matrix * jnp.expand_dims(v, (1, 2)),
                axis=3,
            )  # Shape: (len(env.available_goals), n_states, n_actions, |params| if return_grad)
            discounted_next_value_non_terminal = jnp.where(
                is_terminal_state, 0, discounted_next_value
            )  # Shape: (len(env.available_goals), n_states, n_actions, |params| if return_grad)

            # Bellman update
            if policy is None and not return_q_value:  # Value Iteration
                value_estimate_carry = jnp.max(
                    reward_matrix + discounted_next_value_non_terminal, -1
                )
            elif (
                isinstance(policy, jnp.ndarray) and not return_q_value
            ):  # Policy Iteration
                value_estimate_carry = jnp.sum(
                    policy * (reward_matrix + discounted_next_value_non_terminal),
                    2,
                )  # Shape: (len(env.available_goals), n_states)
            elif (
                return_q_value and regularization == "KL_divergence"
            ):  # Q-value Iteration　 (or Policy Iteration for Q-value) with Regularization
                KL_divergence = jnp.expand_dims(
                    reg_lambda
                    * jnp.sum(
                        iteration_policy * jnp.log(iteration_policy + 1e-32),
                        axis=2,
                    ),
                    2,
                )  # Shape: (len(env.available_goals), n_states, 1, 1 if vectorized_reward)
                value_estimate_carry = (
                    reward_matrix - KL_divergence + discounted_next_value_non_terminal
                )
            elif isinstance(policy, jnp.ndarray) and return_q_value:  # Q Iteration
                value_estimate_carry = (
                    reward_matrix + discounted_next_value_non_terminal
                )
            else:
                raise NotImplementedError
            return value_estimate_carry, None

        new_value_estimate, _ = jax.lax.scan(
            value_update, value_estimate_policy_carry, None, n_value_iter
        )  # Value iteration
        return new_value_estimate, jnp.max(
            jnp.abs(value_estimate_policy_carry - new_value_estimate)
        )

    arr_final, errors = jax.lax.scan(policy_update, arr_init, None, n_policy_iter)
    return arr_final, errors


def general_value_iteration_return_intermediate(
    env: ConfigurableFourRooms,
    env_params: struct.dataclass,
    gamma: float,
    n_policy_iter: int,
    n_value_iter: int,
    policy: Optional[jnp.ndarray] = None,
    regularization: Optional[str] = None,
    reg_lambda: Optional[float] = jnp.nan,
    return_q_value: Optional[bool] = False,
    stop_policy_iter: Optional[int] = None,
    external_reward: Optional[
        Callable[[EnvState, jnp.ndarray, NamedTuple], jnp.ndarray]
    ] = None,
    arr_init: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    General function for both value iteration and value prediction
    - Regularization is only supported for Q-value iteration and for entropy regularization
    :param env: Environment
    :param env_params: Environment parameters
    :param gamma: Discount factor
    :param n_policy_iter: Number of policy iterations
    :param n_value_iter: Number of value iterations
    :param policy: Policy, Shape: (len(env.available_goals), n_states, n_actions)
        If None, value iteration is performed
        If given, value prediction is performed
    :param regularization: Name of the regularization
    :param reg_lambda: Regularization parameter
    :param return_q_value: If True, Q-value iteration is performed
    :param stop_policy_iter: Policy-iteration step to snapshot intermediate estimate
        (1-indexed, <= n_policy_iter). If None, uses n_policy_iter.
    :param external_reward: External reward function r(state, action, env_params) -> reward
    :param arr_init: Initial value/Q array. If None, initialized as zeros.
        Shape: (len(env.available_goals), n_states) for value or
                (len(env.available_goals), n_states, n_actions, |params| if vectorized_reward) for Q-value
    :return:
        - Value function at n_policy_iter, Shape: (len(env.available_goals), n_states)
        - Intermediate value/Q estimate at stop_policy_iter, Shape: (len(env.available_goals), n_states)
        - Errors, Shape: (len(env.available_goals), n_states)
    """
    if regularization:
        assert reg_lambda is not None, "Regularization parameter is not provided"
        assert return_q_value, "Regularization is only supported for Q-value iteration"
    if regularization == "KL_divergence":
        return_q_value = True
        policy = "softmax" if not isinstance(policy, jnp.ndarray) else policy
    if isinstance(policy, jnp.ndarray) and policy.ndim != 3:
        raise ValueError(
            "policy must be 3D with shape "
            "(len(env.available_goals), n_states, n_actions), "
            f"but got shape {policy.shape}"
        )
    if stop_policy_iter is None:
        stop_policy_iter = n_policy_iter
    if not (1 <= stop_policy_iter <= n_policy_iter):
        raise ValueError(
            f"stop_policy_iter must be in [1, {n_policy_iter}], but got {stop_policy_iter}"
        )
    # Setup variables
    reward_matrix = get_reward_matrix(
        env,
        env_params,
        external_reward=external_reward,
    )  # Shape: (len(env.available_goals), n_states, n_actions, |params| if vectorized_reward)
    vectorized_reward = len(reward_matrix.shape) == 4

    transition_probability_matrix = env.get_transition_probability_matrix(
        env_params
    )  # Shape: (len(env.available_goals), n_states, n_actions, n_states)
    is_terminal_state = jnp.expand_dims(
        env.terminal_states, -1
    )  # Shape: (len(env.available_goals), n_states, 1)
    if vectorized_reward:
        transition_probability_matrix = jnp.expand_dims(
            transition_probability_matrix, -1
        )  # Shape: (len(env.available_goals), n_states, n_actions, n_states, 1)
        is_terminal_state = jnp.expand_dims(
            is_terminal_state, -1
        )  # Shape: (len(env.available_goals), n_states, 1, 1)

    # Update Loop
    if arr_init is None:
        if not return_q_value:
            arr_init = jnp.zeros(
                (reward_matrix.shape[0], reward_matrix.shape[1])
            )  # Shape: (len(env.available_goals), n_states)
        else:
            arr_init = jnp.zeros_like(
                reward_matrix
            )  # Shape: (len(env.available_goals), n_states, n_actions, |params| if vectorized_reward)

    def policy_update(carry, unused):
        value_estimate_policy_carry, iter_idx, arr_intermediate = carry
        if return_q_value:
            if policy == "softmax":
                iteration_policy = jax.nn.softmax(
                    value_estimate_policy_carry / reg_lambda, axis=-1
                )  # Shape: (len(env.available_goals), n_states, n_actions)
                # print("Using softmax policy update")
            elif isinstance(policy, jnp.ndarray):
                if vectorized_reward:
                    iteration_policy = jnp.expand_dims(
                        policy, -1
                    )  # Shape: (len(env.available_goals), n_states, n_actions, 1)
                else:
                    iteration_policy = policy
            elif policy is None:
                iteration_policy = None
            else:
                raise NotImplementedError
        else:
            iteration_policy = policy

        def value_update(value_estimate_carry, unused):
            # Update the state-value estimate
            if return_q_value:
                v = jnp.sum(
                    iteration_policy * value_estimate_carry, axis=2
                )  # Shape: (len(env.available_goals), n_states, |params| if return_grad)
            else:
                v = value_estimate_carry  # Shape: (len(env.available_goals), n_states, |params| if return_grad)
            discounted_next_value = jnp.sum(
                gamma * transition_probability_matrix * jnp.expand_dims(v, (1, 2)),
                axis=3,
            )  # Shape: (len(env.available_goals), n_states, n_actions, |params| if return_grad)
            discounted_next_value_non_terminal = jnp.where(
                is_terminal_state, 0, discounted_next_value
            )  # Shape: (len(env.available_goals), n_states, n_actions, |params| if return_grad)

            # Bellman update
            if policy is None and not return_q_value:  # Value Iteration
                value_estimate_carry = jnp.max(
                    reward_matrix + discounted_next_value_non_terminal, -1
                )
            elif (
                isinstance(policy, jnp.ndarray) and not return_q_value
            ):  # Policy Iteration
                value_estimate_carry = jnp.sum(
                    policy * (reward_matrix + discounted_next_value_non_terminal),
                    2,
                )  # Shape: (len(env.available_goals), n_states)
            elif (
                return_q_value and regularization == "KL_divergence"
            ):  # Q-value Iteration　 (or Policy Iteration for Q-value) with Regularization
                KL_divergence = jnp.expand_dims(
                    reg_lambda
                    * jnp.sum(
                        iteration_policy * jnp.log(iteration_policy + 1e-32),
                        axis=2,
                    ),
                    2,
                )  # Shape: (len(env.available_goals), n_states, 1, 1 if vectorized_reward)
                value_estimate_carry = (
                    reward_matrix - KL_divergence + discounted_next_value_non_terminal
                )
            elif isinstance(policy, jnp.ndarray) and return_q_value:  # Q Iteration
                value_estimate_carry = (
                    reward_matrix + discounted_next_value_non_terminal
                )
            else:
                raise NotImplementedError
            return value_estimate_carry, None

        new_value_estimate, _ = jax.lax.scan(
            value_update, value_estimate_policy_carry, None, n_value_iter
        )  # Value iteration
        arr_intermediate = jax.lax.select(
            iter_idx == (stop_policy_iter - 1),
            new_value_estimate,
            arr_intermediate,
        )
        return (
            new_value_estimate,
            iter_idx + 1,
            arr_intermediate,
        ), jnp.max(jnp.abs(value_estimate_policy_carry - new_value_estimate))

    (arr_final, _, arr_intermediate), errors = jax.lax.scan(
        policy_update,
        (arr_init, jnp.asarray(0), arr_init),
        None,
        n_policy_iter,
    )
    return arr_final, arr_intermediate, errors


def value_prediction(
    env: ConfigurableFourRooms,
    env_params: struct.dataclass,
    gamma: float,
    n_policy_iter: int,
    n_value_iter: int,
    policy: jnp.ndarray,
    regularization: Optional[str] = None,
    reg_lambda: Optional[float] = jnp.nan,
    return_q_value: Optional[bool] = False,
    external_reward: Optional[
        Callable[[EnvState, jnp.ndarray, NamedTuple], jnp.ndarray]
    ] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Value prediction"""
    value, errors = general_value_iteration(
        env=env,
        env_params=env_params,
        gamma=gamma,
        n_policy_iter=n_policy_iter,
        n_value_iter=n_value_iter,
        policy=policy,
        regularization=regularization,
        reg_lambda=reg_lambda,
        return_q_value=return_q_value or regularization is not None,
        external_reward=external_reward,
    )  # Shape: (len(env.available_goals), n_states, Optional n_actions if return_q_value)
    if not return_q_value and regularization == "KL_divergence":
        policy = jax.nn.softmax(value / reg_lambda, axis=-1)
        value = jnp.sum(value * policy, axis=2)  # Shape: (len(env.available_goals), n_states)
    elif not return_q_value and regularization != "KL_divergence" and regularization is not None:
        raise NotImplementedError
    _, _, goal_probs = sample_array(
        jax.random.PRNGKey(0), env.available_goals, env_params.resample_goal_logits
    )
    return jnp.sum(value * jnp.expand_dims(goal_probs, (1, 2) if return_q_value else 1), axis=0), errors


def initial_value_prediction(
    env: ConfigurableFourRooms,
    env_params: struct.dataclass,
    gamma: float,
    n_policy_iter: int,
    n_value_iter: int,
    policy: jnp.ndarray,
    regularization: Optional[str] = None,
    reg_lambda: Optional[float] = jnp.nan,
    return_q_value: Optional[bool] = False,
    external_reward: Optional[
        Callable[[EnvState, jnp.ndarray, NamedTuple], jnp.ndarray]
    ] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    value_array, errors = value_prediction(
        env=env,
        env_params=env_params,
        gamma=gamma,
        n_policy_iter=n_policy_iter,
        n_value_iter=n_value_iter,
        policy=policy,
        regularization=regularization,
        reg_lambda=reg_lambda,
        return_q_value=return_q_value,
        external_reward=external_reward,
    )
    init_position_idx = jnp.all(
        env.coords[..., None] == env.available_init_pos.T[None, ...], axis=1
    )  # Shape: (n_states, n_init_pos)
    init_state_probs = env.state_initialization_distribution(
        env_params.state_initialization_params
    ).probs  # Shape: (n_states,)
    init_probs = jnp.sum(
        jnp.expand_dims(init_state_probs, 0) * init_position_idx, axis=1
    )  # Shape: (n_states,)
    return jnp.sum(value_array * init_probs, axis=0), errors

def value_iteration(
    env: ConfigurableFourRooms,
    env_params: struct.dataclass,
    gamma: float,
    n_policy_iter: int,
    n_value_iter: int,
    regularization: Optional[str] = None,
    reg_lambda: Optional[float] = jnp.nan,
    return_q_value: Optional[bool] = False,
    external_reward: Optional[
        Callable[[EnvState, jnp.ndarray, NamedTuple], jnp.ndarray]
    ] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Value iteration"""
    value, errors = general_value_iteration(
        env=env,
        env_params=env_params,
        gamma=gamma,
        n_policy_iter=n_policy_iter,
        n_value_iter=n_value_iter,
        policy=None,
        regularization=regularization,
        reg_lambda=reg_lambda,
        return_q_value=return_q_value,
        external_reward=external_reward,
    )
    _, _, goal_probs = sample_array(
        jax.random.PRNGKey(0), env.available_goals, env_params.resample_goal_logits
    )
    return jnp.sum(value * jnp.expand_dims(goal_probs, (1, 2) if return_q_value else 1), axis=0), errors
