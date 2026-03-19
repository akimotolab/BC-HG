import jax
import jax.numpy as jnp
import yaml
import argparse
import os
import pickle
from typing import Iterable, Tuple, Any, Dict, Callable, Optional
import orbax
from flax.training import orbax_utils
import time
import distrax
from flax.training.train_state import TrainState
from copy import deepcopy
import flax
from scipy.special import comb

from src.environments.BuildingThermalControl import (
    EnvParams,
    EnvState,
    BuildingThermalControl,
    setup_environment
)
from src.algorithms.regularized_lqr import (
    create_regularized_lqr, 
    Transition,
    RegularizedLQROutputs,
    RegularizedLQRPolicy,
    create_state_value_fn,
    create_q_value_fn,
    update_dictionary,
)
from src.models.StaticModel import create_state_model as create_static_train_state
from src.models.StaticModel import restore_state_model as restore_static_train_state
from src.train.utils import update_nested_pytree, remove_non_list_entries
from src.models.ValueNetwork import mse
from src.models.ValueNetwork import create_train_state as create_train_state_value_model
from src.models.ValueNetwork import restore_train_state as restore_train_state_value_model


def update_transition_params(
    params: EnvParams,
    train_state_dict: Dict[str, TrainState],
) -> EnvParams:
    """
    Update the transition parameters with the current values from the training state
    """
    return params.replace(
        transition_params=params.transition_params.replace(
            insulation_level=train_state_dict["insulation_level"].apply_fn(
                train_state_dict["insulation_level"].params
            ),
            airflow_adjustment=train_state_dict["airflow_adjustment"].apply_fn(
                train_state_dict["airflow_adjustment"].params
            ),
        ),
    )


def clip_grads_per_sample_norm(grads: Any, max_norm: Optional[float]):
    """
    Clip gradients by L2 norm for each individual sample.
    For arrays with shape (n_steps, num_envs, params_dim), 
    clip each (params_dim,) vector independently.
    
    Args:
        grads: pytree of arrays with shape (n_steps, num_envs, params_dim)
        max_norm: maximum L2 norm for each individual gradient vector
    Returns:
        clipped_grads: pytree with same structure, clipped per sample
        max_norm_per_leaf: dict of maximum norms found in each leaf
    """
    if max_norm is None or max_norm <= 0:
        return grads, {}

    def clip_leaf(leaf):
        if leaf is None:
            return leaf, 0.0
        # Calculate L2 norm for each (params_dim,) vector
        # leaf shape: (n_steps, num_envs, params_dim)
        norms = jnp.linalg.norm(leaf, axis=-1, keepdims=True)  # (n_steps, num_envs, 1)
        # Scale factor for each vector
        eps = 1e-12
        scale = jnp.minimum(1.0, max_norm / (norms + eps))  # (n_steps, num_envs, 1)
        # Apply clipping
        clipped = leaf * scale  # (n_steps, num_envs, params_dim)
        # Return max norm found in this leaf
        max_norm_found = jnp.max(norms)
        return clipped, max_norm_found

    # Apply to all leaves
    clipped_grads = jax.tree_map(
        lambda x: clip_leaf(x)[0], 
        grads, 
        is_leaf=lambda x: x is None or isinstance(x, jnp.ndarray)
    )
    
    max_norms = jax.tree_map(
        lambda x: clip_leaf(x)[1], 
        grads, 
        is_leaf=lambda x: x is None or isinstance(x, jnp.ndarray)
    )
    
    return clipped_grads, max_norms


def create_trajectory_batch_sample(
    config_create: Dict,
    env: BuildingThermalControl,
    for_eval: bool,
) -> Callable[[jax.random.PRNGKey, TrainState, EnvParams, Optional[float]], Transition]:
    """
    Create the trajectory batch sampling function
    :param config_create:
    :param env:
    :param env_params:
    :return:
    """
    vmap_reset = lambda n_envs: lambda rng, params: jax.vmap(
        env.reset, in_axes=(0, None)
    )(jax.random.split(rng, n_envs), params)
    vmap_step = lambda n_envs: lambda rng, env_state, action, params: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, params)

    def get_trajectory_batch(
        key: jax.random.PRNGKey,
        lqr_output: RegularizedLQROutputs,
        env_params_sampling: EnvParams,
    ):
        policy = RegularizedLQRPolicy(
            Ks=lqr_output.Ks,
            Sigma=lqr_output.Sigma,
            sqrtSigma=lqr_output.sqrtSigma
        )

        def rollout_step(carry, unused):
            rng_carry, env_state_carry, last_obs = carry
            rng_carry, rng_a1, rng_s = jax.random.split(rng_carry, 3)            
            
            # Get the action
            rng_keys = jax.random.split(rng_a1, config_create["num_envs"])
            action = policy.get_actions(last_obs, rng_keys)
            
            # # Ensure action has the right shape for vmap
            # if action.ndim == 0:
            #     action = jnp.expand_dims(action, axis=0)
            # elif action.ndim == 1 and action.shape[0] != config_create["num_envs"]:
            #     action = jnp.atleast_1d(action)
            #     if action.shape[0] == 1 and config_create["num_envs"] > 1:
            #         action = jnp.repeat(action, config_create["num_envs"], axis=0)

            obs, env_state_carry_new, reward, done, info = vmap_step(
                config_create["num_envs"]
            )(rng_s, env_state_carry, action, env_params_sampling)
            transition = Transition(
                obs=last_obs,
                action=action,
                reward=reward,
                done=done,
                state=env_state_carry,
            )

            carry = (rng_carry, env_state_carry_new, obs)
            return carry, transition

        key_init, key_rollout = jax.random.split(key, 2)
        init_obs, init_env_state = vmap_reset(config_create["num_envs"])(
            key_init, env_params_sampling
        )
        steps = (config_create["num_estimation_steps"] if not for_eval
                 else config_create["num_eval_steps"])
        _, traj_batch = jax.lax.scan(
            rollout_step,
            (key_rollout, init_env_state, init_obs),
            None,
            steps // config_create["num_envs"],
        )
        return traj_batch

    return get_trajectory_batch


def calculate_discounted_rewards(
    reward_function_params,
    reward_function: Callable,
    traj_batch: Transition,
    discount_factor: float,
    initial_value: Any,
) -> Tuple[jax.Array, jax.Array]:
    """
    Calculate the discounted rewards for a trajectory batch
    :param reward_function_params:
    :param reward_function:
    :param traj_batch:
    :param discount_factor:
    :param initial_value: Initial value for the discounted rewards, matching the shape of the output of reward_function
    :return: Discounted rewards, matching the shape of the initial_value
        The returned array has shape (n_steps, num_envs, pyTree structure of initial_value)
    """

    def _get_discounted_reward(
        rolling_discounted_rewards: jax.Array,
        scan_input,
    ) -> Tuple[jax.Array, jax.Array]:
        transition = scan_input
        # vmap over num_envs dimension
        reward = jax.vmap(
            reward_function,
            in_axes=(0, 0, None),
        )(
            transition.state, transition.action, reward_function_params
        )  # Shape: (num_envs, Optional[params_dim] )
        done = transition.done.astype(jnp.float32)  # Shape: (num_envs,)
        rolling_discounted_rewards = jax.tree_map(
            lambda x, y: (
                x
                + discount_factor
                * (1 - (done if len(x.shape) == 1 else jnp.atleast_2d(done).T))
                * y
            ),
            reward,
            rolling_discounted_rewards,
        )  # Calculate discounted reward, Shape: (num_envs, Optional[params_dim])
        return rolling_discounted_rewards, (reward, rolling_discounted_rewards)
        
    _, (rewards, discounted_rewards) = jax.lax.scan(
        _get_discounted_reward,
        initial_value,
        traj_batch,
        reverse=True,
    )  # Shape: (n_steps, num_envs, Optional[params_dim])
    return rewards, discounted_rewards


def upper_level_reward_gradient(
    env: BuildingThermalControl,
    env_params: EnvParams,
    traj_batch: Transition,
    upper_level_train_states: Dict[str, TrainState],
) -> Tuple[
    jax.Array,
    Tuple[Dict[str, Dict[str, jax.Array]], Dict[str, Dict[str, jax.Array]]],
]:
    """
    Estimate the gradient of the upper-level reward (i.e., upper-level immediate reward)
    :return: Tuple with the gradient dictionaries for the two parameters
    """

    def upper_level_reward(
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
        IL_params: jax.Array,  # insulation level parameters
        AA_params: jax.Array,  # airflow adjustment parameters
    ) -> jax.Array:
        """Auxiliary function to calculate the upper-level reward as a function of the tax parameters"""
        insulation_level = upper_level_train_states["insulation_level"].apply_fn(IL_params)
        airflow_adjustment = upper_level_train_states["airflow_adjustment"].apply_fn(AA_params)
        params_env_tmp = params.replace(
            transition_params=params.transition_params.replace(
                insulation_level=insulation_level,
                airflow_adjustment=airflow_adjustment,
            ),
        )
        return env.upper_level_reward(
            state, action, params_env_tmp
        )[0]

    upper_level_reward_grad = jax.grad(upper_level_reward, argnums=[3, 4])
    upper_level_reward_grad_vmap = jax.vmap(
        jax.vmap(upper_level_reward_grad, in_axes=(0, 0, None, None, None)),
        in_axes=(0, 0, None, None, None),
    )
    return upper_level_reward_grad_vmap(
        traj_batch.state,  # Shape: (n_steps, num_envs, state_dim)
        traj_batch.action, # Shape: (n_steps, num_envs, action_dim)
        env_params,
        upper_level_train_states["insulation_level"].params,
        upper_level_train_states["airflow_adjustment"].params,
    )  # Shape: ((n_steps, num_envs, dim_IL_param), (n_steps, num_envs, dim_AA_param))


def estimate_value_function(
    X: jax.Array,
    X_next: jax.Array,
    rewards: jax.Array,
    value_function_estimator: TrainState,
    num_steps: int,
    discount_factor: float,
    l2_reg: float,
):
    """
    Estimate the value function from the trajectory batch
    :param traj_batch:
    :param rewards: array of shape: (n_steps, num_envs, Optional[params_dim])
    :param value_function_estimator: TrainState of the value function estimator
    :param num_steps: Number of training steps
    :param discount_factor: Discount factor
    :param l2_reg: L2 regularization parameter for the MSE
    :return:
    """
    X = X.reshape(
        X.shape[0] * X.shape[1], *X.shape[2:]
    )  # Shape: (n_steps*num_envs, Optional[params_dim])
    X_next = X_next.reshape(
        X_next.shape[0] * X_next.shape[1], *X_next.shape[2:]
    )  # Shape: (n_steps*num_envs, Optional[params_dim])
    if rewards.ndim == 2:
        rewards_reshaped = jnp.expand_dims(rewards, -1)
    else:
        rewards_reshaped = rewards
    rewards_reshaped = rewards_reshaped.reshape(
        rewards_reshaped.shape[0] * rewards_reshaped.shape[1],
        *rewards_reshaped.shape[2:],
    )  # Shape: (n_steps*num_envs, Optional[params_dim])
    rewards_max = jnp.max(
        jnp.abs(jnp.where(jnp.isnan(rewards_reshaped), 0.0, rewards_reshaped)), 
        axis=0, keepdims=True
    )  # Shape: (Optional[params_dim],)
    rewards_reshaped = rewards_reshaped / rewards_max  # Normalize the rewards

    # Fitting
    mse_grad_fn = jax.value_and_grad(mse)

    def value_network_update(train_state_carry, unused):
        v_next = train_state_carry.apply_fn(
            train_state_carry.params, X_next
        )  # Shape: (n_steps*num_envs, Optional[params_dim])
        target = rewards_reshaped + discount_factor * jax.lax.stop_gradient(v_next)
        loss, grads = mse_grad_fn(
            train_state_carry.params, train_state_carry, X, target, l2_reg
        )
        train_state_carry = train_state_carry.apply_gradients(grads=grads)
        return train_state_carry, loss

    value_model_fitted, losses = jax.lax.scan(
        value_network_update,
        value_function_estimator,
        None,
        length=num_steps,
    )

    # Return estimate values for the trajectory batch
    value_estimate = value_model_fitted.apply_fn(
        value_model_fitted.params, X
    )  # Shape: (n_steps*num_envs, Optional[params_dim])
    value_estimate = (rewards_max * value_estimate).reshape(*rewards.shape)
    return value_model_fitted, value_estimate, losses


def calculate_transition_logprob_gradient(
    env: BuildingThermalControl,
    env_params: EnvParams,
    traj_batch: Transition,
    upper_level_train_states: Dict[str, TrainState],
    grad_clip: float,
):
    """
    Calculate the gradient of the transition dynamics log probability
    Assumes truncated normal distribution
    """

    def transition(
        state: EnvState,
        action: jax.Array,
        params_env: EnvParams,
        params_IL: jax.Array,  # insulation level parameters
        params_AA: jax.Array,  # airflow adjustment parameters
    ):
        """Auxiliary function to calculate the transition as a function of the tax parameters"""
        insulation_level = upper_level_train_states["insulation_level"].apply_fn(params_IL)
        airflow_adjustment = upper_level_train_states["airflow_adjustment"].apply_fn(params_AA)
        params_env_tmp = params_env.replace(
            transition_params=params_env.transition_params.replace(
                insulation_level=insulation_level,
                airflow_adjustment=airflow_adjustment,
            )
        )
        return env.transition(state, action, params_env_tmp.transition_params).temperature_deviations

    def transition_logprob(
        state: EnvState,
        action: jax.Array,
        new_state: EnvState,
        params_env: EnvParams,
        params_IL: jax.Array,  # insulation level parameters
        params_AA: jax.Array,  # airflow adjustment parameters
    ):
        """Auxiliary function to calculate the transition log probability as a function of the tax parameters"""
        mean = transition(state, action, params_env, params_IL, params_AA)
        std = params_env.transition_params.transition_std  # scalar: Covariance is diagonal with same std for all variables
        lower_bound, upper_bound = params_env.transition_params.temperature_range
        logprobs = jax.scipy.stats.truncnorm.logpdf(
            new_state.temperature_deviations,  # shape: (dim_obs,)
            a=lower_bound,
            b=upper_bound,
            loc=mean,
            scale=std,
        )
        return jnp.sum(logprobs)  # Because of independence between dimensions

    transition_logprob_grad_f = jax.grad(transition_logprob, argnums=[4, 5])
    # Forward-shift the trajectory batch, add NaNs where the time is 0 (i.e. no previous state)
    traj_batch_back_shift = jax.tree_map(
        lambda x: jnp.where(
            jnp.expand_dims(traj_batch.state.time == 0, -1)
            if len(x.shape) > 2
            else traj_batch.state.time == 0,
            jnp.nan,
            jnp.roll(x, shift=1, axis=0),
        ),
        traj_batch,
    )

    grads = jax.vmap(
        jax.vmap(
            lambda s, a, s_next, p: jax.tree_map(
                lambda x: jnp.nan_to_num(x, nan=0.0),
                transition_logprob_grad_f(
                    s,
                    a, 
                    s_next,
                    p,
                    upper_level_train_states["insulation_level"].params,
                    upper_level_train_states["airflow_adjustment"].params,
                ),
            ),
            in_axes=(0, 0, 0, None),
        ),
        in_axes=(0, 0, 0, None),
    )(
        traj_batch_back_shift.state,
        traj_batch_back_shift.action,
        traj_batch.state,
        env_params,
    )
    grads = clip_grads_per_sample_norm(grads, grad_clip)[0]
    return grads  # Shape: ((n_steps, num_envs, dim_IL_param), (n_steps, num_envs, dim_AA_param)),  grads[0]: insulation_level, grads[1]: airflow_adjustment


def create_update_step(
    env: BuildingThermalControl,
    env_params_create: EnvParams,
    config: Dict,
) -> Callable:
    config_lower_optimisation = config["lower_optimisation"]
    config_upper_optimisation = config["upper_optimisation"]
    get_trajectory_batch = create_trajectory_batch_sample(
        config_upper_optimisation,
        env,
        for_eval=False,
    )
    get_trajectory_batch_eval = create_trajectory_batch_sample(
        config_upper_optimisation,
        env,
        for_eval=True,
    )
    regularized_lqr = create_regularized_lqr(
        env, config_lower_optimisation
    )

    def update_step(carry, step_input):
        (
            rng_carry,
            env_params_train_carry,
            upper_level_train_states_carry,
            lower_level_train_states_carry,
            value_function_estimators_carry,
        ) = carry
        t, xi_idx = step_input

        # Train the lower-level
        lower_level_states_output, lower_metrics, _ = regularized_lqr(
            env_params_train_carry,
            lower_level_train_states_carry,
        )
        LL_value_fn = create_state_value_fn(lower_level_states_output)

        # Sample a trajectory batch
        rng_carry, _rng1, _rng2 = jax.random.split(rng_carry, 3)
        traj_batch = get_trajectory_batch(
            _rng1,
            lower_level_states_output,
            env_params_train_carry,
        )  # Shape: (n_steps, num_envs, PyTree Structure of Transition)

        # Sample trajectories for evaluation
        traj_batch_eval = get_trajectory_batch_eval(
            _rng2,
            lower_level_states_output,
            env_params_train_carry,
        )  # Shape: (n_steps, num_envs, PyTree Structure of Transition)

        # Calculate the discounted social upper-level rewards for the trajectory batch
        # UL_reward_discounted: upper-level value
        UL_reward, UL_reward_discounted = calculate_discounted_rewards(
            env_params_train_carry,
            lambda s,a,p: env.upper_level_reward(s, a, p)[0],
            traj_batch,
            config_upper_optimisation["discount_factor"],
            initial_value=jnp.zeros((config_upper_optimisation["num_envs"],)),
        )  # Shape: (n_steps, num_envs)

        # GRADIENT ESTIMATION

        # Data preparation for value estimation
        X = traj_batch.obs  # Shape: (n_steps, num_envs, obs_dim)
        X_next = jnp.where(
            jnp.expand_dims(traj_batch.done, axis=list(range(2, X.ndim))),
            jnp.full_like(X, jnp.nan, dtype=X.dtype),
            jnp.roll(X, shift=-1, axis=0),
        )

        # UPPER-LEVEL PARTIAL GRADIENTS ESTIMATION
        # Upper-level reward gradient
        UL_reward_grad = upper_level_reward_gradient(
            env,
            env_params_train_carry,
            traj_batch,
            upper_level_train_states_carry,
        )  # Shape: ((n_steps, num_envs, dim_IL_param), (n_steps, num_envs, dim_AA_param))
        UL_reward_grad_IL = UL_reward_grad[0]["params"]["weights"]  # Shape: (n_steps, num_envs, dim_IL_param)
        UL_reward_grad_AA = UL_reward_grad[1]["params"]["weights"]  # Shape: (n_steps, num_envs, dim_AA_param)

        # Transition dynamics log prob gradient
        transition_logprob_grad = calculate_transition_logprob_gradient(
            env,
            env_params_train_carry,
            traj_batch,
            upper_level_train_states_carry,
            grad_clip=config_upper_optimisation["transition_logprob_grad_clip"],
        )  # Shape: ((n_steps, num_envs, dim_IL_param), (n_steps, num_envs, dim_AA_param))

        # Upper-level value estimation
        value_model_params = config_upper_optimisation["value_model_params"]
        _, UL_value_estimate, _ = estimate_value_function(
            X,
            X_next,
            UL_reward,
            value_function_estimators_carry["UL_value"],
            num_steps=value_model_params["num_training_steps"],
            discount_factor=config_upper_optimisation["discount_factor"],
            l2_reg=0.0
        )  # Shape: (n_steps, num_envs)

        UL_value_estimate_normalized = UL_value_estimate
        transition_grad_IL = transition_logprob_grad[0]["params"][
            "weights"
        ] * jnp.expand_dims(
            UL_value_estimate_normalized, -1
        )  # Shape: (n_steps, num_envs, dim_IL_param)
        transition_grad_AA = transition_logprob_grad[1]["params"][
            "weights"
        ] * jnp.expand_dims(
            UL_value_estimate_normalized, -1
        )  # Shape: (n_steps, num_envs, dim_AA_param)
        upper_level_grad_IL = (
            UL_reward_grad_IL + transition_grad_IL
        )  # Shape: (n_steps, num_envs, dim_IL_param)
        upper_level_grad_AA = (
            UL_reward_grad_AA + transition_grad_AA
        )  # Shape: (n_steps, num_envs, dim_AA_param)

        # COLLECT GRADIENTS
        traj_batch_discounting = jnp.power(
            config_upper_optimisation["discount_factor"], traj_batch.state.time
        )  # Shape: (n_steps, num_envs)
        num_episodes = jnp.sum(traj_batch.done)  # Shape: ()

        # Insulation Level Grad
        IL_grad = upper_level_grad_IL  # Shape: (n_steps, num_envs, dim_IL_param)
        IL_grad = (
            jnp.nansum(
                IL_grad * jnp.expand_dims(traj_batch_discounting, -1), axis=(0, 1)
            )
            / jnp.clip(num_episodes, a_min=1)
        )  # Shape: (dim_IL_param,)

        # Airflow Adjustment Grad
        AA_grad = upper_level_grad_AA  # Shape: (n_steps, num_envs, dim_AA_param)
        AA_grad = (
            jnp.nansum(
                AA_grad * jnp.expand_dims(traj_batch_discounting, -1), axis=(0, 1)
            )
            / jnp.clip(num_episodes, a_min=1)
        )  # Shape: (dim_AA_param,)

        grad = {
            "insulation_level": {"params": {"weights": -IL_grad}},
            "airflow_adjustment": {"params": {"weights": -AA_grad}},
        }

        # Update the upper-level training states
        upper_level_train_states_carry = {
            key: ts.apply_gradients(
                grads=flax.core.FrozenDict(grad[key])
                if jax.__version__ == "0.4.10"
                else grad[key],
            )
            for key, ts in upper_level_train_states_carry.items()
        }

        # Output metrics
        train_return_UL = jnp.where(
            traj_batch.state.time == 0,
            UL_reward_discounted,
            jnp.nan,
        )  # Shape: (n_steps, num_envs)
        train_return_UL = jnp.nanmean(train_return_UL)
        discounting_arr = jnp.power(
            config["lower_optimisation"]["discount_factor"],
            traj_batch.state.time,
        )  # Shape: (n_steps, num_envs)
        discounting_arr_eval = jnp.power(
            config["lower_optimisation"]["discount_factor"],
            traj_batch_eval.state.time,
        )  # Shape: (n_steps, num_envs)
        num_episodes_eval = jnp.sum(traj_batch_eval.done)  # Shape: ()
        UL_rewards_eval, info = jax.vmap(
            jax.vmap(
                env.upper_level_reward,
                in_axes=(0, 0, None),
            ),
            in_axes=(0, 0, None),
        )(
            traj_batch_eval.state,
            traj_batch_eval.action,
            env_params_train_carry,
        )  # Shape: (n_steps, num_envs)
        return_UL_stability = jnp.sum(
            info["stability_reward"] * discounting_arr_eval
        ) / jnp.clip(num_episodes_eval, a_min=1)
        return_UL_energy_cost = jnp.sum(
            info["energy_cost"] * discounting_arr_eval
        ) / jnp.clip(num_episodes_eval, a_min=1)
        return_UL_insulation_cost = jnp.sum(
            info["insulation_cost"] * discounting_arr_eval
        ) / jnp.clip(num_episodes_eval, a_min=1)
        return_UL_airflow_cost = jnp.sum(
            info["airflow_cost"] * discounting_arr_eval
        ) / jnp.clip(num_episodes_eval, a_min=1)
        return_UL = (
            env_params_train_carry.reward_params.stability_weight * return_UL_stability
            - env_params_train_carry.reward_params.energy_weight * return_UL_energy_cost
            - env_params_train_carry.reward_params.insulation_cost_weight * return_UL_insulation_cost
            - env_params_train_carry.reward_params.airflow_cost_weight * return_UL_airflow_cost
        )
        train_return_LL = jnp.sum(
            traj_batch.reward * discounting_arr
        ) / jnp.clip(num_episodes, a_min=1)
        return_LL = jnp.sum(
            traj_batch_eval.reward * discounting_arr_eval
        ) / jnp.clip(num_episodes_eval, a_min=1)
        V_LL = jax.vmap(
            jax.vmap(
                LL_value_fn,
                in_axes=(0,),
            ),
            in_axes=(0,),
        )(
            traj_batch.state.temperature_deviations
        )  # Shape: (n_steps, num_envs)
        episode_length = config["environment"]["params"]["max_steps_in_episode"]
        metrics = {
            "return_UL": return_UL,
            "return_UL_stability": return_UL_stability,
            "return_UL_energy_cost": return_UL_energy_cost,
            "return_UL_insulation_cost": return_UL_insulation_cost,
            "return_UL_airflow_cost": return_UL_airflow_cost,
            "return_LL": return_LL,
            "return_UL_train": train_return_UL,
            "return_LL_train": train_return_LL,
            "V_UL": jnp.nanmean(UL_value_estimate),
            "V_LL": jnp.nanmean(V_LL),
            "insulation_level": env_params_train_carry.transition_params.insulation_level,
            "airflow_adjustment": env_params_train_carry.transition_params.airflow_adjustment,
            "insulation_level_grad": IL_grad,
            "airflow_adjustment_grad": AA_grad,
            "traj_batch_last_obs_mean": jnp.mean(traj_batch.obs[episode_length-1], 0),  # Mean over num_envs
            "traj_batch_last_action_mean": jnp.mean(traj_batch.action[episode_length-1], 0),  # Mean over num_envs
            "LQR_Riccati_error_mean": lower_metrics["Riccati_error"],
            "LQR_value_error_mean": lower_metrics["value_error"],
            "LQR_total_iterations": lower_metrics["total_steps"],
            "LQR_positive_definit_S_rate": jnp.nanmean(jnp.where(
                lower_metrics["logdet_sign"] < 0, 0.0, lower_metrics["logdet_sign"]
            )),
            "LQR_controllability": lower_metrics["controllability"],
        }
        env_params_train_carry = update_transition_params(
            env_params_train_carry, upper_level_train_states_carry
        )
        return (
            rng_carry,
            env_params_train_carry,
            upper_level_train_states_carry,
            lower_level_train_states_carry,
            value_function_estimators_carry,
        ), metrics

    return update_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir", type=str, help="Path to the experiment directory"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to the checkpoint file"
    )
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    checkpoint = args.checkpoint
    print("Output directory: ", experiment_dir)
    if checkpoint is not None:
        print("Resuming from checkpoint: ", checkpoint)
    print("Device used: ", jax.devices())
    print("Number of devices: ", jax.local_device_count())

    # Read config
    with open(f"{experiment_dir}/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Config: ", config)
    rng = jax.random.PRNGKey(config["random_seed"])
    config_init = deepcopy(config)

    # Create the update dictionary
    update_dict = remove_non_list_entries(
        config,
        list_parameters=[
            "init_temp_mean",
            "insulation_effect",
            "temperature_range",
            "hvac_range",
            "hidden_layers",
            "scale",
            "layer_size",
        ],
        matrix_parameters=[
            "Q",
            "R",
            "A",
            "B",
        ],
    )
    update_dict = jax.tree_map(
        lambda x: jnp.array(x), update_dict, is_leaf=lambda x: isinstance(x, list)
    )
    leaves, tree_structure = jax.tree_util.tree_flatten(
        update_dict, is_leaf=lambda x: isinstance(x, jax.Array)
    )
    leaves_idx = [jnp.arange(len(leaf)) for leaf in leaves]
    meshgrid = jnp.meshgrid(*leaves_idx)
    update_dict = jax.tree_map(
        lambda idx, x: x[idx.reshape(-1), ...],
        jax.tree_util.tree_unflatten(tree_structure, meshgrid),
        update_dict,
    )
    print("Update dict: ", update_dict)
    if len(update_dict) > 0:
        n_configs = len(jax.tree_util.tree_leaves(update_dict)[0])

    # Create environment
    basic_env, basic_env_params = setup_environment(config_init["environment"])
    print("Basic Env params: ", basic_env_params)

    # Method name
    method_name = "baseline"

    # Resume from checkpoint if provided
    if checkpoint is not None:
        checkpoint_path = os.path.join(checkpoint, f"checkpoint_incentive_{method_name}")
        checkpoint_config_path = os.path.join(checkpoint, "config.yaml")
        checkpoinct_metrics_path = os.path.join(
            checkpoint, f"metrics_{method_name}.pkl"
        )
        if not os.path.exists(checkpoint_path):
            raise ValueError("Checkpoint file does not exist: ", checkpoint_path)
        if not os.path.exists(checkpoint_config_path):
            raise ValueError("Checkpoint config file does not exist: ", checkpoint_config_path)
        with open(checkpoint_config_path, "r") as f:
            checkpoint_config = yaml.safe_load(f)
        with open(checkpoinct_metrics_path, "rb") as f:
            checkpoint_metrics = pickle.load(f)
        try:
            checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            restored = checkpointer.restore(checkpoint_path)
        except:
            raise ValueError("Failed to load checkpoint from file: ", checkpoint_path)
        if len(update_dict) > 0:
            if ((not len(restored["key"].shape) == 3)
                or (not restored["key"].shape[0] == config["num_seeds"])
                or (not restored["key"].shape[1] == n_configs)):
                raise ValueError(
                    "Checkpoint shape does not match the shape of update_dict"
                )
        else:
            if ((not len(restored["key"].shape) == 2)
                or (not restored["key"].shape[0] == config["num_seeds"])):
                raise ValueError(
                    "Checkpoint shape does not match the shape of update_dict"
                )

    def run_experiment(
        key: jax.random.PRNGKey,
        config_update: Dict[str, Any],
        env_params_exp: EnvParams,
        seed_idx: int,
        param_idx: int,
    ) -> Tuple[
            Tuple[jax.Array, EnvParams, Dict[str, TrainState], jax.Array],
            Tuple[jax.Array],
        ]:
        config_exp = deepcopy(config_init)
        config_exp = update_dictionary(config_exp, config_update)
        env_params_exp = update_nested_pytree(
            env_params_exp, config_exp["environment"]["params"]
        )
        env_params_exp = env_params_exp.replace(
            transition_params=env_params_exp.transition_params.replace(
                num_zones=env_params_exp.transition_params.A.shape[0],
                num_hvac_units=env_params_exp.transition_params.B.shape[1],
            )
        )

        n_dim_insulation_level = env_params_exp.transition_params.num_zones
        n_dim_airflow_adjustment = env_params_exp.transition_params.num_zones
        config_upper_optimisation_model = config_exp["upper_optimisation"][
            "model_params"
        ]
        config_value_model = config_exp["upper_optimisation"]["value_model_params"]
        obs_dim = env_params_exp.transition_params.num_zones

        if checkpoint is None:
            # Initialize the upper level
            
            key, _rng1, _rng2 = jax.random.split(key, 3)
            init_value = {
                "insulation_level": jnp.array(
                    config_exp["upper_optimisation"]["init_insulation_level"],
                    dtype=jnp.float32,
                ) if config_exp["upper_optimisation"]["init_insulation_level"] is not None
                else None,
                "airflow_adjustment": jnp.array(
                    config_exp["upper_optimisation"]["init_airflow_adjustment"],
                    dtype=jnp.float32,
                ) if config_exp["upper_optimisation"]["init_airflow_adjustment"] is not None
                else None,
            }    
            upper_level_train_states = {
                "insulation_level": create_static_train_state(
                    param_shape=(n_dim_insulation_level,),
                    key=_rng1,
                    init_value=init_value["insulation_level"],
                    **config_upper_optimisation_model,
                ),
                "airflow_adjustment": create_static_train_state(
                    param_shape=(n_dim_airflow_adjustment,),
                    key=_rng2,
                    init_value=init_value["airflow_adjustment"],
                    **config_upper_optimisation_model,
                ),
            }
            lower_level_train_states = RegularizedLQROutputs(
                A=env_params_exp.transition_params.A,
                B=env_params_exp.transition_params.B,
                Q=env_params_exp.reward_params.Q,
                R=env_params_exp.reward_params.R,
                W=jnp.eye(env_params_exp.transition_params.A.shape[0]),
                P=jnp.copy(env_params_exp.reward_params.Q),
                v=jnp.array(0.0),
                Ks=jnp.zeros((
                    env_params_exp.transition_params.B.shape[1],  # action_dim
                    env_params_exp.transition_params.A.shape[0]   # state_dim
                )),
                Sigma=jnp.eye(env_params_exp.transition_params.B.shape[1]),  # action_dim x action_dim
                sqrtSigma=jnp.eye(env_params_exp.transition_params.B.shape[1])  # action_dim x action_dim
            )
            # Initialize the value function estimators
            key, _rng1 = jax.random.split(key, 2)
            value_function_estimators = {
                "UL_value": create_train_state_value_model(
                    key=_rng1,
                    input_dim=obs_dim,
                    output_dim=1,
                    layer_size=config_value_model["layer_size"],
                    optimizer_params=config_value_model["optimizer_params"],
                ),
            }
            start_itr = 0
        else:
            # Resume from checkpoint
            if len(update_dict) > 0:
                upper_level_train_states_dict = jax.tree_map(
                    lambda x: x[seed_idx, param_idx],
                    restored["upper_level_train_states"]
                )
                lower_level_train_states_dict = jax.tree_map(
                    lambda x: x[seed_idx, param_idx],
                    restored["lower_level_train_states"],
                )
                value_function_estimators_dict = jax.tree_map(
                    lambda x: x[seed_idx, param_idx],
                    restored["value_function_estimators"]
                )
                key = restored["key"][seed_idx, param_idx]
            else:
                upper_level_train_states_dict = jax.tree_map(
                    lambda x: x[seed_idx],
                    restored["upper_level_train_states"]
                )
                lower_level_train_states_dict = jax.tree_map(
                    lambda x: x[seed_idx],
                    restored["lower_level_train_states"],
                )
                value_function_estimators_dict = jax.tree_map(
                    lambda x: x[seed_idx],
                    restored["value_function_estimators"]
                )
                key = restored["key"][seed_idx]
            upper_level_train_states = {
                "insulation_level": restore_static_train_state(
                    upper_level_train_states_dict["insulation_level"],
                    param_shape=(n_dim_insulation_level,),
                    **config_upper_optimisation_model
                ),
                "airflow_adjustment": restore_static_train_state(
                    upper_level_train_states_dict["airflow_adjustment"],
                    param_shape=(n_dim_airflow_adjustment,),
                    **config_upper_optimisation_model
                ),
            }
            lower_level_train_states = RegularizedLQROutputs(**lower_level_train_states_dict)
            value_function_estimators = {
                "UL_value": restore_train_state_value_model(
                    value_function_estimators_dict["UL_value"],
                    output_dim=1,
                    layer_size=config_value_model["layer_size"],
                    optimizer_params=config_value_model["optimizer_params"],
                ),
            }
            start_itr = checkpoint_config["upper_optimisation"]["num_outer_iter"]

        # TRAINING
        env_params_exp = update_transition_params(  # Set the initial upper-level params
            env_params_exp, upper_level_train_states
        )
        update_step = create_update_step(basic_env, env_params_exp, config_exp)
        n_iter = config_exp["upper_optimisation"]["num_outer_iter"]
        time_array = jnp.arange(start_itr + 1, n_iter + 1)
        xi_idx_arr = jnp.zeros_like(time_array)  # Not used in this experiment setup
        carry, metrics = jax.lax.scan(
            update_step,
            (
                key,
                env_params_exp,
                upper_level_train_states,
                lower_level_train_states,
                value_function_estimators,
            ),
            (time_array, xi_idx_arr),
            n_iter - start_itr,
        )
        metrics["xi_idx"] = xi_idx_arr
        return carry, metrics

    # RUN EXPERIMENT
    start_time = time.time()
    if len(update_dict) > 0:
        run_experiment_vmap = jax.vmap(
            jax.vmap(
                run_experiment,
                in_axes=(None, jax.tree_map(lambda x: 0, update_dict), None, None, 0),
            ),
            in_axes=(0, None, None, 0, None),
        )
        carry_out, output_metrics = jax.block_until_ready(
            jax.jit(run_experiment_vmap)(
                jax.random.split(rng, config_init["num_seeds"]),
                update_dict,
                basic_env_params,
                jnp.arange(config_init["num_seeds"]),
                jnp.arange(n_configs)
            )
        )
    else:
        run_experiment_vmap = jax.vmap(run_experiment, in_axes=(0, None, None, 0, None))
        carry_out, output_metrics = jax.block_until_ready(
            jax.jit(run_experiment_vmap)(
                jax.random.split(rng, config_init["num_seeds"]),
                update_dict,
                basic_env_params,
                jnp.arange(config_init["num_seeds"]),
                None,
            )
        )
    run_time = time.time() - start_time
    print(
        f"Experiment runtime: {(run_time) / 60:.2f} minutes and {(run_time) % 60:.2f} seconds"
    )
    (
        key_out,
        env_params_out,
        upper_level_train_states_out,
        lower_level_train_states_out,
        value_function_estimators_out,
    ) = carry_out

    if checkpoint is not None:
    # Concatenate checkpoint_metrics and output_metrics
        itr_axis = 2 if len(update_dict) > 0 else 1
        output_metrics = jax.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=itr_axis),
            checkpoint_metrics,
            output_metrics,
        )

    # SAVE RESULTS
    with open(os.path.join(experiment_dir, f"metrics_{method_name}.pkl"), "wb") as f:
        pickle.dump(output_metrics, f)
    with open(os.path.join(experiment_dir, f"update_dict_{method_name}.pkl"), "wb") as f:
        pickle.dump(update_dict, f)

    checkpoint_data = {
        "key": key_out,
        "upper_level_train_states": upper_level_train_states_out,
        "lower_level_train_states": lower_level_train_states_out,
        "value_function_estimators": value_function_estimators_out,
    }

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(upper_level_train_states_out)
    orbax_checkpointer.save(
        os.path.join(os.path.abspath(experiment_dir), f"checkpoint_incentive_{method_name}"),
        checkpoint_data,
        force=True,
    )
