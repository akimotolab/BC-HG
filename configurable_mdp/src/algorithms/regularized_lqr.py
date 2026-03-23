import flax.struct
import jax
import jax.numpy as jnp
from jax.numpy.linalg import inv, slogdet, cholesky

import gymnax

from typing import Dict, Sequence, Union
from src.environments.BuildingThermalControl import BuildingThermalControl
from src.environments.BuildingThermalControl import EnvParams

import collections.abc


def update_dictionary(dictionary, update):
    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping):
            dictionary[k] = update_dictionary(dictionary.get(k, {}), v)
        else:
            dictionary[k] = v
    return dictionary


@flax.struct.dataclass
class Transition:
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    state: jax.Array = None


@flax.struct.dataclass
class RegularizedLQROutputs:
    A: jax.Array
    B: jax.Array
    Q: jax.Array
    R: jax.Array
    W: jax.Array
    P: jax.Array
    v: float
    Ks: jax.Array
    Sigma: jax.Array
    sqrtSigma: jax.Array


class RegularizedLQRPolicy:
    def __init__(self, Ks: jax.Array, Sigma: jax.Array, sqrtSigma: jax.Array):
        self.Ks = Ks
        self.Sigma = Sigma
        self.sqrtSigma = sqrtSigma

    def get_action(self, obs: jax.Array, key: jax.Array) -> jax.Array:
        # Ensure proper vector shape for matrix operations
        obs_vec = jnp.atleast_1d(obs.reshape(-1))
        mean_action = - self.Ks @ obs_vec
        noise = jax.random.normal(key, shape=mean_action.shape) @ self.sqrtSigma.T
        return mean_action + noise

    def get_actions(self, obs: jax.Array, keys: jax.Array) -> jax.Array:
        # Ensure proper vector shape for matrix operations
        obs_reshaped = obs.reshape(obs.shape[0], -1)  # (batch_size, obs_dim)
        mean_actions = - jnp.einsum('ij,bj->bi', self.Ks, obs_reshaped)  # (batch_size, action_dim)
        noises = jax.vmap(
            lambda key: jax.random.normal(
                key, 
                shape=mean_actions[0].shape
            ) @ self.sqrtSigma.T
        )(keys)
        return mean_actions + noises


def create_state_value_fn(lqr_outputs: RegularizedLQROutputs):
    def state_value_fn(obs: jax.Array) -> jax.Array:
        # Ensure proper vector shape for matrix operations
        obs_vec = jnp.atleast_1d(obs.reshape(-1))
        negative_value = obs_vec @ lqr_outputs.P @ obs_vec + lqr_outputs.v
        return -negative_value
    return state_value_fn


def create_q_value_fn(lqr_outputs: RegularizedLQROutputs, gamma: float):
    def q_value_fn(obs: jax.Array, action: jax.Array) -> jax.Array:
        # Ensure proper vector shapes for matrix operations
        obs_vec = jnp.atleast_1d(obs.reshape(-1))
        action_vec = jnp.atleast_1d(action.reshape(-1))
        
        expected_next_obs = lqr_outputs.A @ obs_vec + lqr_outputs.B @ action_vec
        expected_next_value = (
            expected_next_obs @ lqr_outputs.P @ expected_next_obs 
            + jnp.sum(lqr_outputs.P * lqr_outputs.W)
            + lqr_outputs.v
        )
        r = obs_vec @ lqr_outputs.Q @ obs_vec + action_vec @ lqr_outputs.R @ action_vec
        return r + gamma * expected_next_value
    return q_value_fn


def create_regularized_lqr(env: BuildingThermalControl, config: Dict):

    compute_parameterized_A = env.compute_parameterized_A
    compute_process_noise_covariance = env.compute_process_noise_covariance
    tol = config["training"]["tol"]
    max_steps = config["training"]["max_steps"]
    gamma  = config["discount_factor"]
    beta = config["reg_lambda"]

    @jax.jit
    def regularized_lqr(
        env_params_train: EnvParams, train_states_carry: RegularizedLQROutputs,
    ) -> Union[RegularizedLQROutputs, Dict]:
        """Regularized LQR solver via value iteration."""

        # Initialize variables
        A = compute_parameterized_A(env_params_train.transition_params)
        B = env_params_train.transition_params.B
        Q = env_params_train.reward_params.Q
        R = env_params_train.reward_params.R
        W = compute_process_noise_covariance(env_params_train.transition_params)
        m = B.shape[1]

        # Check controllability
        def _matrix_powers(A, num):
            def step(carry, _):
                next_power = carry @ A
                return next_power, next_power
            init = jnp.eye(A.shape[0], dtype=A.dtype)
            _, powers = jax.lax.scan(step, init, None, length=num)
            return jnp.concatenate([init[None], powers], axis=0)  # shape: (num+1, n, n)
        
        powers = _matrix_powers(A, A.shape[0]-1)  # 0 to n-1 powers
        AB_blocks = jax.vmap(lambda Ap: Ap @ B)(powers)
        C = jnp.hstack(AB_blocks)
        rank = jnp.linalg.matrix_rank(C)
        is_controllable = rank == A.shape[0]

        # TRAINING LOOP
        def _update_step(runner_state):
            (P, v, step, metrics_array) = runner_state

            S = R + gamma * (B.T @ P @ B)
            S_inv = inv(S)

            # Riccati-like update for P
            P_new = Q + gamma * A.T @ P @ A \
                    - (gamma**2) * A.T @ P @ B @ S_inv @ B.T @ P @ A
            
            S_tmp = R + gamma * (B.T @ P_new @ B)
            sign, logdetS = slogdet(S_tmp)  # stable log-det
            eps = jnp.array(1e-8, dtype=P_new.dtype)

            # If S_tmp is not positive definite (sign<=0), add tiny jitter in a JAX-friendly way
            def _fix(_):
                S_fixed = S_tmp + eps * jnp.eye(S_tmp.shape[-1], dtype=S_tmp.dtype)
                return slogdet(S_fixed)
            def _keep(_):
                return (sign, logdetS)
            sign, logdetS = jax.lax.cond(sign <= 0.0, _fix, _keep, operand=None)
            
            v_new = (
                - beta * (m / 2.0) * jnp.log(jnp.array(jnp.pi * beta, dtype=P_new.dtype))
                + 0.5 * beta * logdetS
                + gamma * jnp.trace(P_new @ W)
            ) / (1.0 - gamma)

            # Relative Riccati error
            P_norm = jnp.linalg.norm(P, ord="fro")
            riccati_error_abs = jnp.linalg.norm(P_new - P, ord="fro")
            riccati_error_rel = riccati_error_abs / (P_norm + 1e-8)

            metrics = jnp.array([
                riccati_error_rel,   # Riccati_error
                jnp.abs(v_new - v),  # value_error
                sign,                # logdet_sign
            ])
            metrics_array = metrics_array.at[step].set(metrics)

            return (P_new, v_new, step + 1, metrics_array)
        
        def _cond_fn(runner_state):
            (_, _, step, metrics_array) = runner_state
            max_steps_reached = step >= max_steps

            # Check convergence
            def check_convergence():
                current_metrics = metrics_array[step - 1]
                return jnp.any(current_metrics[:2] > tol)  # Riccati_error > tol or value_error > tol
            
            def always_continue():
                return True
            
            not_converged = jax.lax.cond(
                step == 0,
                always_continue,
                check_convergence
            )

            continue_condition = ~max_steps_reached & not_converged
            return continue_condition

        # initialization
        P = jnp.copy(train_states_carry.P)
        v = jnp.copy(train_states_carry.v)
        step = 0
        metrics_array = jnp.full((max_steps, 3), jnp.inf)  # [Riccati_error, value_error, controllability]
        
        initial_train_state = (P, v, step, metrics_array)
        train_state = jax.lax.while_loop(_cond_fn, _update_step, initial_train_state)

        # Outputs
        (P, v, step, metrics_array) = train_state

        S = R + gamma * B.T @ P @ B
        S_inv = inv(S)
        Ks = gamma * S_inv @ B.T @ P @ A
        Sigma = (beta / 2.0) * S_inv

        outputs = train_states_carry.replace(
            A=A,         # (state_dim, state_dim)
            B=B,         # (state_dim, action_dim)
            Q=Q,         # (state_dim, state_dim)
            R=R,         # (action_dim, action_dim)
            W=W,         # (state_dim, state_dim)
            P=P,         # (state_dim, state_dim)
            v=v,         # scalar
            Ks=Ks,       # (action_dim, state_dim)
            Sigma=Sigma, # (action_dim, action_dim)
            sqrtSigma=cholesky(Sigma)
        )

        valid_metrics = jnp.where(
            jnp.arange(max_steps)[:, None] < step,
            metrics_array,
            jnp.nan
        )

        metrics = {
            "Riccati_error": valid_metrics[step-1, 0],  # Shape: ()
            "value_error": valid_metrics[step-1, 1],    # Shape: ()
            "logdet_sign": valid_metrics[:, 2],         # Shape: (max_steps,)
            "total_steps": step,                        # Shape: ()
            "controllability": is_controllable,         # Shape: ()
        }

        return outputs, metrics, valid_metrics
    
    return regularized_lqr