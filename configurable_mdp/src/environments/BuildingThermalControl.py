import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Iterable, Tuple, Optional, Dict, Callable, Union
from dataclasses import field

from flax import struct


@struct.dataclass
class EnvState:
    temperature_deviations: jnp.ndarray
    time: int


@struct.dataclass
class RewardParams:
    Q: jnp.ndarray = field(default_factory=lambda: jnp.array([[8.0, 0.0, 0.0, 0.0], 
                                                              [0.0, 1.0, 0.0, 0.0], 
                                                              [0.0, 0.0, 5.0, 0.0], 
                                                              [0.0, 0.0, 0.0, 6.0]], dtype=jnp.float32))  # Shape: (4, 4)
    R: jnp.ndarray = field(default_factory=lambda: jnp.array([[0.05, 0.0], 
                                                              [0.0, 0.05]], dtype=jnp.float32))  # Shape: (2, 2)
    # Upper-level
    stability_weight: float = 1.0
    energy_weight: float = 0.5
    insulation_cost_weight: float = 0.1
    airflow_cost_weight: float = 0.1


@struct.dataclass
class TransitionParams:
    # Constant parameters
    num_zones: int = 4
    A: jnp.ndarray = field(
        default_factory=lambda: jnp.array([[1.0,  0.05,  0.0, 0.05], 
                                           [0.03, 1.0,  0.04,  0.0], 
                                           [0.0,  0.04,  1.0, 0.06], 
                                           [0.05, 0.0,  0.03,  1.0]], dtype=jnp.float32)
    )  # Shape: (4, 4)
    insulation_effect: jnp.ndarray = field(
        default_factory=lambda: jnp.array([-0.08, -0.06, -0.1, -0.09], dtype=jnp.float32)
    )  # Shape: (4,)
    num_hvac_units: int = 2
    B: jnp.ndarray = field(
        default_factory=lambda: jnp.array([[0.1, 0.0], 
                                           [0.6, 0.0], 
                                           [0.0, 0.55], 
                                           [0.0, 0.3]], dtype=jnp.float32)
    )  # Shape: (4, 2)
    transition_std: float = 0.02  # W: jnp.diag(jnp.array([0.2, 0.2, 0.2, 0.2]))
    temperature_range: Tuple[float, float] = (-30.0, 30.0)  # Min and Max temperature deviation
    hvac_range: Tuple[float, float] = (-10.0, 10.0)  # Min and Max HVAC power
    """
    Configurable upper parameters:
        insulation_level[i] \in [0, 1]: 0 means no insulation, 1 means max insulation for zone i 
        airflow_adjustment[i] \in [0, 1]: 0 means max, and 1 means min air exchange between zone i and i+1 (zone 4 exchanges with zone 1)
    """
    insulation_level: jnp.ndarray = None  # Shape: (4,)
    airflow_adjustment: jnp.ndarray = None  # Shape: (4,)


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 100
    reward_params: RewardParams = field(default_factory=RewardParams)
    transition_params: TransitionParams = field(default_factory=TransitionParams)
    init_temp_mean: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32))
    init_temp_std: float = 1.0


class BuildingThermalControl(environment.Environment):
    
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        """Environment name."""
        return "BuildingThermalControl"

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Space:
        """Action space of the environment."""
        action_range = params.transition_params.hvac_range
        action_dim = params.transition_params.num_hvac_units
        return spaces.Box(action_range[0], action_range[1], 
                          (action_dim,), jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Space:
        """Observation space of the environment"""
        obs_range = params.transition_params.temperature_range
        obs_dim = params.transition_params.num_zones
        return spaces.Box(obs_range[0], obs_range[1], 
                          (obs_dim,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Space:
        """State space of the environment."""
        return spaces.Dict(
            {
                "obs": self.observation_space(params),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check if the state is terminal."""
        return state.time >= params.max_steps_in_episode

    def reward(
        self,
        state: EnvState,
        action: jax.Array,
        params: RewardParams,
    ) -> jax.Array:
        """Compute reward from state, action."""
        # Ensure we have proper vector shapes for quadratic forms
        temp_dev = jnp.atleast_1d(state.temperature_deviations.reshape(-1))  # Force to 1D
        action_arr = jnp.atleast_1d(action.reshape(-1))  # Force to 1D
        
        state_cost = temp_dev @ params.Q @ temp_dev
        action_cost = action_arr @ params.R @ action_arr
        raw_reward = - state_cost - action_cost
        return raw_reward

    def transition(
        self, 
        state: EnvState, 
        action: jax.Array, 
        params: TransitionParams, 
        soft_clipping: bool = False
    ) -> EnvState:
        """Compute next state from state, action."""
        # Compute parametrized A matrix
        parametrized_A = self.compute_parameterized_A(params)
        B = params.B
        new_temp_deviations = (
            parametrized_A @ state.temperature_deviations + B @ action
        )

        # Clip temperature deviations to the range
        range_min, range_max = params.temperature_range
        if soft_clipping:
            center = (range_max + range_min) / 2
            scale = (range_max - range_min) / 2
            new_temp_deviations = center + scale * jnp.tanh((new_temp_deviations - center) / scale)
        else:
            new_temp_deviations = jnp.clip(
                new_temp_deviations,
                range_min,
                range_max,
            )

        return EnvState(temperature_deviations=new_temp_deviations, time=state.time + 1)

    def add_transition_noise(
        self, 
        key: jax.random.PRNGKey, 
        new_state: EnvState, 
        params: TransitionParams
    ) -> EnvState:
        """
        Add truncated normal noise to the transition.
        """
        std = params.transition_std
        range_min, range_max = params.temperature_range
        new_temp_deviations = new_state.temperature_deviations
        noise = jax.random.truncated_normal(
            key=key,
            lower=(range_min - new_temp_deviations) / std,
            upper=(range_max - new_temp_deviations) / std,
            shape=new_temp_deviations.shape,
        )
        new_temp_deviations = new_temp_deviations + std * noise
        new_state = new_state.replace(temperature_deviations=new_temp_deviations)
        return new_state

    def step_env(
        self,
        key: jax.random.PRNGKey,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, bool, Dict[str, float]]:
        """Execute a step of the environment."""
        # Compute reward
        reward = self.reward(
            state,
            action,
            params.reward_params,
        )

        # Update state
        new_state = self.transition(state, action, params.transition_params)
        new_state = jax.tree_map(
            lambda x, y: jax.lax.select(
                params.transition_params.transition_std > 0, x, y
            ),
            self.add_transition_noise(key, new_state, params.transition_params),
            new_state
        )

        # Upper-level immediate reward
        upper_level_reward = self.upper_level_reward(
            state,
            action,
            params,
        )

        done = self.is_terminal(new_state, params)
        return (
            lax.stop_gradient(self.get_obs(new_state)),
            lax.stop_gradient(new_state),
            reward,
            done,
            {
                "discount": self.discount(new_state, params),
                "upper_level_reward": upper_level_reward,
            },
        )

    def reset_env(
        self, key: jax.random.PRNGKey, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        std = params.transition_params.transition_std
        range_min, range_max = params.transition_params.temperature_range
        noise = jax.random.truncated_normal(
            key=key,
            lower=(range_min -  params.init_temp_mean) / std,
            upper=(range_max -  params.init_temp_mean) / std,
            shape= params.init_temp_mean.shape,
        )
        init_obs = params.init_temp_mean + params.init_temp_std * noise
        state = EnvState(
            temperature_deviations=init_obs,
            time=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> jax.Array:
        """Return observation from raw state."""
        return state.temperature_deviations

    def upper_level_reward(
        self,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> jax.Array:
        """
        Compute upper-level immediate reward from state.
        
        The upper-level reward captures the long-term building efficiency objectives:
        - Energy efficiency (lower HVAC usage)
        - Temperature stability (lower variance)
        - Insulation and airflow adjustment costs
        """
        reward_params = params.reward_params
        transition_params = params.transition_params
        
        # Energy efficiency: penalize high HVAC usage
        action_arr = jnp.atleast_1d(action.reshape(-1))
        energy_cost = jnp.sum(action_arr ** 2)
        
        # Temperature stability: reward lower temperature variance
        temp_dev = jnp.atleast_1d(state.temperature_deviations.reshape(-1))
        stability_reward = -jnp.var(temp_dev)
        
        # Insulation cost: there's a cost for higher insulation levels
        insulation_cost = jnp.sum(transition_params.insulation_level ** 2)
        
        # Airflow adjustment cost: there's a cost for airflow modifications
        airflow_cost = jnp.sum(transition_params.airflow_adjustment ** 2)
        
        # Combined upper-level reward
        a1 = reward_params.stability_weight
        a2 = reward_params.energy_weight
        a3 = reward_params.insulation_cost_weight
        a4 = reward_params.airflow_cost_weight
        raw_reward = a1*stability_reward - a2*energy_cost - a3*insulation_cost - a4*airflow_cost
        info = {
            'stability_reward': stability_reward,
            'energy_cost': energy_cost,
            'insulation_cost': insulation_cost,
            'airflow_cost': airflow_cost
        }

        return raw_reward, info

    def compute_process_noise_covariance(self, params: TransitionParams) -> jnp.ndarray:
        """
        Compute process noise covariance matrix W.
        """
        std = params.transition_std
        # Use identity matrix scaled by std^2 for simplicity and JIT compatibility
        W = (std**2) * jnp.eye(params.A.shape[0])
        return W

    def compute_parameterized_A(self, params: TransitionParams) -> jnp.ndarray:
        """
        Compute parametrized A matrix from transition parameters.
        """
        num_zones = params.A.shape[0]  # 既存のA行列のshapeから取得
        insulation_effect = jnp.diag(params.insulation_effect * (1 - params.insulation_level))
        airflow_adjustment_effect = jnp.zeros((num_zones, num_zones))

        def body(i, arr):
            j = (i + 1) % num_zones  # Connected zone j of zone i
            arr = arr.at[i, j].set(1 - params.airflow_adjustment[i])
            return arr

        airflow_adjustment_effect = jax.lax.fori_loop(
            0, num_zones, body, airflow_adjustment_effect
        )
        airflow_adjustment_effect = airflow_adjustment_effect + airflow_adjustment_effect.T
        airflow_adjustment_effect = airflow_adjustment_effect * params.A

        parametrized_A = params.A + insulation_effect + airflow_adjustment_effect
        return parametrized_A


class BuildingThermalControl_2(BuildingThermalControl):
    def compute_parameterized_A(self, params: TransitionParams) -> jnp.ndarray:
        """
        Compute parametrized A matrix from transition parameters.
        """
        num_zones = params.A.shape[0]  # 既存のA行列のshapeから取得
        insulation_effect = jnp.diag(params.insulation_effect * (1 - params.insulation_level))
        
        airflow_adjustment_effect = jnp.zeros((num_zones, num_zones))
        def body(i, arr):
            j = (i + 1) % num_zones  # Connected zone j of zone i
            arr = arr.at[i, j].set(params.airflow_adjustment[i])
            return arr
        airflow_adjustment_effect = jax.lax.fori_loop(
            0, num_zones, body, airflow_adjustment_effect
        )
        airflow_adjustment_effect = airflow_adjustment_effect + airflow_adjustment_effect.T
        airflow_adjustment_effect = airflow_adjustment_effect * params.A

        parametrized_A = params.A + insulation_effect + airflow_adjustment_effect
        return parametrized_A


class BuildingThermalControl_3(BuildingThermalControl):
    def compute_parameterized_A(self, params: TransitionParams) -> jnp.ndarray:
        """
        Compute parametrized A matrix from transition parameters.
        
        Args:
            params: TransitionParams containing A, insulation_effect, insulation_level, airflow_adjustment
            
        Returns:
            jnp.ndarray: Parametrized A matrix with shape (4, 4)
        """
        A = params.A
        insulation_effect = params.insulation_effect
        il = params.insulation_level
        aa = params.airflow_adjustment
        diag = [
            A[0,0] + insulation_effect[0]*(1.0 - il[0]) - A[0,1]*aa[0] - A[0,3]*aa[3],
            A[1,1] + insulation_effect[1]*(1.0 - il[1]) - A[1,2]*aa[1] - A[1,0]*aa[0],
            A[2,2] + insulation_effect[2]*(1.0 - il[2]) - A[2,3]*aa[2] - A[2,1]*aa[1],
            A[3,3] + insulation_effect[3]*(1.0 - il[3]) - A[3,0]*aa[3] - A[3,2]*aa[2]
        ]
        parametrized_A = jnp.array(
            [
                [diag[0],      A[0,1]*aa[0], 0.0,          A[0,3]*aa[3]],
                [A[1,0]*aa[0], diag[1],      A[1,2]*aa[1], 0.0         ],
                [0.0,          A[2,1]*aa[1], diag[2],      A[2,3]*aa[2]],
                [A[3,0]*aa[3], 0.0,          A[3,2]*aa[2], diag[3]     ]
            ]
        )
        return parametrized_A


def setup_environment(config_setup: dict) -> Tuple[BuildingThermalControl, EnvParams]:
    """
    Initialize the environment
    :param config_setup: Configuration dictionary
    :return: Environment and parameters
    """
    config_env_params = config_setup["params"]
    if config_setup["name"] == "BuildingThermalControl":
        env = BuildingThermalControl()
    elif config_setup["name"] == "BuildingThermalControl_2":
        env = BuildingThermalControl_2()
    elif config_setup["name"] == "BuildingThermalControl_3":
        env = BuildingThermalControl_3()
    else:
        raise ValueError("Unknown environment name")
    params = env.default_params
    params = params.replace(
        **{
            # key: jax.tree_map(lambda x: jnp.array(x), value)
            key: jnp.array(value)
            for key, value in config_env_params.items()
            if key not in ["reward_params", "transition_params"]
        },
        reward_params=params.reward_params.replace(
            **{
                key: jnp.array(value) if isinstance(value, Iterable) 
                     else value
                for key, value in config_env_params["reward_params"].items()
            }
        ),
        transition_params=params.transition_params.replace(
            **{
                key: jnp.array(value) if isinstance(value, Iterable) else value
                for key, value in config_env_params["transition_params"].items()
            }
        ),
    )
    return env, params