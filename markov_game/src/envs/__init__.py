from gym import register

from ._environment import (Environment, EnvStep, GlobalEnvSpec,
                           EnvSpec, InOutSpec, Wrapper)
from .gym_env import GymEnv
from .normalized_env import normalize

__all__ = [
    'Environment', 'EnvStep', 'GlobalEnvSpec',
    'EnvSpec', 'InOutSpec', 'Wrapper',
    'GymEnv', 'normalize',
]

# Gym environments registration
register(
    id='GuidedCartPole-v0',
    entry_point=f'{__name__}.guided_cartpole:GuidedCartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
    kwargs={'max_leader_action':1.0, 'target_interval':[-0.25, 0.25]},
)
register(
    id='GuidedCartPole-v1',
    entry_point=f'{__name__}.guided_cartpole:GuidedCartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
    kwargs={'max_leader_action':1.0, 'target_interval':[0.25, 0.75]},
)
register(
    id='GuidedPendulum-v0',
    entry_point=f'{__name__}.guided_pendulum:GuidedPendulumEnv',
    max_episode_steps=200,
    kwargs={'max_leader_action':1.0},
)

# Discrete toy environments registration
register(
    id='DiscreteToy1_1a-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv1_1a',
    max_episode_steps=100,
    reward_threshold=47.5, # 最適なリーダーの下で最適フォロワーリターンの目安
)
register(
    id='DiscreteToy1_1b-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv1_1b',
    max_episode_steps=100,
    reward_threshold=47.5,
)
register(
    id='DiscreteToy1_2a-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv1_2a',
    max_episode_steps=150,
    reward_threshold=372.5,
)
register(
    id='DiscreteToy1_2b-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv1_2b',
    max_episode_steps=150,
    reward_threshold=372.5,
)
register(
    id='DiscreteToy1_2c-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv1_2c',
    max_episode_steps=150,
    reward_threshold=372.5,
)
register(
    id='DiscreteToy1_2d-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv1_2d',
    max_episode_steps=150,
    reward_threshold=372.5,
)
register(
    id='DiscreteToy1_2e-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv1_2e',
    max_episode_steps=150,
    reward_threshold=372.5,
)
register(
    id='DiscreteToy1_2f-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv1_2f',
    max_episode_steps=150,
    reward_threshold=1570,
)
register(
    id='DiscreteToy1_2g-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv1_2g',
    max_episode_steps=150,
    reward_threshold=1570,
)
register(
    id='DiscreteToy2_1-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv2_1',
    max_episode_steps=150,
    reward_threshold=145,
)
register(
    id='DiscreteToy2_2-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv2_2',
    max_episode_steps=150,
    reward_threshold=145,
)
register(
    id='DiscreteToy3_1a-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv3_1a',
    max_episode_steps=150,
    reward_threshold=-52.5,
)
register(
    id='DiscreteToy3_1b-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv3_1b',
    max_episode_steps=150,
    reward_threshold=-52.5,
)
register(
    id='DiscreteToy3_2-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv3_2',
    max_episode_steps=150,
    reward_threshold=72.5,
)
register(
    id='DiscreteToy4_1a-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv4_1a',
    max_episode_steps=150,
    reward_threshold=97.5,
    kwargs={'R': 1.0},
)
register(
    id='DiscreteToy4_1a-v1', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv4_1a',
    max_episode_steps=150,
    reward_threshold=145,
    kwargs={'R': 2.0},
)
register(
    id='DiscreteToy4_1b-v0', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv4_1b',
    max_episode_steps=150,
    reward_threshold=97.5,
    kwargs={'R': 1.0},
)
register(
    id='DiscreteToy4_1b-v1', 
    entry_point=f'{__name__}.discrete_toy_env:DiscreteToyEnv4_1b',
    max_episode_steps=150,
    reward_threshold=145,
    kwargs={'R': 2.0},
)

# Continuous toy environments registration
register(
    id='ContinuousToy1_1a-v0', 
    entry_point=f'{__name__}.continuous_toy_env:ContinuousToyEnv1_1a',
    max_episode_steps=100,
    reward_threshold=47.5, # 最適なリーダーの下で最適フォロワーリターンの目安
)
register(
    id='ContinuousToy1_2c-v0', 
    entry_point=f'{__name__}.continuous_toy_env:ContinuousToyEnv1_2c',
    max_episode_steps=150,
    reward_threshold=372.5,
)
register(
    id='ContinuousToy1_2d-v0', 
    entry_point=f'{__name__}.continuous_toy_env:ContinuousToyEnv1_2d',
    max_episode_steps=150,
    reward_threshold=372.5,
)
register(
    id='ContinuousToy1_2e-v0', 
    entry_point=f'{__name__}.continuous_toy_env:ContinuousToyEnv1_2e',
    max_episode_steps=150,
    reward_threshold=372.5,
)
register(
    id='ContinuousToy1_2f-v0', 
    entry_point=f'{__name__}.continuous_toy_env:ContinuousToyEnv1_2f',
    max_episode_steps=150,
    reward_threshold=1570,
)
register(
    id='ContinuousToy1_2g-v0', 
    entry_point=f'{__name__}.continuous_toy_env:ContinuousToyEnv1_2g',
    max_episode_steps=150,
    reward_threshold=1570,
)
register(
    id='ContinuousToy4_1a-v0', 
    entry_point=f'{__name__}.continuous_toy_env:ContinuousToyEnv4_1a',
    max_episode_steps=150,
    reward_threshold=97.5,
    kwargs={'R': 1.0},
)
register(
    id='ContinuousToy4_1a-v1', 
    entry_point=f'{__name__}.continuous_toy_env:ContinuousToyEnv4_1a',
    max_episode_steps=150,
    reward_threshold=145,
    kwargs={'R': 2.0},
)
register(
    id='ContinuousToy4_1b-v0', 
    entry_point=f'{__name__}.continuous_toy_env:ContinuousToyEnv4_1b',
    max_episode_steps=150,
    reward_threshold=97.5,
    kwargs={'R': 1.0},
)
register(
    id='ContinuousToy4_1b-v1', 
    entry_point=f'{__name__}.continuous_toy_env:ContinuousToyEnv4_1b',
    max_episode_steps=150,
    reward_threshold=145,
    kwargs={'R': 2.0},
)

# LQR environment registration
register(
    id='LQREnv-v0',
    entry_point=f'{__name__}.lqr_env:LQREnv_0',
    max_episode_steps=100,
    reward_threshold=None,
)
register(
    id='LQREnv-v1',
    entry_point=f'{__name__}.lqr_env:LQREnv_1',
    max_episode_steps=100,
    reward_threshold=None,
)
register(
    id='LQREnv-v2',
    entry_point=f'{__name__}.lqr_env:LQREnv_2',
    max_episode_steps=100,
    reward_threshold=None,
)
register(
    id='LQREnv-v3',
    entry_point=f'{__name__}.lqr_env:LQREnv_3',
    max_episode_steps=100,
    reward_threshold=None,
    kwargs={
        'initial_state': [1.0, 0.0], 
        'target_state': [-0.5, 0.0]
    },
)
register(
    id='LQREnv-v4',
    entry_point=f'{__name__}.lqr_env:LQREnv_4',
    max_episode_steps=100,
    reward_threshold=None,
    kwargs={
        'initial_state': [1.0, 0.0], 
        'target_state': [-0.5, 0.0]
    },
)