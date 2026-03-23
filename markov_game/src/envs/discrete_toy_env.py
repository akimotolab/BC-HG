import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import akro

from ..policies import CategoricalMLPPolicy

class DiscreteToyEnvBase(gym.Env):
    STATE_TYPE = [
        'S',  # 0
        'A',  # 1
        'B',  # 2
    ]

    LEADER_ACT_TYPE =[
        '0',  # 0
        '1',  # 1
    ]

    FOLLOWER_ACT_TYPE = [
        's',  # 0
        'a',  # 1
        'b',  # 2
    ]

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

        # Transition function transition[state][leader_action][follower_action]
        #          = 0 : transition to S
        #          = 1 : transition to A
        #          = 2 : transition to B
        #          = -1 : stay in the current state
        self.transition = np.ones((3,2,3), dtype=np.int32) * -1
        
        # Reward function rewards[state][leader_action][follower_action]
        self.rewards = np.zeros((3,2,3))
        self.r_range = (-np.inf, np.inf)

        # Target reward function target_rewards[state][leader_action][follower_action]
        self.target_rewards = np.zeros((3,2,3))
        self.target_r_range = (-np.inf, np.inf)

        # The optimal leader policy chooses action "1" at self.key_state (with exceptions)
        self.key_state = None
        self.optimal_action = 1

        # Cost of leader_action costs[state][leader_action]
        self.costs = np.zeros((3,2))

        # Episode length
        self._max_episode_steps = 100

        # Q-function of the deterministic optimal follower policy opt_ag_qtable[leader_action][state][follower_action]
        # ( leader_state = LeaderPolicy[key_state] )
        # The Q-value of the optimal action is 1 and that of the non-optimal is 0.
        self.opt_ag_qtable = np.zeros((2,3,3))
        
        self.observation_space = akro.from_gym(spaces.Discrete(3))
        self.action_space = akro.from_gym(spaces.Discrete(3))
        self.leader_action_space = akro.from_gym(spaces.Discrete(2))

        self.seed()
        self.state = None
        self.actions = [None, None]
        self.render_rewards = [0, 0]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.leader_action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]
    
    def transition_fn(self, state, leader_action, follower_action):
        next_state = self.transition[state, leader_action, follower_action].item()
        if next_state == -1:
            return state
        return next_state
    
    def reward_fn(self, state, leader_action, follower_action):
        return self.rewards[state, leader_action, follower_action].item()
    
    def target_reward_fn(self, state, leader_action, follower_action):
        return self.target_rewards[state, leader_action, follower_action].item()

    def step(self, action):

        leader_action = action[0]
        follower_action = action[1]

        next_state = self.transition_fn(self.state, leader_action, follower_action)
        reward = self.reward_fn(self.state, leader_action, follower_action)
        target_reward = self.target_reward_fn(self.state, leader_action, follower_action)
        self.steps_n += 1
        self.state = next_state
        done = (self.steps_n >= self._max_episode_steps)

        self.actions = [leader_action, follower_action]
        self.render_rewards = [target_reward, reward]
        info = {'leader_action': leader_action, 'target_reward': target_reward}
        return self.state, reward, done, info
    
    def reset(self, initial_state=None):
        if initial_state is not None:
            assert self.observation_space.contains(initial_state), \
                f"Invalid state {initial_state} passed to reset"
            self.state = int(initial_state)
        else:
            self.state = 0
        self.actions = [None, None]
        self.render_rewards = [0, 0]
        self.steps_n = 0
        return self.state
    
    def set_state(self, state):
        self.state = state

    def render(self, mode='human'):
        if self.actions[0] is None or self.actions[1] is None:
            if self.steps_n == 0:
                print(f'Step {self.steps_n}: state={self.STATE_TYPE[self.state]}')
            else:
                print(f'Step {self.steps_n}: state={self.STATE_TYPE[self.state]}')
        else:
            print(f'leader action={self.LEADER_ACT_TYPE[self.actions[0]]}, ' \
                  f'follower action={self.FOLLOWER_ACT_TYPE[self.actions[1]]}, ' \
                  f'reward={self.render_rewards[1]}, target_reward={self.render_rewards[0]}')
            print(f'Step {self.steps_n}: state={self.STATE_TYPE[self.state]}')
        return self.STATE_TYPE[self.state]

    def close(self):
        pass

    def get_opt_ag_act_array(self):
        return self.opt_ag_qtable.argmax(axis=2)
    
    def leader_policy_opt_gap(self, policy):
        """
        Calculate the gap between the optimal leader policy and the given policy.
        """
        if isinstance(policy, CategoricalMLPPolicy):
            _, info = policy.get_action(self.observation_space.flatten(self.key_state))
            gap = 1.0 - info['probs'][self.optimal_action]
            return gap
        else:
            raise NotImplementedError("The policy type is not supported.")


class DiscreteToyEnv1_1a(DiscreteToyEnvBase):
    """
    leader action=0 -> best response return (follower, leader)=(50, -50)
    leader action=1 -> best response return (follower, leader)=(50, 50) 
    """
    def __init__(self):
        super().__init__()
        self.transition[0,0,1] = 1
        self.transition[0,1,1] = 2
        self.transition[0,0,2] = 2
        self.transition[0,1,2] = 1
        self.transition[1,0,0] = 0
        self.transition[1,1,0] = 0
        self.transition[2,0,0] = 0
        self.transition[2,1,0] = 0

        self.rewards[1,:,0] = 1
        self.rewards[2,:,0] = -1
        self.r_range = (-1, 1)

        self.target_rewards[0,:,1] = -1
        self.target_rewards[0,:,2] = 1
        self.target_r_range = (-1, 1)

        # Leader's action is that at state S: key_state=0
        self.key_state = 0
        self.opt_ag_qtable[0,0,1] = 1
        self.opt_ag_qtable[0,1,0] = 1
        self.opt_ag_qtable[0,2,0] = 1
        self.opt_ag_qtable[1,0,2] = 1
        self.opt_ag_qtable[1,1,0] = 1
        self.opt_ag_qtable[1,2,0] = 1

class DiscreteToyEnv1_1b(DiscreteToyEnv1_1a):
    """
    leader action=0 -> best response return (follower, leader)=(50, 0)
    leader action=1 -> best response return (follower, leader)=(50, 50) 
    """
    def __init__(self):
        super().__init__()
        self.target_rewards[0,:,1] = 0
        self.target_r_range = (0, 1)

class DiscreteToyEnv1_2a(DiscreteToyEnvBase):
    """
    leader action=0 -> best response return (follower, leader)=(250, -50)
    leader action=1 -> best response return (follower, leader)=(375, 75) 
    """
    def __init__(self):
        super().__init__()
        self.transition[0,0,1] = 1
        self.transition[0,1,1] = 2
        self.transition[0,0,2] = 2
        self.transition[0,1,2] = 1
        self.transition[1,0,0] = 0
        self.transition[1,1,0] = 0
        self.transition[2,0,0] = 0
        self.transition[2,1,0] = 0
        self.transition[1,0,2] = 2
        self.transition[1,1,2] = 2

        self.rewards[0,:,2] = 2
        self.rewards[1,:,0] = 3
        self.rewards[1,:,2] = 8
        self.rewards[2,:,0] = -3
        self.r_range = (-3, 8)

        self.target_rewards[1,:,0] = 1
        self.target_rewards[1,:,2] = -1
        self.target_r_range = (-1, 1)

        self._max_episode_steps = 150

        # Leader's action is that at state S: key_state=0
        self.key_state = 0
        self.opt_ag_qtable[0,0,1] = 1
        self.opt_ag_qtable[0,1,2] = 1
        self.opt_ag_qtable[0,2,0] = 1
        self.opt_ag_qtable[1,0,2] = 1
        self.opt_ag_qtable[1,1,0] = 1
        self.opt_ag_qtable[1,2,0] = 1

class DiscreteToyEnv1_2b(DiscreteToyEnv1_2a):
    """
    leader action=0 -> best response return (follower, leader)=(250, 0)
    leader action=1 -> best response return (follower, leader)=(375, 75) 
    """
    def __init__(self):
        super().__init__()
        self.target_rewards[1,:,2] = 0
        self.target_r_range = (0, 1)

class DiscreteToyEnv1_2c(DiscreteToyEnvBase):
    """
    leader action=0 -> best response return (follower, leader)=(250, -50)
    leader action=1 -> best response return (follower, leader)=(375, 75) 
    """
    def __init__(self):
        super().__init__()
        self.transition[0,:,1] = 1
        self.transition[1,:,0] = 0
        self.transition[1,:,2] = 2
        self.transition[2,:,0] = 0

        self.rewards[0,1,1] = 2
        self.rewards[1,:,0] = 3
        self.rewards[2,:,0] = 5
        self.r_range = (0, 5)

        self.target_rewards[1,:,0] = 1
        self.target_rewards[1,:,2] = -1
        self.target_r_range = (-1, 1)

        self._max_episode_steps = 150

        # Leader's action is that at state S: key_state=0
        self.key_state = 0
        self.opt_ag_qtable[0,0,1] = 1
        self.opt_ag_qtable[0,1,2] = 1
        self.opt_ag_qtable[0,2,0] = 1
        self.opt_ag_qtable[1,0,1] = 1
        self.opt_ag_qtable[1,1,0] = 1
        self.opt_ag_qtable[1,2,0] = 1

class DiscreteToyEnv1_2d(DiscreteToyEnv1_2c):
    """
    leader action=0 -> best response return (follower, leader)=(250, -50)
    leader action=1 -> best response return (follower, leader)=(375, 75) 
    """
    def __init__(self):
        super().__init__()
        self.rewards[1,:,2] = 2
        self.rewards[2,:,0] = 3
        self.r_range = (0, 3)

class DiscreteToyEnv1_2e(DiscreteToyEnv1_2c):
    """
    leader action=0 -> best response return (follower, leader)=(250, -50)
    leader action=1 -> best response return (follower, leader)=(375, 75) 
    """
    def __init__(self):
        super().__init__()
        self.transition[0,:,2] = 2

        self.rewards[1,:,2] = 8
        self.rewards[2,:,0] = -3
        self.r_range = (-3, 8)

class DiscreteToyEnv1_2f(DiscreteToyEnv1_2e):
    """
    leader policy f(0|S)=1.0:
        - best response return (follower, leader)=(1451, 195) (infinite horizon)
        - horizon=150, discount=1.0: (follower, leader)=(250, -50)
        - S->A->B->S->A->B->...
    leader policy f(1|S)=1.0:
        - best response return (follower, leader)=(1575, 291) (infinite horizon)
        - horizon=150, discount=1.0: (follower, leader)=(404, 75)
        - S->A->S->A->...
    leader policy f(0|S)>0.5 -> S->A->B->S->A->B->...
    leader policy f(1|S)>0.5 -> S->A->S->A->...
    """
    def __init__(self):
        super().__init__()
        g = 0.99  # discount rate must be 0.99
        R = (8*(1+g**4)-3*(g**2+g**5)-3*(g+g**3+g**5))/(0.5*(1+g**2+g**4)-0.5*(1+g**3))
        # R = 2.3858910707072356
        self.rewards[0,1,1] = R
        self._max_episode_steps = np.inf  # Infinite horizon

class DiscreteToyEnv1_2g(DiscreteToyEnv1_2f):
    """
    leader policy f(0|S)=1.0:
        - best response return (follower, leader)=(1451, 195) (infinite horizon)
        - horizon=150, discount=1.0: (follower, leader)=(250, 50)
        - S->A->B->S->A->B->...
    leader policy f(1|S)=1.0:
        - best response return (follower, leader)=(1575, 582) (infinite horizon)
        - horizon=150, discount=1.0: (follower, leader)=(404, 150)
        - S->A->S->A->...
    leader policy f(0|S)>0.5 -> S->A->B->S->A->B->...
    leader policy f(1|S)>0.5 -> S->A->S->A->...
    """
    def __init__(self):
        super().__init__()
        self.target_rewards[1,:,0] = 2
        self.target_rewards[1,:,2] = 1
        self.target_r_range = (0, 2)

class DiscreteToyEnv2_1(DiscreteToyEnvBase):
    """
    leader action=0 -> best response return (follower, leader)=(150, -50)
    leader action=1 -> best response return (follower, leader)=(150, 50) 
    """
    def __init__(self):
        super().__init__()
        self.transition[0,0,1] = 1
        self.transition[0,1,1] = 2
        self.transition[0,0,2] = 2
        self.transition[0,1,2] = 1
        self.transition[1,0,0] = 0
        self.transition[1,1,0] = 0
        self.transition[2,0,0] = 0
        self.transition[2,1,0] = 0
        self.transition[1,0,2] = 2
        self.transition[1,1,2] = 2

        self.rewards[1,:,0] = 1
        self.rewards[2,:,0] = -1
        self.rewards[1,:,2] = 4
        self.r_range = (-1, 4)

        self.target_rewards[0,:,1] = -1
        self.target_rewards[0,:,2] = 1
        self.target_r_range = (-1, 1)

        self._max_episode_steps = 150

        # Leader's action is that at state S: key_state=0
        self.key_state = 0
        self.opt_ag_qtable[0,0,1] = 1
        self.opt_ag_qtable[0,1,2] = 1
        self.opt_ag_qtable[0,2,0] = 1
        self.opt_ag_qtable[1,0,2] = 1
        self.opt_ag_qtable[1,1,2] = 1
        self.opt_ag_qtable[1,2,0] = 1

class DiscreteToyEnv2_2(DiscreteToyEnvBase):
    """
    leader action=0 -> best response return (follower, leader)=(150, 0)
    leader action=1 -> best response return (follower, leader)=(150, 50) 
    """
    def __init__(self):
        super().__init__()
        self.transition[0,0,1] = 1
        self.transition[0,1,1] = 1
        self.transition[1,0,0] = 0
        self.transition[1,1,0] = 2
        self.transition[1,0,2] = 2
        self.transition[1,1,2] = 0
        self.transition[2,0,0] = 0
        self.transition[2,1,0] = 0
        
        self.rewards[0,:,1] = -1
        self.rewards[2,:,0] = 4
        self.r_range = (-1, 4)

        self.target_rewards[1,:,0] = 1
        self.target_r_range = (0, 1)

        self._max_episode_steps = 150

        # Leader's action is that at state A: key_state=1
        self.key_state = 1
        self.opt_ag_qtable[0,0,1] = 1
        self.opt_ag_qtable[0,1,2] = 1
        self.opt_ag_qtable[0,2,0] = 1
        self.opt_ag_qtable[1,0,1] = 1
        self.opt_ag_qtable[1,1,0] = 1
        self.opt_ag_qtable[1,2,0] = 1

class DiscreteToyEnv3_1a(DiscreteToyEnvBase):
    """
    leader action=0 -> best response return (follower, leader)=(75, -75)
    leader action=1 -> best response return (follower, leader)=(-50, 50) 
    """
    def __init__(self):
        super().__init__()
        self.transition[0,0,1] = 1
        self.transition[0,1,1] = 2
        self.transition[0,0,2] = 2
        self.transition[0,1,2] = 1
        self.transition[1,0,0] = 0
        self.transition[1,1,0] = 0
        self.transition[2,0,0] = 0
        self.transition[2,1,0] = 0
        self.transition[1,0,2] = 2
        self.transition[1,1,2] = 2
        
        self.rewards = np.ones(self.rewards.shape) * -1
        self.rewards[0,:,2] = -3
        self.rewards[1,:,0] = 2
        self.rewards[1,:,2] = 2
        self.rewards[2,:,0] = 0
        self.r_range = (-3, 2)

        self.target_rewards[1,:,2] = 1
        self.target_rewards[1,:,0] = -1
        self.target_r_range = (-1, 1)

        self._max_episode_steps = 150

        # Leader's action is that at state S: key_state=0
        self.key_state = 0
        self.opt_ag_qtable[0,0,1] = 1
        self.opt_ag_qtable[0,1,0] = 1
        self.opt_ag_qtable[0,2,0] = 1
        self.opt_ag_qtable[1,0,2] = 1
        self.opt_ag_qtable[1,1,2] = 1
        self.opt_ag_qtable[1,2,0] = 1

class DiscreteToyEnv3_1b(DiscreteToyEnv3_1a):
    """
    leader action=0 -> best response return (follower, leader)=(75, 0)
    leader action=1 -> best response return (follower, leader)=(-50, 50)
    """
    def __init__(self):
        super().__init__()
        self.target_rewards[1,:,0] = 0
        self.target_r_range = (0, 1)

class DiscreteToyEnv3_2(DiscreteToyEnvBase):
    """
    leader action=0 -> best response return (follower, leader)=((big_reward+1)*50, 0)
    leader action=1 -> best response return (follower, leader)=(75, 75)
    """
    def __init__(self, big_reward=29):
        super().__init__()
        self.transition[0,0,1] = 1
        self.transition[0,1,1] = 1
        self.transition[1,0,0] = 0
        self.transition[1,1,0] = 0
        self.transition[1,0,2] = 2
        self.transition[2,0,0] = 0
        self.transition[2,1,0] = 0

        self.rewards[0,:,1] = 1
        self.rewards[2,:,0] = big_reward
        self.r_range = (0, big_reward)

        self.target_rewards[1,:,0] = 1
        self.target_r_range = (0, 1)

        self._max_episode_steps = 150

        # Leader's action is that at state A: key_state=1
        self.key_state = 1
        self.opt_ag_qtable[0,0,1] = 1
        self.opt_ag_qtable[0,1,2] = 1
        self.opt_ag_qtable[0,2,0] = 1
        self.opt_ag_qtable[1,0,1] = 1
        self.opt_ag_qtable[1,1,0] = 1
        self.opt_ag_qtable[1,2,0] = 1

class DiscreteToyEnv4_1a(DiscreteToyEnvBase):
    """
    optimal action=0
    leader action=0 -> best response return (follower, leader)=(100, 50)
    leader action=1 -> best response return (follower, leader)=(75, -75)
    """
    def __init__(self, R=1.0):
        super().__init__()
        self.transition[0,:,1] = 1
        self.transition[0,:,2] = 2
        self.transition[1,0,0] = 0
        self.transition[1,0,2] = 2
        self.transition[1,1,:] = 0
        self.transition[2,:,0] = 0

        self.rewards[0,:,1] = 1
        self.rewards[0,:,2] = 1
        self.rewards[1,1,:] = -1
        self.rewards[1,0,2] = R
        self.r_range = (-1, max(1, R))

        self.target_rewards[0,:,1] = 1
        self.target_rewards[0,:,2] = -1
        self.target_r_range = (-1, 1)

        self._max_episode_steps = 150

        # Leader's action is that at state A: key_state=1
        self.key_state = 1
        self.optimal_action = 1
        self.opt_ag_qtable[0,0,1] = 1
        self.opt_ag_qtable[0,1,2] = 1
        self.opt_ag_qtable[0,2,0] = 1
        self.opt_ag_qtable[1,0,2] = 1
        self.opt_ag_qtable[1,1,0] = 1
        self.opt_ag_qtable[1,2,0] = 1

class DiscreteToyEnv4_1b(DiscreteToyEnv4_1a):
    """
    optimal action=0
    leader action=0 -> best response return (follower, leader)=(100, 50)
    leader action=1 -> best response return (follower, leader)=(75, 0)
    """
    def __init__(self, R=1.0):
        super().__init__(R)
        self.target_rewards[0,:,2] = 0
        self.target_r_range = (0, 1)