import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import akro
from garage.torch.policies import Policy

class ContinuousToyEnvBase(gym.Env):
    STATE_TYPE = [
        'S',  # 0
        'A',  # 1
        'B',  # 2
    ]

    LEADER_ACT_TYPE = [
        '0',  # 0
        '1',  # 1
    ]
    LEADER_ACT_RANGE = (0.0, 1.0)  # The probability of taking leader action "1"

    FOLLOWER_ACT_TYPE = [
        's',  # 0
        'a',  # 1
        'b',  # 2
    ]

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

        s_num = len(self.STATE_TYPE)  # state number
        la_num = len(self.LEADER_ACT_TYPE)  # leader action number
        fa_num = len(self.FOLLOWER_ACT_TYPE)  # follower action number

        # Transition function transition[state][leader_action][follower_action]
        #          = 0 : transition to S
        #          = 1 : transition to A
        #          = 2 : transition to B
        #          = -1 : stay in the current state
        self.transition = np.ones((s_num,la_num,fa_num), dtype=np.int32) * -1
        
        # Reward function rewards[state][leader_action][follower_action]
        self.rewards = np.zeros((s_num,la_num,fa_num), dtype=np.float32)
        self.r_range = (-np.inf, np.inf)

        # Target reward function target_rewards[state][leader_action][follower_action]
        self.target_rewards = np.zeros((s_num,la_num,fa_num), dtype=np.float32)
        self.target_r_range = (-np.inf, np.inf)

        # Leader action is a continuous value representing the probability of choosing leader action "1" at each state.
        # State transitions and immediate rewards occur according to leader action "0"/"1".
        # The optimal leader policy selects self.optimal_action with probability 1.0 at state self.key_state.
        self.key_state = None
        self.optimal_action = None # 0 or 1 (changed by task)

        # Episode length
        self._max_episode_steps = 100
        
        self.observation_space = akro.from_gym(spaces.Discrete(3))
        self.action_space = akro.from_gym(spaces.Discrete(3))
        self.leader_action_space = akro.from_gym(spaces.Box(low=self.LEADER_ACT_RANGE[0],
                                                            high=self.LEADER_ACT_RANGE[1],
                                                            shape=(1,), dtype=np.float32))

        self.seed()
        self.state = None
        self.actions = [None, None, None]
        self.render_rewards = [0, 0]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.leader_action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]
    
    def transition_fn(self, state, leader_action, follower_action):
        next_state = self.transition[state,leader_action,follower_action]
        if next_state == -1:
            return state
        return next_state
    
    def reward_fn(self, state, leader_action, follower_action):
        return self.rewards[state,leader_action,follower_action]
    
    def target_reward_fn(self, state, leader_action, follower_action):
        return self.target_rewards[state,leader_action,follower_action]

    def step(self, action):

        leader_action = action[0]
        follower_action = action[1]

        assert self.leader_action_space.contains(leader_action), \
            f"Invalid leader action {leader_action} passed to step"

        sampled_la = self.np_random.choice([0,1], p=[1.0-leader_action[0], leader_action[0]])

        next_state = self.transition_fn(self.state, sampled_la, follower_action)
        reward = self.reward_fn(self.state, sampled_la, follower_action)
        target_reward = self.target_reward_fn(self.state, sampled_la, follower_action)
        
        self.state = next_state
        self.steps_n += 1
        done = (self.steps_n >= self._max_episode_steps)

        self.actions = [leader_action, follower_action, sampled_la]
        self.render_rewards = [target_reward, reward]
        info = {'leader_action': leader_action, 
                'sampled_leader_action': sampled_la, 
                'target_reward': target_reward}
        return self.state, reward, done, info
    
    def reset(self, initial_state=None):
        if initial_state is not None:
            assert self.observation_space.contains(initial_state), \
                f"Invalid state {initial_state} passed to reset"
            self.state = initial_state
        else:
            self.state = 0
        self.actions = [None, None, None]
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
            print(f'leader action={self.actions[0]}, ' \
                  f'sampled leader action={self.LEADER_ACT_TYPE[self.actions[2]]}, ' \
                  f'follower action={self.FOLLOWER_ACT_TYPE[self.actions[1]]}, ' \
                  f'reward={self.render_rewards[1]}, target_reward={self.render_rewards[0]}')
            print(f'Step {self.steps_n}: state={self.STATE_TYPE[self.state]}')
        return self.STATE_TYPE[self.state]

    def close(self):
        pass
    
    def leader_policy_opt_gap(self, policy):
        """
        Calculate the gap between the optimal leader policy and the given policy.
        """
        if isinstance(policy, Policy):  # if deterministic policy
            a, _ = policy.get_action(self.observation_space.flatten(self.key_state))
            if self.optimal_action == 1:
                gap = 1.0 - a
            elif self.optimal_action == 0:
                gap = a
            else:
                raise ValueError("The optimal action must be 0 or 1.")
            return gap
        else:
            raise NotImplementedError("The policy type is not supported.")
        

class ContinuousToyEnv1_1a(ContinuousToyEnvBase):
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

        self.key_state = 0
        self.optimal_action = 1  # leader action "1" is optimal in state "0"

class ContinuousToyEnv1_2c(ContinuousToyEnvBase):
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

        self.key_state = 0
        self.optimal_action = 1  # leader action "1" is optimal in state "0"

class ContinuousToyEnv1_2d(ContinuousToyEnv1_2c):
    """
    leader action=0 -> best response return (follower, leader)=(250, -50)
    leader action=1 -> best response return (follower, leader)=(375, 75) 
    """
    def __init__(self):
        super().__init__()
        self.rewards[1,:,2] = 2
        self.rewards[2,:,0] = 3
        self.r_range = (0, 3)

class ContinuousToyEnv1_2e(ContinuousToyEnv1_2c):
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

class ContinuousToyEnv1_2f(ContinuousToyEnv1_2e):
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

class ContinuousToyEnv1_2g(ContinuousToyEnv1_2f):
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

class ContinuousToyEnv4_1a(ContinuousToyEnvBase):
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
        self.optimal_action = 0 # leader action "0" is optimal in state "1"

class ContinuousToyEnv4_1b(ContinuousToyEnv4_1a):
    """
    optimal action=0
    leader action=0 -> best response return (follower, leader)=(100, 50)
    leader action=1 -> best response return (follower, leader)=(75, 0)
    """
    def __init__(self, R=1.0):
        super().__init__(R)
        self.target_rewards[0,:,2] = 0
        self.target_r_range = (0, 1)
