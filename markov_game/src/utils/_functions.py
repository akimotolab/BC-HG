"""Functions exposed directly in the garage namespace."""
from collections import defaultdict
import time

import torch
import click
import numpy as np

from garage import EpisodeBatch, StepType
from garage.np import discount_cumsum, stack_tensor_dict_list


class _Default:  # pylint: disable=too-few-public-methods
    """A wrapper class to represent default arguments.

    Args:
        val (object): Argument value.

    """

    def __init__(self, val):
        self.val = val


def make_optimizer(optimizer_type, module=None, **kwargs):
    """Create an optimizer for pyTorch & tensorflow algos.

    Args:
        optimizer_type (Union[type, tuple[type, dict]]): Type of optimizer.
            This can be an optimizer type such as 'torch.optim.Adam' or a
            tuple of type and dictionary, where dictionary contains arguments
            to initialize the optimizer e.g. (torch.optim.Adam, {'lr' : 1e-3})
        module (optional): If the optimizer type is a `torch.optimizer`.
            The `torch.nn.Module` module whose parameters needs to be optimized
            must be specify.
        kwargs (dict): Other keyword arguments to initialize optimizer. This
            is not used when `optimizer_type` is tuple.

    Returns:
        torch.optim.Optimizer: Constructed optimizer.

    Raises:
        ValueError: Raises value error when `optimizer_type` is tuple, and
            non-default argument is passed in `kwargs`.

    """
    if isinstance(optimizer_type, tuple):
        opt_type, opt_args = optimizer_type
        for name, arg in kwargs.items():
            if not isinstance(arg, _Default):
                raise ValueError('Should not specify {} and explicit \
                    optimizer args at the same time'.format(name))
        if module is not None:
            return opt_type(module.parameters(), **opt_args)
        else:
            return opt_type(**opt_args)

    opt_args = {
        k: v.val if isinstance(v, _Default) else v
        for k, v in kwargs.items()
    }
    if module is not None:
        return optimizer_type(module.parameters(), **opt_args)
    else:
        return optimizer_type(**opt_args)


def rollout(env,
            agent,
            *,
            max_episode_length=np.inf,
            animated=False,
            render_mode='human',
            pause_per_frame=None,
            deterministic_l=False,
            deterministic_f=False,
            zero_action_l=False,
            zero_action_f=False):
    """Sample a single episode of the agent in the environment.

    Args:
        agent (Policy): Policy used to select actions.
        env (Environment): Environment to perform actions in.
        max_episode_length (int): If the episode reaches this many timesteps,
            it is truncated.
        animated (bool): If true, render the environment after each step.
        pause_per_frame (float): Time to sleep between steps. Only relevant if
            animated == true.
        deterministic_l (bool): Whether to use deterministic leader policy.
        deterministic_f (bool): Whether to use deterministic follower policy.
        zero_action_l (bool): Whether to use zero action for leader policy.
        zero_action_f (bool): Whether to use zero action for follower policy.

    Returns:
        dict[str, np.ndarray or dict]: Dictionary, with keys:
            * observations(np.array): Flattened array of observations.
                There should be one more of these than actions. Note that
                observations[i] (for i < len(observations) - 1) was used by the
                agent to choose actions[i]. Should have shape
                :math:`(T + 1, S^*)`, i.e. the unflattened observation space of
                    the current environment.
            * actions(np.array): Non-flattened array of actions. Should have
                shape :math:`(T, S^*)`, i.e. the unflattened action space of
                the current environment.
            * rewards(np.array): Array of rewards of shape :math:`(T,)`, i.e. a
                1D array of length timesteps.
            * agent_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `agent_info` arrays.
            * env_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `env_info` arrays.
            * dones(np.array): Array of termination signals.

    """
    env_steps = []
    agent_infos = []
    observations = []
    last_obs, episode_infos = env.reset()
    agent.reset()
    episode_length = 0
    if animated:
        env.visualize(mode=render_mode)
    while episode_length < (max_episode_length or np.inf):
        if pause_per_frame is not None:
            time.sleep(pause_per_frame)
        a, agent_info = agent.get_action(last_obs, 
                                         deterministic_leader=deterministic_l, 
                                         deterministic_follower=deterministic_f, 
                                         zero_leader=zero_action_l,
                                         zero_follower=zero_action_f)
        es = env.step(a)
        env_steps.append(es)
        observations.append(last_obs)
        agent_infos.append(agent_info)
        episode_length += 1
        if es.last:
            break
        last_obs = es.observation

    return dict(
        episode_infos=episode_infos,
        observations=np.array(observations),
        actions=np.array([es.action for es in env_steps]),
        rewards=np.array([es.reward for es in env_steps]),
        agent_infos=stack_tensor_dict_list(agent_infos),
        env_infos=stack_tensor_dict_list([es.env_info for es in env_steps]),
        dones=np.array([es.terminal for es in env_steps]),
        frames=np.array(env.frames)
    )


def obtain_evaluation_episodes(policy,
                               env,
                               max_episode_length=1000,
                               num_eps=100,
                               deterministic_l=False,
                               deterministic_f=False,
                               zero_action_l=False,
                               zero_action_f=False):
    """Sample the policy for num_eps episodes and return average values.

    Args:
        policy (Policy): Policy to use as the actor when gathering samples.
        env (Environment): The environement used to obtain episodes.
        max_episode_length (int): Maximum episode length. The episode will
            truncated when length of episode reaches max_episode_length.
        num_eps (int): Number of episodes.
        deterministic_l (bool): Whether to use deterministic leader policy.
        deterministic_f (bool): Whether to use deterministic follower policy.
        zero_action_l (bool): Whether to use zero action for leader policy.
        zero_action_f (bool): Whether to use zero action for follower policy.

    Returns:
        EpisodeBatch: Evaluation episodes, representing the best current
            performance of the algorithm.

    """
    episodes = []
    # Use a finite length rollout for evaluation.

    with click.progressbar(range(num_eps), label='Evaluating') as pbar:
        for _ in pbar:
            eps = rollout(env,
                          policy,
                          max_episode_length=max_episode_length,
                          deterministic_l=deterministic_l,
                          deterministic_f=deterministic_f,
                          zero_action_l=zero_action_l,
                          zero_action_f=zero_action_f,
                          )
            episodes.append(eps)
    return EpisodeBatch.from_list(env.spec, episodes)


def log_multitask_performance(tabular, itr, batch, discount, name_map=None):
    r"""Log performance of episodes from multiple tasks.

    Args:
        itr (int): Iteration number to be logged.
        batch (EpisodeBatch): Batch of episodes. The episodes should have
            either the "task_name" or "task_id" `env_infos`. If the "task_name"
            is not present, then `name_map` is required, and should map from
            task id's to task names.
        discount (float): Discount used in computing returns.
        name_map (dict[int, str] or None): Mapping from task id's to task
            names. Optional if the "task_name" environment info is present.
            Note that if provided, all tasks listed in this map will be logged,
            even if there are no episodes present for them.

    Returns:
        numpy.ndarray: Undiscounted returns averaged across all tasks. Has
            shape :math:`(N \bullet [T])`.

    """
    eps_by_name = defaultdict(list)
    for eps in batch.split():
        task_name = '__unnamed_task__'
        if 'task_name' in eps.env_infos:
            task_name = eps.env_infos['task_name'][0]
        elif 'task_id' in eps.env_infos:
            name_map = {} if name_map is None else name_map
            task_id = eps.env_infos['task_id'][0]
            task_name = name_map.get(task_id, 'Task #{}'.format(task_id))
        eps_by_name[task_name].append(eps)
    if name_map is None:
        task_names = eps_by_name.keys()
    else:
        task_names = name_map.values()
    for task_name in task_names:
        if task_name in eps_by_name:
            episodes = eps_by_name[task_name]
            log_performance(itr,
                            EpisodeBatch.concatenate(*episodes),
                            discount,
                            prefix=task_name)
        else:
            with tabular.prefix(task_name + '/'):
                tabular.record('Iteration', itr)
                tabular.record('NumEpisodes', 0)
                tabular.record('AverageDiscountedReturn', np.nan)
                tabular.record('AverageReturn', np.nan)
                tabular.record('StdReturn', np.nan)
                tabular.record('MaxReturn', np.nan)
                tabular.record('MinReturn', np.nan)
                tabular.record('TerminationRate', np.nan)
                tabular.record('SuccessRate', np.nan)

    return log_performance(itr, batch, discount=discount, prefix='Average')


def log_performance(tabular, itr, batch, discount, prefix='Evaluation'):
    """Evaluate the performance of an algorithm on a batch of episodes.

    Args:
        itr (int): Iteration number.
        batch (EpisodeBatch): The episodes to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    if isinstance(discount, (list, tuple)):
        discount_leader = discount[0]
        discount_follower = discount[1]
    else:
        discount_leader = discount
        discount_follower = discount
    returns = []
    target_returns = []
    undiscounted_returns = []
    undiscounted_t_returns = []
    termination = []
    success = []
    for eps in batch.split():
        returns.append(discount_cumsum(eps.rewards, discount_follower))
        target_returns.append(discount_cumsum(eps.env_infos['target_reward'], discount_leader))
        undiscounted_returns.append(sum(eps.rewards))
        undiscounted_t_returns.append(sum(eps.env_infos['target_reward']))
        termination.append(
            float(
                any(step_type == StepType.TERMINAL
                    for step_type in eps.step_types)))
        if 'success' in eps.env_infos:
            success.append(float(eps.env_infos['success'].any()))

    average_discounted_return = np.mean([rtn[0] for rtn in returns])
    average_discounted_t_return = np.mean([rtn[0] for rtn in target_returns])

    performance = dict(
        AverageReturn=np.mean(undiscounted_returns),
        StdReturn=np.std(undiscounted_returns),
        AverageTargetReturn=np.mean(undiscounted_t_returns),
        StdTargetReturn=np.std(undiscounted_t_returns),
    )

    with tabular.prefix(prefix + '/'):
        tabular.record('Iteration', itr)
        tabular.record('NumEpisodes', len(returns))
        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', performance['AverageReturn'])
        tabular.record('StdReturn', performance['StdReturn'])
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))
        tabular.record('AverageDiscountedTargetReturn', average_discounted_t_return)
        tabular.record('AverageTargetReturn', performance['AverageTargetReturn'])
        tabular.record('StdTargetReturn', performance['StdTargetReturn'])
        tabular.record('StdTargetReturn', np.std(undiscounted_t_returns))
        tabular.record('MaxTargetReturn', np.max(undiscounted_t_returns))
        tabular.record('MinTargetReturn', np.min(undiscounted_t_returns))
        tabular.record('TerminationRate', np.mean(termination))
        if success:
            tabular.record('SuccessRate', np.mean(success))

    return performance

