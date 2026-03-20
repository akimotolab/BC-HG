"""This module implements a replay buffer memory.

Replay buffer is an important technique in reinforcement learning. It
stores transitions in a memory buffer of fixed size. When the buffer is
full, oldest memory will be discarded. At each step, a batch of memories
will be sampled from the buffer to update the agent's parameters. In a
word, replay buffer breaks temporal correlations and thus benefits RL
algorithms.

"""
import abc

class ReplayBufferBase(abc.ABC):
    """Abstract class for Replay Buffer.

    Args:
        env_spec (EnvSpec): Environment specification.
        size_in_transitions (int): total size of transitions in the buffer
        time_horizon (int): time horizon of epsiode.

    """
    def __init__(self, size):
        self._size = size
        
    @abc.abstractmethod
    def sample_transitions(self, batch_size):
        """Sample a transition of batch_size.

        Args:
            batch_size(int): The number of transitions to be sampled.
        """
        
    @abc.abstractmethod
    def add_transition(self, **kwargs):
        """Add one transition into the replay buffer.

        Args:
            kwargs (dict(str, [numpy.ndarray])): Dictionary that holds
                the transitions.

        """

    @abc.abstractmethod
    def add_transitions(self, **kwargs):
        """Add multiple transitions into the replay buffer.

        A transition contains one or multiple entries, e.g.
        observation, action, reward, terminal and next_observation.
        The same entry of all the transitions are stacked, e.g.
        {'observation': [obs1, obs2, obs3]} where obs1 is one
        numpy.ndarray observation from the environment.

        Args:
            kwargs (dict(str, [numpy.ndarray])): Dictionary that holds
                the transitions.

        """

    @abc.abstractmethod
    def _initialize_buffer(self, **kwargs):
        """Initialize the replay buffer.

        Args:
            kwargs (dict(str, [numpy.ndarray])): Dictionary that holds
                the transitions.
            
        """

    @abc.abstractmethod
    def _get_storage_idx(self, size_increment=1):
        """Get the storage index for the episode to add into the buffer.

        Args:
            size_increment(int): The number of storage indeces that new
                transitions will be placed in.

        Returns:
            numpy.ndarray: The indeces to store size_incremente transitions at.

        """
    
    @abc.abstractmethod
    def clear(self):
        """Clear buffer."""

