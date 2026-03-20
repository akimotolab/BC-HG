"""This module implements a replay buffer memory.

Replay buffer is an important technique in reinforcement learning. It
stores transitions in a memory buffer of fixed size. When the buffer is
full, oldest memory will be discarded. At each step, a batch of memories
will be sampled from the buffer to update the agent's parameters. In a
word, replay buffer breaks temporal correlations and thus benefits RL
algorithms.

"""
import numpy as np

from ._base import ReplayBufferBase

class GammaReplayBuffer(ReplayBufferBase):
    """Abstract class for Replay Buffer.

    Args:
        env_spec (EnvSpec): Environment specification.
        size_in_transitions (int): total size of transitions in the buffer
        time_horizon (int): time horizon of epsiode.

    """

    def __init__(self, env_spec, size, gamma=1.0):
        del env_spec
        self._current_size = 0
        self._current_ptr = 0
        self._n_transitions_stored = 0
        self._size = size
        self._initialized_buffer = False
        self._buffer = {}
        self.gamma = gamma

    def sample_transitions(self, batch_size, replace=True, discount=False, with_subsequence=False):
        """Sample a transition of batch_size.

        Args:
            batch_size(int): The number of transitions to be sampled.
            discount(bool): Whether to discount the sampling prtobailities.
            with_subsequence(bool): Whether to sample the subsequence trajectories of the each sample.
        """
        if discount and not 'time_step' in self._buffer.keys():
            raise ValueError('time_step is not stored in the replay buffer.')
        if discount:
            probabilities = self.gamma ** self._buffer['time_step'][:self._current_size]
            probabilities /= probabilities.sum()
            indices = np.random.choice(self._current_size, 
                                       batch_size, 
                                       replace=replace, 
                                       p=probabilities)
        else:
            indices = np.random.choice(self._current_size, 
                                       batch_size, 
                                       replace=replace)

        sampled_transitions = {key: self._buffer[key][indices] for key in self._buffer.keys()}

        if with_subsequence:
            subseqs = {key: [] for key in self._buffer.keys()}
            for i in indices:
                j = i
                while not self._buffer['last'][j] and j != self._current_ptr:
                    j = (j + 1) % self._size
                if j < i:
                    idx_a = np.arange(i, self._size)
                    idx_b = np.arange(0, j+1)
                    idx = np.concatenate([idx_a, idx_b])
                else:
                    idx = np.arange(i, j+1)
                for key in self._buffer.keys():
                    subseqs[key].append(self._buffer[key][idx])
            sampled_transitions['subsequence'] = subseqs

        return sampled_transitions
        

    def add_transition(self, **kwargs):
        """Add one transition into the replay buffer.

        Args:
            kwargs (dict(str, [numpy.ndarray])): Dictionary that holds
                the transitions.

        """
        transition = {k: [v] for k, v in kwargs.items()}
        self.add_transitions(**transition)

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
        if not self._initialized_buffer:
            self._initialize_buffer(**kwargs)

        assert self._buffer.keys() == kwargs.keys(), 'Keys of the buffer and transitions do not match.'

        num_transitions = len(kwargs['observation'])
        idx = self._get_storage_idx(num_transitions)

        for key, value in kwargs.items():
            self._buffer[key][idx] = np.asarray(value)

        self._n_transitions_stored = min(
            self._size, self._n_transitions_stored + num_transitions)


    def _initialize_buffer(self, **kwargs):
        for key, value in kwargs.items():
            values = np.array(value)
            self._buffer[key] = np.zeros([self._size, *values.shape[1:]],
                                          dtype=values.dtype)
        self._initialized_buffer = True

    def _get_storage_idx(self, size_increment=1):
        """Get the storage index for the episode to add into the buffer.

        Args:
            size_increment(int): The number of storage indeces that new
                transitions will be placed in.

        Returns:
            numpy.ndarray: The indeces to store size_incremente transitions at.

        """
        if self._current_ptr + size_increment < self._size:
            idx = np.arange(self._current_ptr,
                            self._current_ptr + size_increment)
            self._current_ptr += size_increment
        else:
            overflow = size_increment - (self._size - self._current_ptr)
            idx_a = np.arange(self._current_ptr, self._size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self._current_ptr = overflow

        # Update replay size
        self._current_size = min(self._size,
                                 self._current_size + size_increment)

        if size_increment == 1:
            idx = idx[0]
        return idx
    
    def clear(self):
        """Clear buffer."""
        self._buffer.clear()
        self._current_size = 0
        self._current_ptr = 0
        self._n_transitions_stored = 0
        self._initialized_buffer = False

    @property
    def full(self):
        """Whether the buffer is full.

        Returns:
            bool: True of the buffer has reachd its maximum size.
                False otherwise.

        """
        return self._current_size == self._size

    @property
    def n_transitions_stored(self):
        """Return the size of the replay buffer.

        Returns:
            int: Size of the current replay buffer.

        """
        return self._n_transitions_stored
