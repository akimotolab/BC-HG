"""Categorical MLP Policy.

A policy represented by a Categorical distribution
which is parameterized by a multilayer perceptron (MLP).
"""
import akro
import numpy as np
import torch
from torch import nn
from garage.torch import as_torch
from garage.torch.modules import MLPModule
from garage.torch.policies.stochastic_policy import StochasticPolicy


class CategoricalMLPPolicy(StochasticPolicy):
    """CategoricalMLPPolicy.

    A policy that contains a MLP to make prediction based on
    a categorical distribution.

    It only works with akro.Discrete action space.

    Args:
        env_spec (garage.EnvSpec): Environment specification.
        hidden_sizes (tuple[int]): Output dimension of dense layer(s) for
            the MLP. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation. For CategoricalMLPPolicy, this
            is usually None, as the MLP output is treated as logits.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.

    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 name='CategoricalMLPPolicy'):

        if not isinstance(env_spec.action_space, akro.Discrete):
            raise ValueError('CategoricalMLPPolicy only works '
                             'with akro.Discrete action space.')
        if isinstance(env_spec.observation_space, akro.Dict):
            raise ValueError('CategoricalMLPPolicy does not support '
                             'akro.Dict observation spaces directly. '
                             'Flatten the observation space first.')

        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.n

        self._module = MLPModule(
            input_dim=self._obs_dim,
            output_dim=self._action_dim,
            hidden_sizes=list(hidden_sizes),
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)

    @property
    def obs_dim(self):
        """Return the observation dimension."""
        return self._obs_dim

    @property
    def action_dim(self):
        """Return the action dimension."""
        return self._action_dim

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations.
                Shape should be (batch_size, obs_dim).

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors.
        """
        logits = self._module(observations)
        dist = torch.distributions.Categorical(logits=logits)
        mode = torch.argmax(dist.logits, dim=-1)
        return dist, {'mode': mode, 'probs': dist.probs, 'logits': logits}
