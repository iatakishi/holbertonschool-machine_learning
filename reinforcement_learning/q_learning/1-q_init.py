# !/usr/bin/env python3
"""Module to initialize a Q-table for a given environment."""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table.

    Args:
        env: The FrozenLakeEnv instance.

    Returns:
        numpy.ndarray: The Q-table as a numpy
        array of zeros.
    """
    # Get the number of states and actions from
    # the environment's spaces
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize and return the Q-table as a numpy array of zeros
    return np.zeros((n_states, n_actions))
