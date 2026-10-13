# !/usr/bin/env python3
"""Module to determine the next action using the epsilon-greedy strategy."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.

    Args:
        Q (numpy.ndarray): The Q-table.
        state (int): The current state.
        epsilon (float): The epsilon to use for the calculation.

    Returns:
        int: The next action index.
    """
    # Sample p to determine whether to explore or exploit
    p = np.random.uniform()

    # Get the number of possible actions from the Q-table
    n_actions = Q.shape[1]

    if p < epsilon:
        # Explore: pick a random action
        return np.random.randint(n_actions)
    else:
        # Exploit: pick the action with the highest
        # Q-value for the current state
        return np.argmax(Q[state])
