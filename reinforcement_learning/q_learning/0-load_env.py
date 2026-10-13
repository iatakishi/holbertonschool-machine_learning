#!/usr/bin/env python3
"""Module to load the FrozenLake environment from gymnasium."""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium.

    Args:
        desc (list of lists, optional): Custom description of the map.
        map_name (str, optional): Pre-made map to load.
        is_slippery (bool, optional): Determines if the ice is slippery.

    Returns:
        gymnasium.Env: The loaded FrozenLake environment.
    """
    # Create and return the FrozenLake environment with
    # the given parameters
    # Added render_mode="ansi" to allow string rendering
    # of the environment
    env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name,
                   is_slippery=is_slippery, render_mode="ansi")

    return env
