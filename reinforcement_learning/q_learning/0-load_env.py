#!/usr/bin/env python3
"""Module to load the FrozenLake environment."""

try:
    import gymnasium as gym
    _IS_GYMNASIUM = True
except ImportError:
    import gym
    _IS_GYMNASIUM = False


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment.

    Args:
        desc (list of lists, optional): Custom description of the map.
        map_name (str, optional): Pre-made map to load.
        is_slippery (bool, optional): Determines if the ice is slippery.

    Returns:
        gym.Env: The loaded FrozenLake environment.
    """
    if _IS_GYMNASIUM:
        env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name,
                       is_slippery=is_slippery, render_mode="ansi")
    else:
        env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name,
                       is_slippery=is_slippery)
    return env
