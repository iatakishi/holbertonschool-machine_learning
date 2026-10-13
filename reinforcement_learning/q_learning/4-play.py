#!/usr/bin/env python3
"""Module to play an episode using a trained Q-table."""

import numpy as np

try:
    import gymnasium as gym

    _IS_GYMNASIUM = True
except ImportError:
    import gym

    _IS_GYMNASIUM = False


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode.

    Args:
        env: The FrozenLakeEnv instance.
        Q (numpy.ndarray): The Q-table.
        max_steps (int, optional): Maximum number of
        steps in the episode. Defaults to 100.

    Returns:
        tuple: (total_rewards, rendered_outputs)
    """
    # Handle both gym (returns state) and gymnasium (returns state, info)
    reset_output = env.reset()
    if isinstance(reset_output, tuple):
        state = reset_output[0]
    else:
        state = reset_output

    total_rewards = 0
    rendered_outputs = []

    # Handle rendering differences
    if _IS_GYMNASIUM:
        rendered_outputs.append(env.render())
    else:
        rendered_outputs.append(env.render(mode='ansi'))

    for step in range(max_steps):
        action = np.argmax(Q[state])

        # Handle step output differences
        step_output = env.step(action)
        if len(step_output) == 5:
            next_state, reward, terminated, truncated, info = step_output
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_output

        total_rewards += reward

        if _IS_GYMNASIUM:
            rendered_outputs.append(env.render())
        else:
            rendered_outputs.append(env.render(mode='ansi'))

        state = next_state

        if done:
            break

    return total_rewards, rendered_outputs
