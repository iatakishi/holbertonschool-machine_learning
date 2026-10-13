#!/usr/bin/env python3
"""
Module for having a trained agent play an episode in FrozenLake.
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode.

    Args:
        env: The FrozenLakeEnv instance.
        Q: A numpy.ndarray containing the Q-table.
        max_steps: The maximum number of steps in the episode.

    Returns:
        The total rewards for the episode and a list of rendered outputs
        representing the board state at each step.
    """
    state = env.reset()
    # Check if the returned state is a tuple (Gymnasium vs older Gym)
    if isinstance(state, tuple):
        state = state[0]

    rendered_outputs = []
    # Capture the initial state of the board
    rendered_outputs.append(env.render())

    total_rewards = 0

    for _ in range(max_steps):
        # Exploit the Q-table to get the best action
        action = np.argmax(Q[state, :])

        # Take the action
        step_result = env.step(action)

        # Unpack based on Gym version (4 values for old Gym, 5 for Gymnasium)
        if len(step_result) == 4:
            state, reward, done, _ = step_result
        else:
            state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated

        total_rewards += reward

        # Capture the rendered state
        rendered_outputs.append(env.render())

        if done:
            break

    return total_rewards, rendered_outputs
