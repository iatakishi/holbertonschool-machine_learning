#!/usr/bin/env python3
"""
Module for playing an episode using a trained Q-table.
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
    # Reset the environment to start the episode
    reset_result = env.reset()
    # Handle compatibility between Gym and Gymnasium
    if isinstance(reset_result, tuple):
        state = reset_result[0]
    else:
        state = reset_result

    total_rewards = 0.0
    rendered_outputs = []

    # Append the initial state of the board
    rendered_outputs.append(env.render())

    for _ in range(max_steps):
        # Exploit the Q-table to find the best action
        action = np.argmax(Q[state, :])

        # Take the action
        step_result = env.step(action)

        # Handle compatibility between Gym and Gymnasium step return values
        if len(step_result) == 4:
            next_state, reward, done, info = step_result
        else:
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated

        total_rewards += reward

        # Append the new state of the board
        rendered_outputs.append(env.render())

        if done:
            break

        state = next_state

    return total_rewards, rendered_outputs
