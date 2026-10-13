#!/usr/bin/env python3
"""Module to play an episode using a trained Q-table."""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode.

    Args:
        env: The FrozenLakeEnv instance.
        Q (numpy.ndarray): The Q-table.
        max_steps (int, optional): Maximum number of steps in the episode.
        Defaults to 100.

    Returns:
        tuple: (total_rewards, rendered_outputs)
            - total_rewards (float): The total rewards for the episode.
            - rendered_outputs (list): List of rendered outputs
            representing the board state at each step.
    """
    # Reset the environment and get the initial state
    state, info = env.reset()

    total_rewards = 0
    rendered_outputs = []

    # Render and store the initial state of the board
    rendered_outputs.append(env.render())

    # Iterate over the maximum number of steps
    for step in range(max_steps):
        # Exploit the Q-table: pick the action with the highest Q-value
        # for the current state
        action = np.argmax(Q[state])

        # Take the action in the environment
        next_state, reward, terminated, truncated, info = env.step(action)

        # Accumulate the reward for the episode
        total_rewards += reward

        # Render and store the state of the board after the action is taken
        rendered_outputs.append(env.render())

        # Move to the next state
        state = next_state

        # Break the loop if the episode is done (agent reached goal
        # or fell in hole)
        if terminated or truncated:
            break

    # Return the total accumulated rewards and the list of
    # rendered board states
    return total_rewards, rendered_outputs
