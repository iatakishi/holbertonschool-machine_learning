#!/usr/bin/env python3
"""Module to play an episode using a trained Q-table."""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode.

    Args:
        env: The FrozenLakeEnv instance.
        Q (numpy.ndarray): The Q-table.
        max_steps (int, optional): Maximum number of steps in the episode. Defaults to 100.

    Returns:
        tuple: (total_rewards, rendered_outputs)
    """
    # Handle both gym (returns state) and gymnasium (returns state, info)
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        state = reset_out[0]
    else:
        state = reset_out

    total_rewards = 0
    rendered_outputs = []

    # Handle rendering differences: gym returns None for default render,
    # while gymnasium returns the string if render_mode="ansi" is set.
    out = env.render()
    if out is None:
        out = env.render(mode='ansi')
    rendered_outputs.append(out)

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        step_out = env.step(action)

        # Handle step output differences between gym (4 items) and gymnasium (5 items)
        if len(step_out) == 5:
            next_state, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_out

        total_rewards += reward

        out = env.render()
        if out is None:
            out = env.render(mode='ansi')
        rendered_outputs.append(out)

        state = next_state
        if done:
            break

    return total_rewards, rendered_outputs
