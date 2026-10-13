#!/usr/bin/env python3
"""Module to perform the Monte Carlo algorithm."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm.

    Args:
        env: environment instance
        V (numpy.ndarray): shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action
        episodes (int): total number of episodes to train over
        max_steps (int): maximum number of steps per episode
        alpha (float): learning rate
        gamma (float): discount rate

    Returns:
        numpy.ndarray: V, the updated value estimate
    """
    for _ in range(episodes):
        # Handle both gym (returns state) and gymnasium (returns state, info)
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            state = reset_out[0]
        else:
            state = reset_out

        episode = []

        for _ in range(max_steps):
            action = policy(state)
            step_out = env.step(action)

            # Handle step output differences between gym (4 items) and gymnasium (5 items)
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_out

            episode.append((state, reward))
            state = next_state

            if done:
                break

        # Calculate returns backwards and update V incrementally
        G = 0
        for t in reversed(range(len(episode))):
            state_t, reward_t = episode[t]
            # Ensure the multiplication operators (*) are present
            G = reward_t + gamma * G
            V[state_t] = V[state_t] + alpha * (G - V[state_t])

    return V
