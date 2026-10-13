#!/usr/bin/env python3
"""Module to perform Q-learning on the FrozenLake environment."""

import numpy as np

try:
    import gymnasium as gym

    _IS_GYMNASIUM = True
except ImportError:
    import gym

    _IS_GYMNASIUM = False

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning.
    """
    total_rewards = []

    for episode in range(episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state = reset_output[0]
        else:
            state = reset_output

        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, info = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_output

            # Check if the agent fell into a hole and update the reward to -1
            if hasattr(env, 'unwrapped'):
                desc = env.unwrapped.desc
                ncols = env.unwrapped.ncols
            else:
                desc = env.desc
                ncols = env.ncols

            row = next_state // ncols
            col = next_state % ncols
            if desc[row, col] == b'H':
                reward = -1

            if done:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])

            Q[state, action] += alpha * (target - Q[state, action])

            episode_reward += reward
            state = next_state

            if done:
                break

        total_rewards.append(episode_reward)
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q, total_rewards
