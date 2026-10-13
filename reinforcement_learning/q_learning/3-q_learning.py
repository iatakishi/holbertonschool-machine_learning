# !/usr/bin/env python3
"""Module to perform Q-learning on the FrozenLake environment."""

import numpy as np

# Import the epsilon_greedy function from the previous task
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning.

    Args:
        env: The FrozenLakeEnv instance.
        Q (numpy.ndarray): The Q-table.
        episodes (int, optional): Total number of episodes. Defaults to 5000.
        max_steps (int, optional): Maximum number of steps per episode. Defaults to 100.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount rate. Defaults to 0.99.
        epsilon (float, optional): Initial threshold for epsilon greedy. Defaults to 1.
        min_epsilon (float, optional): Minimum value for epsilon. Defaults to 0.1.
        epsilon_decay (float, optional): Decay rate for epsilon. Defaults to 0.05.

    Returns:
        tuple: (Q, total_rewards)
            - Q (numpy.ndarray): The updated Q-table.
            - total_rewards (list): List containing the rewards per episode.
    """
    # Initialize the list to store total rewards per episode
    total_rewards = []

    # Iterate over the specified number of episodes
    for episode in range(episodes):
        # Reset the environment and get the initial state
        state, info = env.reset()
        episode_reward = 0

        # Iterate over the maximum number of steps per episode
        for step in range(max_steps):
            # Determine the next action using epsilon-greedy strategy
            action = epsilon_greedy(Q, state, epsilon)

            # Take the action in the environment
            next_state, reward, terminated, truncated, info = env.step(action)

            # Check if the agent fell into a hole and update the reward to -1
            row = next_state // env.unwrapped.ncols
            col = next_state % env.unwrapped.ncols
            if env.unwrapped.desc[row, col] == b'H':
                reward = -1

            # Determine if the episode is done
            done = terminated or truncated

            # Calculate the target value for the Q-learning update
            if done:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])

            # Update the Q-table using the Q-learning formula
            Q[state, action] += alpha * (target - Q[state, action])

            # Accumulate the reward for the current episode
            episode_reward += reward

            # Move to the next state
            state = next_state

            # Break the loop if the episode is done
            if done:
                break

        # Store the total reward for the current episode
        total_rewards.append(episode_reward)

        # Decay the epsilon value for the next episode
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    # Return the updated Q-table and the list of total rewards
    return Q, total_rewards
