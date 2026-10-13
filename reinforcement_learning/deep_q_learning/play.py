#!/usr/bin/env python3
"""Script to play Atari Breakout using the trained DQN agent."""

import numpy as np
import gymnasium as gym
from tensorflow import keras
from tensorflow.keras import layers
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory


class GymnasiumWrapper(gym.Wrapper):
    """Wrapper to make Gymnasium API compatible with keras-rl2."""
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def render(self, **kwargs):
        return self.env.render()


def build_model(nb_actions, input_shape):
    """Builds the CNN model for the DQN agent."""
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(nb_actions, activation='linear'))
    return model


if __name__ == '__main__':
    # Create and wrap the environment
    # Added render_mode='human' so the game window actually displays
    env = gym.make('BreakoutDeterministic-v4', render_mode='human')
    env = GymnasiumWrapper(env)

    nb_actions = env.action_space.n
    input_shape = env.observation_space.shape

    # Build the exact same model architecture
    model = build_model(nb_actions, input_shape)

    # Configure the agent with GreedyQPolicy (no exploration)
    memory = SequentialMemory(limit=10000, window_length=1)
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=1000,
        target_model_update=1e-2,
        policy=policy
    )

    dqn.compile(keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])

    # Load the trained weights
    dqn.load_weights('policy.h5')

    # Play the game and visualize
    print("Starting playback...")
    dqn.test(env, nb_episodes=5, visualize=True)
