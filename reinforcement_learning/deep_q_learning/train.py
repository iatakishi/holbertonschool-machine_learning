#!/usr/bin/env python3
"""Script to train a DQN agent to play Atari Breakout."""

import numpy as np
import gymnasium as gym
from tensorflow import keras
from tensorflow.keras import layers
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


class GymnasiumWrapper(gym.Wrapper):
    """Wrapper to make Gymnasium API compatible with keras-rl2."""
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        # Gymnasium returns (obs, info), keras-rl expects just obs
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        # Gymnasium returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = self.env.step(action)
        # keras-rl expects (obs, reward, done, info)
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
    env = gym.make('BreakoutDeterministic-v4')
    env = GymnasiumWrapper(env)

    nb_actions = env.action_space.n
    input_shape = env.observation_space.shape

    # Build the model
    model = build_model(nb_actions, input_shape)

    # Configure the agent
    memory = SequentialMemory(limit=10000, window_length=1)
    policy = EpsGreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=1000,
        target_model_update=1e-2,
        policy=policy
    )

    dqn.compile(keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])

    # Train the agent
    print("Starting training...")
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

    # Save the final policy network
    dqn.save_weights('policy.h5', overwrite=True)
    print("Training complete. Weights saved to policy.h5")
