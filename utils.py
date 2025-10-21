import numpy as np
import optuna
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import random

class DiscretizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, Box), "The observation space must be a Box space."

        low = self.observation_space.low if low is None else np.array(low)
        high = self.observation_space.high if high is None else np.array(high)

        # Clip values in low and high to be within the specified bounds
        max_abs_value = 1e6
        low = np.clip(low, -max_abs_value, max_abs_value)
        high = np.clip(high, -max_abs_value, max_abs_value)

        # Calculate the width of each bin
        self.bin_widths = (high - low) / n_bins

        self.n_bins = n_bins
        self.low = low
        self.high = high
        self.ob_shape = self.observation_space.shape

        # Define the new discrete observation space
        self.observation_space = Discrete(n_bins ** low.size)
        print("New observation space:", self.observation_space)

    def observation(self, obs):
        """Transforms a continuous observation into a discrete one."""
        # Map the observation to its bin index
        bins = ((obs - self.low) / self.bin_widths).astype(int)

        # Clip to ensure indices are within [0, n_bins-1]
        bins = np.clip(bins, 0, self.n_bins - 1)

        # Encode the bins array (e.g. [i_pos, i_vel]) into a single index (scalar)
        state_index = 0
        for i in range(len(bins)):
            # Formula: index = i_0 + i_1*N + i_2*N^2 + ...
            state_index += bins[i] * (self.n_bins ** i)
            
        return int(state_index)


class EpsilonGreedyPolicy():
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def __call__(self, q_values):
        if random.random() < self.epsilon:
            return random.randint(0,len(q_values)-1) # Exploration
        else:
            return np.argmax(q_values) # Exploitation