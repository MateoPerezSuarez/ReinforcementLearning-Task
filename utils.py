import numpy as np
import optuna
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import random

class DiscretizedObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper que discretiza un espacio de observación Box continuo 
    en un único espacio de observación Discrete.
    """
    def __init__(self, env, bins):
        super().__init__(env)
        
        assert isinstance(env.observation_space, Box), "Observation space must be Box (continuous)."

        self.n_bins = np.array(bins, dtype=int)
        low = self.observation_space.low
        high = self.observation_space.high
        
        max_abs_value = 1e6 
        self.low = np.clip(low, -max_abs_value, max_abs_value)
        self.high = np.clip(high, -max_abs_value, max_abs_value)

        self.bin_widths = (self.high - self.low) / self.n_bins
        
        total_discrete_states = np.prod(self.n_bins)
        
        self.observation_space = Discrete(total_discrete_states)

    def observation(self, observation):
        clipped_obs = np.clip(observation, self.low, self.high - 1e-6)
        discrete_obs = ((clipped_obs - self.low) / self.bin_widths).astype(int)
        
        return np.ravel_multi_index(discrete_obs, self.n_bins)


class EpsilonGreedyPolicy():
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def __call__(self, q_values):
        if random.random() < self.epsilon:
            return random.randint(0,len(q_values)-1) # Exploration
        else:
            return np.argmax(q_values) # Exploitation