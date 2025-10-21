import numpy as np
import optuna
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

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

# Training Hyperparameters
EPSILON = 1.0 
N_EPISODES = 5000

def objective(trial):

    alpha = trial.suggest_float('alpha', 0.05, 0.5)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    n_bins = trial.suggest_int('n_bins', 20, 25)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.995, 0.9999)

    # Environment Setup
    env = gym.make("MountainCar-v0")
    env = DiscretizedObservationWrapper(env, n_bins=n_bins)

    # Q-Table: 1D porque el wrapper convierte a índice único
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    # Parámetros de entrenamiento
    epsilon = EPSILON

    rewards_history = []
    
    # Training Loop
    for episode in range(N_EPISODES):
        state_discrete, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-Greedy Action Selection
            if np.random.random() < epsilon:
                action = env.action_space.sample() # Exploration
            else:
                action = np.argmax(q_table[state_discrete]) # Exploitation

            # Execute action
            next_state_discrete, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Q-Learning update
            old_value = q_table[state_discrete, action]
            next_max = np.max(q_table[next_state_discrete]) # max_a' Q(s', a')
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state_discrete, action] = new_value

            state_discrete = next_state_discrete
            
        # Epsilon decay
        epsilon = max(0.05, epsilon * epsilon_decay)
        rewards_history.append(total_reward)

        # Report metrics for pruning
        if episode % 100 == 0 and episode > 0:
            current_mean_reward = np.mean(rewards_history[-100:])
            trial.report(current_mean_reward, episode)
            if trial.should_prune():
                # Stops the trial if the performance is not promising
                raise optuna.exceptions.TrialPruned()

    # Evaluate final performance
    final_score = np.mean(rewards_history[-100:]) # Avg reward over last 100 episodes
    return final_score


if __name__ == "__main__":

    N_TRIALS = 70
    
    print(f"Intitializing Optuna hyperparameter optimization with {N_TRIALS} trials.")
    print(f"Each trial will train for {N_EPISODES} episodes.\n")

    # Reward maximization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    print("\n" + "="*50)
    print("           FINAL RESULTS OF OPTIMIZATION")
    print("="*50)
    
    # Best Hyperparameters
    print("\nBest Hyperparameters (Q-Learning):")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"   - {key}: {value:.6f}")

    # Best Score
    print(f"\nBest Mean Reward (Score): {study.best_value:.4f}")

    # 3 best trials
    top_trials = study.trials_dataframe().sort_values(by='value', ascending=False).head(3)

    print("\nTop 3 Combinations of Hyperparameters for Long Training:")
    top_combinations = []
    
    for i in range(3):
        row = top_trials.iloc[i]
        params = {
            'alpha': row['params_alpha'],
            'gamma': row['params_gamma'],
            'n_bins': int(row['params_n_bins']),
            'epsilon_decay': row['params_epsilon_decay']
        }
        score = row['value']
        top_combinations.append(params)
        
        print(f"\n  --- Combination {i+1} (Score: {score:.4f}) ---")
        for key, value in params.items():
            print(f"  {key}: {value}")