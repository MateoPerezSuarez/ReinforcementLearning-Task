import numpy as np
import optuna
import gymnasium as gym
from utils import DiscretizedObservationWrapper, EpsilonGreedyPolicy

def train_q_learning(alpha, gamma, n_bins, epsilon_decay, n_episodes=500_000, epsilon_start=1.0):
    # Environment Setup
    env = gym.make("MountainCar-v0")
    env = DiscretizedObservationWrapper(env, n_bins=n_bins)

    # Q-Table: 1D because the wrapper converts to a single index
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    epsilon = epsilon_start
    rewards_history = []
    
    # Training Loop
    for episode in range(n_episodes):
        state_discrete, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = EpsilonGreedyPolicy(epsilon)(q_table[state_discrete])

            # Execute action
            next_state_discrete, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Q-Learning update
            old_value = q_table[state_discrete, action]
            next_max = np.max(q_table[next_state_discrete]) 
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state_discrete, action] = new_value

            state_discrete = next_state_discrete
            
        # Epsilon decay
        epsilon = max(0.05, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
    
    return rewards_history

if __name__ == "__main__":

    comb1_rewards = train_q_learning(0.1436, 0.9972, 23, 0.9953)
    print("1st Training completed. Total rewards:", comb1_rewards)

    comb2_rewards = train_q_learning(0.1, 0.99, 20, 0.99)
    print("2nd Training completed. Total rewards:", comb2_rewards)

    comb3_rewards = train_q_learning(0.2, 0.98, 15, 0.98)
    print("3rd Training completed. Total rewards:", comb3_rewards)