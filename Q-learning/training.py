import numpy as np
import optuna
import gymnasium as gym
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import DiscretizedObservationWrapper, EpsilonGreedyPolicy

def train_q_learning(alpha, gamma, n_bins, epsilon_decay, n_episodes=100_000, epsilon_start=1.0):
    # Environment Setup
    env = gym.make("MountainCar-v0")
    env = DiscretizedObservationWrapper(env, bins=[n_bins, n_bins])

    # Q-Table
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    epsilon = epsilon_start
    rewards_history = []
    
    # Training Loop
    for episode in range(n_episodes):
        state_discrete, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Create policy instance and get action
            policy = EpsilonGreedyPolicy(epsilon)
            action = policy(q_table[state_discrete])

            # Execute action
            next_state_discrete, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Q-Learning update
            old_value = q_table[state_discrete, action]
            next_max = np.max(q_table[next_state_discrete]) if not done else 0
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state_discrete, action] = new_value

            state_discrete = next_state_discrete
            
        # Epsilon decay
        epsilon = max(0.05, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        
        # Progress feedback
        if episode % 50000 == 0:
            avg_reward = np.mean(rewards_history[-1000:]) if len(rewards_history) >= 1000 else np.mean(rewards_history)
            print(f"Episode {episode}, Avg Reward (last 1000): {avg_reward:.2f}, Epsilon: {epsilon:.4f}")
    
    # Close environment
    env.close()
    return rewards_history

if __name__ == "__main__":
    print("Starting Q-Learning training...")
    
    print("\n=== Training with optimal parameters ===")
    comb1_rewards = train_q_learning(0.1436, 0.9972, 23, 0.9953)
    print(f"1st Training completed. Final average reward: {np.mean(comb1_rewards[-1000:]):.2f}")

    print("\n=== Training with standard parameters ===")
    comb2_rewards = train_q_learning(0.1, 0.99, 20, 0.99)
    print(f"2nd Training completed. Final average reward: {np.mean(comb2_rewards[-1000:]):.2f}")

    print("\n=== Training with aggressive parameters ===")
    comb3_rewards = train_q_learning(0.2, 0.98, 15, 0.98)
    print(f"3rd Training completed. Final average reward: {np.mean(comb3_rewards[-1000:]):.2f}")
    
    print("\nAll training sessions completed!")