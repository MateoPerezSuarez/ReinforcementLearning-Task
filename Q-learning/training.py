import numpy as np
import gymnasium as gym
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import DiscretizedObservationWrapper, EpsilonGreedyPolicy

def train_q_learning(alpha, gamma, position_bins, velocity_bins, epsilon_decay, 
                     n_episodes=100_000, epsilon_start=1.0, run_name="default"):
    # Create TensorBoard writer
    log_dir = f"runs/{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    
    # Log hyperparameters
    writer.add_text('Hyperparameters', f"""
    ## Configuration: {run_name}
    - **Alpha (learning rate)**: {alpha}
    - **Gamma (discount factor)**: {gamma}
    - **Position bins**: {position_bins}
    - **Velocity bins**: {velocity_bins}
    - **Total states**: {position_bins * velocity_bins}
    - **Epsilon decay**: {epsilon_decay}
    - **Epsilon start**: {epsilon_start}
    - **Episodes**: {n_episodes}
    """)
    
    # Environment Setup
    env = gym.make("MountainCar-v0")
    env = DiscretizedObservationWrapper(env, bins=[position_bins, velocity_bins])

    # Q-Table
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    epsilon = epsilon_start
    rewards_history = []
    episode_lengths = []
    success_count = 0  # Count successful episodes
    
    # Training Loop
    for episode in range(n_episodes):
        state_discrete, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Create policy instance and get action
            policy = EpsilonGreedyPolicy(epsilon)
            action = policy(q_table[state_discrete])

            # Execute action
            next_state_discrete, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # Q-Learning update
            old_value = q_table[state_discrete, action]
            next_max = np.max(q_table[next_state_discrete]) if not done else 0
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state_discrete, action] = new_value

            state_discrete = next_state_discrete
        
        # Check successful episode
        if total_reward > -200:
            success_count += 1
            
        # Epsilon decay
        epsilon = max(0.05, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        episode_lengths.append(steps)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Training/Episode_Reward', total_reward, episode)
        writer.add_scalar('Training/Episode_Length', steps, episode)
        writer.add_scalar('Training/Epsilon', epsilon, episode)
        
        # Log Q-table statistics every 100 episodes
        if episode % 100 == 0:
            q_mean = np.mean(q_table)
            q_max = np.max(q_table)
            q_min = np.min(q_table)
            q_std = np.std(q_table)
            q_nonzero = np.count_nonzero(q_table)
            q_sparsity = (q_table.size - q_nonzero) / q_table.size * 100
            
            writer.add_scalar('Q_Table/Mean', q_mean, episode)
            writer.add_scalar('Q_Table/Max', q_max, episode)
            writer.add_scalar('Q_Table/Min', q_min, episode)
            writer.add_scalar('Q_Table/Std', q_std, episode)
            writer.add_scalar('Q_Table/Sparsity_%', q_sparsity, episode)
        
        # Log moving averages every 100 episodes
        if episode % 100 == 0 and episode > 0:
            avg_reward_100 = np.mean(rewards_history[-100:])
            avg_length_100 = np.mean(episode_lengths[-100:])
            success_rate_100 = sum(1 for r in rewards_history[-100:] if r > -200) / 100 * 100
            
            writer.add_scalar('Training/Avg_Reward_100', avg_reward_100, episode)
            writer.add_scalar('Training/Avg_Length_100', avg_length_100, episode)
            writer.add_scalar('Training/Success_Rate_100_%', success_rate_100, episode)
        
        # Log moving averages every 1000 episodes
        if episode % 1000 == 0 and episode > 0:
            avg_reward_1000 = np.mean(rewards_history[-1000:])
            avg_length_1000 = np.mean(episode_lengths[-1000:])
            success_rate_1000 = sum(1 for r in rewards_history[-1000:] if r > -200) / 1000 * 100
            
            writer.add_scalar('Training/Avg_Reward_1000', avg_reward_1000, episode)
            writer.add_scalar('Training/Avg_Length_1000', avg_length_1000, episode)
            writer.add_scalar('Training/Success_Rate_1000_%', success_rate_1000, episode)
        
        # Progress feedback
        if episode % 10000 == 0 and episode > 0:
            avg_reward = np.mean(rewards_history[-1000:])
            avg_length = np.mean(episode_lengths[-1000:])
            success_rate = sum(1 for r in rewards_history[-1000:] if r > -200) / 1000 * 100
            print(f"Episode {episode:6d} | Avg Reward: {avg_reward:7.2f} | Avg Length: {avg_length:6.1f} | "
                  f"Success Rate: {success_rate:5.1f}% | Epsilon: {epsilon:.4f}")
    
    # Log final Q-table distribution
    writer.add_histogram('Q_Table/Final_Distribution', q_table.flatten(), n_episodes)
    
    # Calculate final statistics
    final_avg_reward = np.mean(rewards_history[-1000:])
    final_avg_length = np.mean(episode_lengths[-1000:])
    final_success_rate = sum(1 for r in rewards_history[-1000:] if r > -200) / 1000 * 100
    best_reward = max(rewards_history)
    
    # Log hyperparameters with results for comparison
    writer.add_hparams(
        {
            'alpha': alpha,
            'gamma': gamma,
            'position_bins': position_bins,
            'velocity_bins': velocity_bins,
            'total_states': position_bins * velocity_bins,
            'epsilon_decay': epsilon_decay,
        },
        {
            'hparam/final_avg_reward': final_avg_reward,
            'hparam/final_avg_length': final_avg_length,
            'hparam/final_success_rate': final_success_rate,
            'hparam/best_reward': best_reward,
        }
    )
    
    # Close environment and writer
    env.close()
    writer.close()
    
    print(f"✓ Training completed | Final Avg Reward: {final_avg_reward:.2f} | "
          f"Success Rate: {final_success_rate:.1f}%")
    print(f"  TensorBoard logs: {log_dir}\n")
    
    return rewards_history, episode_lengths

if __name__ == "__main__":
    print("="*80)
    print("Q-LEARNING TRAINING - HYPERPARAMETER COMPARISON")
    print("="*80)
    print("\nTo visualize results in real-time, run in another terminal:")
    print("  tensorboard --logdir=runs")
    print("  Then open: http://localhost:6006\n")
    print("="*80)
    
    # 1st combination (best)
    print("\n[1/3] Training: HIGH RESOLUTION")
    print("-" * 80)
    print("Strategy: More bins for better state representation")
    print("Config: alpha=0.3682, gamma=0.9769, pos_bins=32, vel_bins=23, eps_decay=0.9969")
    comb1_rewards, comb1_lengths = train_q_learning(
        alpha=0.3682,
        gamma=0.9769,
        position_bins=32,
        velocity_bins=23,
        epsilon_decay=0.9969,
        run_name="1st_combination"
    )
    
    # Configuration 2
    print("[2/3] Training: BALANCED")
    print("-" * 80)
    print("Strategy: Balance between resolution and learning speed")
    print("Config: alpha=0.2693, gamma=0.9711, pos_bins=28, vel_bins=24, eps_decay=0.9996")
    comb2_rewards, comb2_lengths = train_q_learning(
        alpha=0.2693,
        gamma=0.9711,
        position_bins=28,
        velocity_bins=24,
        epsilon_decay=0.9996,
        run_name="2nd_combination"
    )
    
    # Configuration 3
    print("[3/3] Training: FAST LEARNING")
    print("-" * 80)
    print("Strategy: Fewer bins and higher learning rate for faster convergence")
    print("Config: alpha=0.3946, gamma=0.9780, pos_bins=31, vel_bins=24, eps_decay=0.993")
    comb3_rewards, comb3_lengths = train_q_learning(
        alpha=0.3946,
        gamma=0.9780,
        position_bins=31,
        velocity_bins=24,
        epsilon_decay=0.993,
        run_name="3rd_combination"
    )
    
    # Print final comparison
    print("="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"{'Configuration':<20} {'Final Avg Reward':>18} {'Final Avg Length':>18}")
    print("-"*80)
    print(f"{'High Resolution':<20} {np.mean(comb1_rewards[-1000:]):>18.2f} {np.mean(comb1_lengths[-1000:]):>18.1f}")
    print(f"{'Balanced':<20} {np.mean(comb2_rewards[-1000:]):>18.2f} {np.mean(comb2_lengths[-1000:]):>18.1f}")
    print(f"{'Fast Learning':<20} {np.mean(comb3_rewards[-1000:]):>18.2f} {np.mean(comb3_lengths[-1000:]):>18.1f}")
    print("="*80)
    print("\n✓ All training sessions completed!")
    print("\nTo compare results:")
    print("  1. Run: tensorboard --logdir=runs")
    print("  2. Open: http://localhost:6006")
    print("  3. Check the 'HPARAMS' tab for side-by-side comparison")
    print("  4. Use the 'SCALARS' tab to compare learning curves")
    print("="*80)