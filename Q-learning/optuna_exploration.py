import numpy as np
import optuna
import gymnasium as gym
from utils import DiscretizedObservationWrapper, EpsilonGreedyPolicy


# Training Hyperparameters
EPSILON = 1.0 
N_STEPS_TRIAL = 500_000
PRUNING_REPORT_INTERVAL = 50000

def objective(trial):

    alpha = trial.suggest_float('alpha', 0.05, 0.5)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    n_bins_pos = trial.suggest_int('n_bins_pos', 20, 35)
    n_bins_vel = trial.suggest_int('n_bins_vel', 20, 35)
    n_bins = [n_bins_pos, n_bins_vel]
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.995, 0.9999)

    # Environment Setup
    env = gym.make("MountainCar-v0")
    env = DiscretizedObservationWrapper(env, bins=n_bins)

    # Q-Table: 1D porque el wrapper convierte a índice único
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    # Parámetros de entrenamiento
    epsilon = EPSILON

    rewards_history = []

    # Iniciar el primer episodio
    state_discrete, _ = env.reset()
    total_reward = 0
    
    # Training Loop
    for step in range(N_STEPS_TRIAL):

        action = EpsilonGreedyPolicy(epsilon)(q_table[state_discrete])

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

        if done:
            rewards_history.append(total_reward)
            state_discrete, _ = env.reset()
            total_reward = 0 #reset total reward for new episode

        # Report metrics for pruning
        if step % PRUNING_REPORT_INTERVAL == 0 and step > 0:
            if len(rewards_history) > 50:
                current_mean_reward = np.mean(rewards_history[-50:])
            else:
                # If not enough data, use a default low value
                current_mean_reward = np.mean(rewards_history) if rewards_history else -200

            trial.report(current_mean_reward, step)

            if trial.should_prune():
                # Stops the trial if the performance is not promising
                raise optuna.exceptions.TrialPruned()


    # Evaluate final performance
    final_score = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
    return final_score


if __name__ == "__main__":

    N_TRIALS = 70
    
    print(f"Intitializing Optuna hyperparameter optimization with {N_TRIALS} trials.")
    print(f"Each trial will train for {N_STEPS_TRIAL} steps.\n")

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
            score = row['value']
            
            print(f"\n  --- Combinación {i+1} (Score: {score:.4f}) ---")
            print(f"  alpha: {row['params_alpha']:.6f}")
            print(f"  gamma: {row['params_gamma']:.6f}")
            print(f"  epsilon_decay: {row['params_epsilon_decay']:.6f}")
            print(f"  Discretización (Pos/Vel): [{int(row['params_n_bins_pos'])}, {int(row['params_n_bins_vel'])}]")
    
    print("\nEstas combinaciones ofrecen la mejor convergencia en 500.000 pasos.")