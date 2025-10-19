import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium import ObservationWrapper, spaces
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

class DiscretizedObservationWrapper(ObservationWrapper):
    """Wrapper para discretizar observaciones continuas en estados discretos"""
    
    def __init__(self, env, num_bins=10):
        super().__init__(env)
        self.num_bins = num_bins
        
        # Obtener límites del espacio de observación continuo
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        
        self.bins = [np.linspace(low[i], high[i], num_bins) 
                     for i in range(len(low))]
        
        # Nuevo espacio: discreto con n_states posibles
        self.n_states = num_bins ** len(low)
        self.observation_space = spaces.Discrete(self.n_states)
    
    def observation(self, obs):
        """Convierte observación continua a estado discreto"""
        indices = [np.digitize(obs[i], self.bins[i]) - 1 
                   for i in range(len(obs))]
        # Clamp a rango válido
        indices = [np.clip(idx, 0, self.num_bins - 1) 
                   for idx in indices]
        # Convertir a índice único
        state = sum(indices[i] * (self.num_bins ** i) 
                    for i in range(len(indices)))
        return state

def create_env():
    """Crea el ambiente Mountain Car con discretización"""
    env = gym.make("MountainCar-v0")
    env = DiscretizedObservationWrapper(env, num_bins=10)
    env = TimeLimit(env, max_episode_steps=500)
    return env

def q_learning_train(env, alpha, gamma, epsilon, num_episodes, writer, trial_num):
    """Q-Learning con logging en TensorBoard"""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    
    epsilon_decay = 0.995
    min_epsilon = 0.01
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            max_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * max_next - Q[state, action])
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Logging en TensorBoard
        writer.add_scalar(f'trial_{trial_num}/episode_reward', episode_reward, episode)
        writer.add_scalar(f'trial_{trial_num}/epsilon', epsilon, episode)
        
        if (episode + 1) % 50 == 0:
            print(f"  Trial {trial_num} | Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.4f}")
    
    return Q

def sarsa_train(env, alpha, gamma, epsilon, num_episodes, writer, trial_num):
    """SARSA con logging en TensorBoard"""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    
    epsilon_decay = 0.995
    min_epsilon = 0.01
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            
            episode_reward += reward
            state = next_state
            action = next_action
            steps += 1
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        writer.add_scalar(f'trial_{trial_num}/episode_reward', episode_reward, episode)
        writer.add_scalar(f'trial_{trial_num}/epsilon', epsilon, episode)
        
        if (episode + 1) % 50 == 0:
            print(f"  Trial {trial_num} | Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.4f}")
    
    return Q

def expected_sarsa_train(env, alpha, gamma, epsilon, num_episodes, writer, trial_num):
    """Expected SARSA con logging en TensorBoard"""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    
    epsilon_decay = 0.995
    min_epsilon = 0.01
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            policy_probs = np.ones(n_actions) * epsilon / n_actions
            policy_probs[np.argmax(Q[next_state])] += 1 - epsilon
            expected_value = np.sum(policy_probs * Q[next_state])
            
            Q[state, action] += alpha * (reward + gamma * expected_value - Q[state, action])
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        writer.add_scalar(f'trial_{trial_num}/episode_reward', episode_reward, episode)
        writer.add_scalar(f'trial_{trial_num}/epsilon', epsilon, episode)
        
        if (episode + 1) % 50 == 0:
            print(f"  Trial {trial_num} | Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.4f}")
    
    return Q

def train_best_models(results_file="optuna_results.json"):
    """Entrena los 3 mejores modelos con más episodios"""
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    best_trials = results["best_trials"]
    
    # Crear directorio para logs
    log_dir = f"runs/mountain_car_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    print("=" * 70)
    print("ENTRENAMIENTO CON LOS 3 MEJORES MODELOS")
    print("=" * 70)
    
    for trial_num, trial_data in enumerate(best_trials, 1):
        print(f"\n{'=' * 70}")
        print(f"Trial {trial_num}: {trial_data['params']['algorithm']}")
        print(f"Score Optuna: {trial_data['value']:.4f}")
        print(f"Parámetros: {trial_data['params']}")
        print(f"{'=' * 70}")
        
        env = create_env()
        
        params = trial_data['params']
        algorithm = params['algorithm']
        alpha = params['alpha']
        gamma = params['gamma']
        epsilon = params['epsilon']
        num_episodes = 500  # Más episodios que en Optuna
        
        if algorithm == "Q-Learning":
            Q = q_learning_train(env, alpha, gamma, epsilon, num_episodes, writer, trial_num)
        elif algorithm == "SARSA":
            Q = sarsa_train(env, alpha, gamma, epsilon, num_episodes, writer, trial_num)
        else:  # Expected SARSA
            Q = expected_sarsa_train(env, alpha, gamma, epsilon, num_episodes, writer, trial_num)
        
        env.close()
        print(f"\n✓ Trial {trial_num} completado")
        
        # Guardar Q-table
        np.save(f"Q_table_trial_{trial_num}.npy", Q)
    
    writer.close()
    print(f"\n{'=' * 70}")
    print(f"Entrenamientos completados. Logs guardados en: {log_dir}")
    print("Para visualizar: tensorboard --logdir=runs")
    print("=" * 70)

if __name__ == "__main__":
    train_best_models()