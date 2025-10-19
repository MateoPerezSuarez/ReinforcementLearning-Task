import optuna
from optuna.pruners import MedianPruner
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium import ObservationWrapper, spaces
import numpy as np
import json
from datetime import datetime
import os

# Configuración
STUDY_NAME = "mountain_car_tabular"
STORAGE = f"sqlite:///{STUDY_NAME}.db"
NUM_TRIALS = 150
SEED = 42

np.random.seed(SEED)

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

def q_learning(env, alpha, gamma, epsilon, num_episodes=100):
    """Implementación de Q-Learning"""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    
    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
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
        
        rewards.append(episode_reward)
        epsilon = max(0.01, epsilon * 0.995)
    
    return np.mean(rewards[-10:])

def sarsa(env, alpha, gamma, epsilon, num_episodes=100):
    """Implementación de SARSA"""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    
    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        episode_reward = 0
        done = False
        
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
        
        rewards.append(episode_reward)
        epsilon = max(0.01, epsilon * 0.995)
    
    return np.mean(rewards[-10:])

def expected_sarsa(env, alpha, gamma, epsilon, num_episodes=100):
    """Implementación de Expected SARSA"""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    
    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
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
        
        rewards.append(episode_reward)
        epsilon = max(0.01, epsilon * 0.995)
    
    return np.mean(rewards[-10:])

def objective(trial):
    """Función objetivo para Optuna"""
    algorithm = trial.suggest_categorical("algorithm", ["Q-Learning", "SARSA", "Expected SARSA"])
    alpha = trial.suggest_float("alpha", 0.01, 0.5, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.99, log=True)
    epsilon = trial.suggest_float("epsilon", 0.1, 1.0)
    
    env = create_env()
    
    try:
        if algorithm == "Q-Learning":
            score = q_learning(env, alpha, gamma, epsilon, num_episodes=100)
        elif algorithm == "SARSA":
            score = sarsa(env, alpha, gamma, epsilon, num_episodes=100)
        else:  # Expected SARSA
            score = expected_sarsa(env, alpha, gamma, epsilon, num_episodes=100)
        
        env.close()
        return score
    
    except Exception as e:
        env.close()
        return float('-inf')

def run_study():
    """Ejecuta el estudio Optuna"""
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = MedianPruner()
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=False
    )
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando optimización con Optuna...")
    print(f"Número total de trials: {NUM_TRIALS}")
    print("-" * 60)
    
    study.optimize(objective, n_trials=NUM_TRIALS, show_progress_bar=True)
    
    print("\n" + "=" * 60)
    print("OPTIMIZACIÓN COMPLETADA")
    print("=" * 60)
    
    # Mejores trials
    trials_df = study.trials_dataframe()
    print(f"\nMejor valor alcanzado: {study.best_value:.4f}")
    print(f"\nMejores 3 combinaciones:")
    for i, trial in enumerate(study.best_trials[:3], 1):
        print(f"\n{i}. Score: {trial.value:.4f}")
        print(f"   Parámetros: {trial.params}")
    
    # Guardar resultados
    results = {
        "best_trials": [
            {
                "value": trial.value,
                "params": trial.params,
                "trial_number": trial.number
            }
            for trial in study.best_trials[:3]
        ],
        "num_trials": len(study.trials),
        "timestamp": datetime.now().isoformat()
    }
    
    with open("optuna_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResultados guardados en optuna_results.json")
    print(f"Base de datos: {STUDY_NAME}.db")

if __name__ == "__main__":
    run_study()