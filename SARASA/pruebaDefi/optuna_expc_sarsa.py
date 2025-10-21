import os
import gymnasium as gym
import random
import numpy as np
import optuna

# IMPORTACIONES (Ajusta la ruta si es necesario)
from deustorl.common import EpsilonGreedyPolicy, evaluate_policy, max_policy 
from deustorl.common import DiscretizedObservationWrapper 
from deustorl.expected_sarsa import ExpectedSarsa 
# No necesitamos QLearning ni Sarsa en este script.


# =========================================================================
# 2. FUNCIÓN DE SETUP DEL ENTORNO (Misma que para SARSA)
# =========================================================================

def setup_environment(n_bins):
    """Crea y modifica el entorno MountainCar con discretización."""
    env_base = gym.make("MountainCar-v0", render_mode="ansi") 
    env_base.unwrapped.force = 0.0005 # Coche más débil
    env = DiscretizedObservationWrapper(env_base, bins=n_bins)
    
    seed = 3
    random.seed(seed)
    env.reset(seed=seed)
    
    return env


# =========================================================================
# 3. FUNCIÓN OBJETIVO DE OPTUNA PARA EXPECTED SARSA
# =========================================================================

def objective_expected_sarsa(trial):
    """Optimiza los hiperparámetros para el algoritmo Expected SARSA."""
    
    # ------------------- 3.1. Hiperparámetros de Entorno (Discretización) -------------------
    n_bins_pos = trial.suggest_int('n_bins_pos', 15, 30) 
    n_bins_vel = trial.suggest_int('n_bins_vel', 15, 30)
    n_bins = [n_bins_pos, n_bins_vel]
    env = setup_environment(n_bins)

    # ------------------- 3.2. Hiperparámetros de Expected SARSA -------------------
    # Los rangos suelen ser similares a SARSA
    gamma = trial.suggest_float('gamma', 0.99, 0.9999, log=True) 
    lr = trial.suggest_float('lr', 0.005, 0.5, log=True)      
    epsilon = trial.suggest_float('epsilon', 0.05, 0.2)       
    lrdecay = trial.suggest_float('lrdecay', 0.99, 1.0)
    
    N_STEPS_TRIAL = 80_000 
    N_EVAL_EPISODES = 5 
    
    # ------------------- 3.3. Entrenamiento y Evaluación -------------------
    algo = ExpectedSarsa(env, gamma=gamma)
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)
    
    algo.learn(epsilon_greedy_policy, N_STEPS_TRIAL, lr=lr, lrdecay=lrdecay, n_episodes_decay=50) 
    
    avg_reward, _ = evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=N_EVAL_EPISODES, verbose=False)

    # Se guarda el nombre del algoritmo para la base de datos
    trial.set_user_attr('algorithm', 'ExpectedSARSA') 
    
    return avg_reward


# =========================================================================
# 4. EJECUCIÓN DEL ESTUDIO OPTUNA
# =========================================================================

if __name__ == "__main__":
    STUDY_NAME = 'ExpectedSARSA_MountainCar_Study'
    STORAGE_URL = 'sqlite:///EXPECTED_SARSA_study.db'

    # Eliminar la base de datos anterior para un estudio limpio
    os.system(f"rm -f {STORAGE_URL.split('///')[1]}") 
    
    study = optuna.create_study(
        direction='maximize', 
        study_name=STUDY_NAME,
        storage=STORAGE_URL
    )
    
    # Optuna requiere al menos 50 trials por algoritmo
    N_TRIALS = 50 
    print(f"Iniciando estudio Optuna para Expected SARSA con {N_TRIALS} trials...")

    try:
        study.optimize(objective_expected_sarsa, n_trials=N_TRIALS, n_jobs=-1) 
    except KeyboardInterrupt:
        print("Estudio interrumpido.")
    
    # Resultados
    print("\n--- Resultados Expected SARSA ---")
    print(f"Mejor Recompensa: {study.best_value:.4f}")
    print("Mejores Hiperparámetros:")
    print(study.best_params)
    
    # Guardar los 3 mejores trials
    best_trials = study.trials_dataframe().sort_values('value', ascending=False).head(3)
    print("\nLos 3 mejores trials:")
    print(best_trials[['value', 'params_n_bins_pos', 'params_lr', 'params_gamma', 'params_epsilon']])