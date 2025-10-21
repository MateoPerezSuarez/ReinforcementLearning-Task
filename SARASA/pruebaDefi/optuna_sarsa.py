import os
import gymnasium as gym
import random
import numpy as np
import optuna
from gymnasium.spaces import Box, Discrete
import gymnasium as gym
import numpy as np
from deustorl.common import EpsilonGreedyPolicy, evaluate_policy, max_policy
from deustorl.sarsa import Sarsa

class DiscretizedObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper que discretiza un espacio de observación Box continuo 
    en un único espacio de observación Discrete.
    """
    def __init__(self, env, bins):
        super().__init__(env)
        
        # 1. Validación y Definición de Bins
        assert isinstance(env.observation_space, Box), "El espacio de observación debe ser Box (continuo)."

        # 'bins' debe ser un array o lista con el número de bins por dimensión (e.g., [20, 20])
        self.n_bins = np.array(bins, dtype=int)
        
        # Obtenemos los límites del entorno (MountainCar: [-1.2, 0.6] para pos, [-0.07, 0.07] para vel)
        low = self.observation_space.low
        high = self.observation_space.high
        
        # Manejo de límites infinitos (importante si usas otros entornos)
        max_abs_value = 1e6 
        self.low = np.clip(low, -max_abs_value, max_abs_value)
        self.high = np.clip(high, -max_abs_value, max_abs_value)

        # Calculamos el ancho de cada bin
        self.bin_widths = (self.high - self.low) / self.n_bins
        
        # Calculamos el tamaño total del nuevo espacio discreto.
        # Es el producto de los bins en cada dimensión (ej: 20 * 20 = 400)
        total_discrete_states = np.prod(self.n_bins)
        
        # 2. Definir el nuevo espacio de observación (Discrete)
        self.observation_space = Discrete(total_discrete_states)
        print("New observation space (Discrete):", self.observation_space)

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
# 3. FUNCIÓN OBJETIVO DE OPTUNA PARA SARSA
# =========================================================================

def objective_sarsa(trial):
    """Optimiza los hiperparámetros para el algoritmo SARSA."""
    
    # ------------------- 3.1. Hiperparámetros de Entorno (Discretización) -------------------
    n_bins_pos = trial.suggest_int('n_bins_pos', 15, 30) 
    n_bins_vel = trial.suggest_int('n_bins_vel', 15, 30)
    n_bins = [n_bins_pos, n_bins_vel]
    env = setup_environment(n_bins)

    # ------------------- 3.2. Hiperparámetros de SARSA -------------------
    # El factor de descuento es crucial en Mountain Car (se necesita visión a largo plazo)
    gamma = trial.suggest_float('gamma', 0.99, 0.9999, log=True) 
    lr = trial.suggest_float('lr', 0.005, 0.5, log=True)      
    epsilon = trial.suggest_float('epsilon', 0.05, 0.2)       
    lrdecay = trial.suggest_float('lrdecay', 0.99, 1.0)
    
    N_STEPS_TRIAL = 80_000 
    N_EVAL_EPISODES = 5 
    
    # ------------------- 3.3. Entrenamiento y Evaluación -------------------
    algo = Sarsa(env, gamma=gamma)
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)
    
    algo.learn(epsilon_greedy_policy, N_STEPS_TRIAL, lr=lr, lrdecay=lrdecay, n_episodes_decay=50) 
    
    avg_reward, _ = evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=N_EVAL_EPISODES, verbose=False)

    # Se guarda el nombre del algoritmo para la base de datos (SARSA)
    trial.set_user_attr('algorithm', 'SARSA') 
    
    return avg_reward


# =========================================================================
# 4. EJECUCIÓN DEL ESTUDIO OPTUNA
# =========================================================================

if __name__ == "__main__":
    STUDY_NAME = 'SARSA_MountainCar_Study'
    STORAGE_URL = 'sqlite:///SARSA_study.db'

    # Eliminar la base de datos anterior para un estudio limpio
    os.system(f"rm -f {STORAGE_URL.split('///')[1]}") 
    
    study = optuna.create_study(
        direction='maximize', 
        study_name=STUDY_NAME,
        storage=STORAGE_URL
    )
    
    # Optuna requiere al menos 50 trials por algoritmo
    N_TRIALS = 50 
    print(f"Iniciando estudio Optuna para SARSA con {N_TRIALS} trials...")

    try:
        study.optimize(objective_sarsa, n_trials=N_TRIALS, n_jobs=-1) 
    except KeyboardInterrupt:
        print("Estudio interrumpido.")
    
    # Resultados
    print("\n--- Resultados SARSA ---")
    print(f"Mejor Recompensa: {study.best_value:.4f}")
    print("Mejores Hiperparámetros:")
    print(study.best_params)
    
    # Guardar los 3 mejores trials
    best_trials = study.trials_dataframe().sort_values('value', ascending=False).head(3)
    print("\nLos 3 mejores trials:")
    print(best_trials[['value', 'params_n_bins_pos', 'params_lr', 'params_gamma', 'params_epsilon']])