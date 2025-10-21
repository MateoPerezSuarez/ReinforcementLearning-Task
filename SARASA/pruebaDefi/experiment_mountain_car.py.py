import os
import gymnasium as gym
import random
import time
import numpy as np

# IMPORTACIONES NECESARIAS
# Asumo que las clases y políticas están aquí.
from deustorl.common import * # Asumo que tienes Sarsa y ExpectedSarsa implementadas.
from deustorl.sarsa import Sarsa 
from deustorl.expected_sarsa import ExpectedSarsa 
# Si tu wrapper de discretización no está en deustorl.common, cámbialo aquí.

from deustorl.helpers import DiscretizedObservationWrapper 

def test(algo, n_steps=60000, **kwargs):
    """Función para ejecutar el aprendizaje y la evaluación del algoritmo."""
    # Mantenemos epsilon a 0.1, aunque esto también puede ser un hiperparámetro para Optuna.
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=0.1) 
    start_time = time.time()
    algo.learn(epsilon_greedy_policy, n_steps, **kwargs)
    print("----- {:0.4f} secs. -----".format(time.time() - start_time))

    # max_policy se asume una política greedy a partir de la Q-table
    return evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100, verbose=False)

os.system("rm -rf ./logs/") 

# ----------------- Configuración del Entorno MOUNTAIN CAR -----------------

# Definición de los bins de discretización
POSITION_BINS = 20
VELOCITY_BINS = 20
N_BINS = [POSITION_BINS, VELOCITY_BINS] 
# El número de estados discretos será 20 x 20 = 400

# 1. Crear el entorno base MountainCar-v0 (ya es discreto en acciones)
env_base = gym.make("MountainCar-v0", render_mode="ansi") 

# 2. Envolver el entorno con el Discretized Observation Wrapper
env = DiscretizedObservationWrapper(env_base, bins=N_BINS)

# La tarea define "solución" de MountainCar-v0 como un promedio de recompensa de -110.0
# La recompensa es -1 por cada paso, y el episodio termina en 200 pasos o al llegar a la meta.
# Un buen agente debe terminar en menos de 110 pasos.

env_base.unwrapped.force = 0.0005 
print(f"Nueva fuerza del motor: {env_base.unwrapped.force}")

# Aumentar la gravedad (0.0025 -> 0.0035)
env_base.unwrapped.gravity = 0.0035 
print(f"Nueva gravedad: {env_base.unwrapped.gravity}")

seed = 3
random.seed(seed)
env.reset(seed=seed) 

n_rounds = 10
# El entrenamiento de Mountain Car a menudo requiere más pasos/episodios que Frozen Lake.
n_steps_per_round = 100_000 

# --- EXPERIMENTACIÓN SARSA ---
print("\n=== Testing SARSA on MountainCar ({}x{} states) ===".format(POSITION_BINS, VELOCITY_BINS))
total_reward_sarsa = 0
for _ in range(n_rounds):
    algo = Sarsa(env)
    # Puedes empezar con hiperparámetros de ejemplo para luego optimizarlos con Optuna.
    avg_reward, _ = test(algo, n_steps=n_steps_per_round, lr=0.1, lrdecay=0.99, n_episodes_decay=50) 
    total_reward_sarsa += avg_reward
print("------\r\nAverage reward SARSA over {} rounds: {:.4f}\r\n------".format(n_rounds, total_reward_sarsa/n_rounds))

# --- EXPERIMENTACIÓN EXPECTED SARSA ---
print("\n=== Testing Expected SARSA on MountainCar ({}x{} states) ===".format(POSITION_BINS, VELOCITY_BINS))
total_reward_expected_sarsa = 0
for _ in range(n_rounds):
    # La clase ExpectedSarsa debe ser importada o implementada.
    algo = ExpectedSarsa(env) 
    # Mismos hiperparámetros iniciales para comparación.
    avg_reward, _ = test(algo, n_steps=n_steps_per_round, lr=0.1, lrdecay=0.99, n_episodes_decay=50) 
    total_reward_expected_sarsa += avg_reward
print("------\r\nAverage reward Expected SARSA over {} rounds: {:.4f}\r\n------".format(n_rounds, total_reward_expected_sarsa/n_rounds))