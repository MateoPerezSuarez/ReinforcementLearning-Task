# Posición: Un valor entre $-1.2 y $0.6.
# Velocidad: Un valor entre $-0.07 y $0.07.import gymnasium as gym
# Como Q-Learning necesita estados discretos (enteros) para indexar una tabla,
# no podemos usar la posición y la velocidad directamente.
import numpy as np
import optuna

# 1. Función para discretizar el estado
def discretize_state(state, env, n_bins):
    # Lógica de mapeo de estado continuo a índice discreto (i_pos, i_vel)
    pos, vel = state
    pos_bins = np.linspace(-1.2, 0.6, n_bins)
    vel_bins = np.linspace(-0.07, 0.07, n_bins)
    i_pos = np.digitize(pos, pos_bins) - 1
    i_vel = np.digitize(vel, vel_bins) - 1
    return (i_pos, i_vel)

# 2. Función objetivo para Optuna
def objective(trial):
    # 2.1 Sugerir Hiperparámetros
    alpha = trial.suggest_float('alpha', 0.05, 0.5)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    n_bins = trial.suggest_int('n_bins', 15, 30) # Ej. Bins para pos y vel
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.99, 0.9999)

    # Inicialización del entorno
    env = gym.make("MountainCar-v0")
    
    # Inicialización de la Q-Tabla (tamaño: n_bins x n_bins x n_acciones)
    q_table = np.zeros((n_bins, n_bins, env.action_space.n))
    
    # Parámetros de entrenamiento
    n_episodes = 3000
    epsilon = 1.0 
    
    rewards_history = []
    
    # 2.2 Entrenar al Agente (Bucle de Episodios)
    for episode in range(n_episodes):
        state, _ = env.reset()
        current_state_discrete = discretize_state(state, env, n_bins)
        total_reward = 0
        done = False
        
        while not done:
            # Política epsilon-greedy para seleccionar acción
            if np.random.random() < epsilon:
                action = env.action_space.sample() # Exploración
            else:
                action = np.argmax(q_table[current_state_discrete]) # Explotación
                
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            next_state_discrete = discretize_state(next_state, env, n_bins)

            # 2.3 Ecuación de Actualización de Q-Learning
            old_value = q_table[current_state_discrete + (action,)]
            next_max = np.max(q_table[next_state_discrete])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[current_state_discrete + (action,)] = new_value

            current_state_discrete = next_state_discrete
            
        # Decaimiento de epsilon
        epsilon = max(0.05, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        
        # Opcional: Reportar progreso a Optuna para Pruning
        # trial.report(np.mean(rewards_history[-100:]), episode) 
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    # 2.4 Evaluar y Retornar Métrica
    # Usamos la recompensa media de los últimos 100 episodios
    final_score = np.mean(rewards_history[-100:])
    return final_score # Optuna intentará maximizar este valor

# 3. Ejecutar el estudio de Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10) # Número de pruebas a ejecutar

# 4. Resultados
print("Mejores hiperparámetros encontrados:", study.best_params)
print("Mejor recompensa media (puntuación):", study.best_value)