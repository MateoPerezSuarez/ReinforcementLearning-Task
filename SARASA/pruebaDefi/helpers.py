from gymnasium.spaces import Box, Discrete
import gymnasium as gym
import numpy as np

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


    def observation(self, obs):
        """
        Discretiza la observación continua (posición, velocidad) y 
        la mapea a un único estado entero (aplanamiento).
        """
        # 1. Discretizar cada valor de la observación
        # Clipping para asegurar que la observación esté dentro de los límites [low, high]
        obs = np.clip(obs, self.low, self.high)
        
        # Convertir la observación continua a un índice de bin (0 a N-1)
        discretized_obs = (obs - self.low) / self.bin_widths
        discretized_obs = np.floor(discretized_obs).astype(int)
        
        # Aseguramos que el último bin N-1 no se desborde si el valor es 'high'
        discretized_obs = np.clip(discretized_obs, 0, self.n_bins - 1)
        
        # 2. Aplanar (Flattening): Convertir el vector de índices discretos a un único entero
        # Si tienes [10 (pos), 5 (vel)] en un grid de 20x20, el estado es 10 + 5*20 = 110.
        state_id = 0
        multiplier = 1
        for i in range(len(discretized_obs)):
            state_id += discretized_obs[i] * multiplier
            multiplier *= self.n_bins[i]
            
        return int(state_id)