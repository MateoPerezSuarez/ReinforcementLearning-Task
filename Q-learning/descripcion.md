# Algoritmo Q-Learning
## Configuración Inicial: El Entorno y la Q-Tabla
### Paso 1: Entorno y Espacio de Estados Continuo (MountainCar)
En el entorno MountainCar-v0 el estado del coche se describe con dos números continuos (decimales):

- Posición: Un valor entre $-1.2$ y $0.6$.
- Velocidad: Un valor entre $-0.07$ y $0.07$

Como Q-Learning necesita estados discretos (enteros) para indexar una tabla, no podemos usar la posición y la velocidad directamente

### Paso 2: La Discretización de Estados (El Mapeo) 🗺️
Para que el Q-Learning funcione, debes discretizar el espacio de estados. Esto significa dividir el rango continuo de la posición y la velocidad en un número finito de bins.

**ESTRO DESCRIBIRLO CON LOS VALORES QUE USEMOS:** Si decides usar 20 bins para la posición y 20 para la velocidad, tendrías un total de $20 \times 20 = 400$ estados discretos.Una posición real como $-0.5532$ podría mapearse al bin de posición número 5, y una velocidad de $0.0211$ al bin de velocidad número 12.El estado discreto es ahora el par de índices $(5, 12)$.

### Paso 3: Inicialización de la Q-Tabla
La Q-Tabla es la "memoria" del agente. Es una tabla tridimensional donde se almacenan los valores de "calidad" o recompensa esperada.

- Dimensiones: (Bins de Posición) $\times$ (Bins de Velocidad) $\times$ (Número de Acciones).
- Para MountainCar: $N_{bins} \times N_{bins} \times 3$ (Las 3 acciones son: empujar izquierda, no hacer nada, empujar derecha).
- Contenido Inicial: Todos los valores $Q(s, a)$ se inicializan a cero o a un número pequeño y aleatorio. Esto significa que al principio, el agente piensa que todas las acciones son igual de inútiles (o útiles) en todas las situaciones.

## El Bucle de Entrenamiento: Episodios y Pasos
El entrenamiento ocurre a lo largo de muchos episodios. Un episodio comienza con el coche en la posición inicial y termina cuando alcanza la bandera (meta) o el número máximo de pasos.

### Paso 4: Selección de la Acción ($\epsilon$-Greedy)
En cada paso del episodio, el agente debe decidir qué acción tomar:
- **Exploración** (Probabilidad $\epsilon$): El agente elige una acción al azar. Esto es crucial para descubrir nuevas estrategias que podrían ser mejores.
- **Explotación** (Probabilidad $1-\epsilon$): El agente consulta la Q-Tabla para el estado actual $(i_{pos}, i_{vel})$ y elige la acción con el valor Q más alto (la que ha sido más exitosa en el pasado).

Al inicio, $\epsilon$ es alto (ej. 1.0), por lo que el agente explora mucho. Con el tiempo, $\epsilon$ decae, y el agente explota (usa lo que ha aprendido) cada vez más.

### Paso 5: Interacción con el Entorno
El agente ejecuta la acción seleccionada ($a$) en el estado actual ($s$). El entorno devuelve:

1. Un nuevo estado ($s'$).
2. Una recompensa ($R$): Para MountainCar, es $-1$ por cada paso (castigo) y $0$ si alcanza la meta.
3. Si el episodio ha terminado (done).

### Paso 6: El Corazón del Q-Learning (Actualización de la Q-Tabla)
Este es el paso fundamental. El agente usa la información que acaba de recibir ($s, a, R, s'$) para actualizar el valor $Q(s, a)$ usando la Ecuación de Bellman para Q-Learning:
$$\text{Nuevo } Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
1. **El Error Temporal (TD Error):** Lo que está entre corchetes es el error de predicción (qué tan equivocado estaba el valor Q anterior).
    - **Lo Esperado:** $R + \gamma \max_{a'} Q(s', a')$
        - **$R$:** La recompensa inmediata recibida.
        - **$\gamma \max_{a'} Q(s', a')$:** La mejor recompensa futura que se puede obtener desde el nuevo estado $s'$ (el valor Q más alto de las acciones posibles en $s'$). Esta es la parte off-policy.
    - **Lo Predicho:** $Q(s, a)$ (el valor Q que tenías antes de la acción).
2. **La Actualización:** Multiplicas el error por la tasa de aprendizaje ($\alpha$) y lo añades al valor $Q(s, a)$ antiguo.
ACTUALIZAR CON LO QUE HAGAMOS CON EL VALOR DE $\alpha$
    - Si $\alpha$ es grande, el agente aprende rápido pero puede ser inestable.
    - Si $\alpha$ es pequeño, el aprendizaje es lento pero más estable.

### Paso 7: Transición
El nuevo estado $s'$ se convierte en el estado actual $s$, y el proceso se repite desde el Paso 4 hasta que el episodio termina.

## 3. Optuna: Encontrando los Mejores Hiperparámetros
Los hiperparámetros ($\alpha, \gamma, \epsilon$ decay, $N_{bins}$) controlan el aprendizaje. Elegirlos manualmente es difícil. Optuna automatiza esto.
### Paso 8: La Función Objetivo
Defines una función (objective) que hace lo siguiente:
- **Pregunta a Optuna:** "Dame un valor para $\alpha$, dame un valor para $\gamma$, etc."
- **Entrena:** Ejecuta todo el proceso de Q-Learning (Pasos 1 a 7) con esos valores.
- **Evalúa:** Mide el rendimiento (ej., la recompensa media de los últimos 100 episodios).Informa a Optuna: Devuelve esa recompensa media.

### Paso 9: El Estudio de Optimización
Optuna ejecuta esta función objetivo (ej., 100 veces)ACTUALIZAR CON LOS PASOS QUE HAGAMOS Y LA DESCRIPCION DEL NUESTRO. En cada prueba, utiliza algoritmos inteligentes (TPE, CMA-ES) para sugerir combinaciones de hiperparámetros que probablemente den mejores resultados basándose en las pruebas anteriores.

### Paso 10: El Resultado Óptimo
Al finalizar, Optuna te dirá: "La mejor combinación de hiperparámetros es esta, y con ellos, el agente consiguió esta recompensa media máxima."

Resumen del Flujo Lógico:
1. Optuna Sugiere Parámetros ($\alpha, \gamma, N_{bins}$).
2. Q-Learning Inicializa la Q-Tabla usando $N_{bins}$.
3. Episodios: El agente interactúa y $\epsilon$ decae.
4. Actualización: Se usa $\alpha$ y $\gamma$ en la Ecuación de Bellman.
5. Rendimiento: Se calcula la recompensa media.
6. Optuna Aprende: Optuna usa este resultado para elegir mejores parámetros para la siguiente prueba.