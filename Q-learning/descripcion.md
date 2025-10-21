# Q-Learning Algorithm
## Initial Configuration: Environment and Q-Table
### Step 1: Environment and Continuous State Space (MountainCar)
In the MountainCar-v0 environment, the car's state is described with two continuous numbers (decimals):

- Position: A value between $-1.2$ and $0.6$.
- Velocity: A value between $-0.07$ and $0.07$

Since Q-Learning needs discrete states (integers) to index a table, we cannot use position and velocity directly.

### Step 2: State Discretization (The Mapping)
For Q-Learning to work, we have to discretize the state space. This means dividing the continuous range of position and velocity into a finite number of bins.

We have implemented a `DiscretizedObservationWrapper` that allows Optuna to optimize the number of bins between **20-25** for both dimensions. This configuration creates between 400 (20×20) and 625 (25×25) discrete states, finding the optimal balance between granularity and computational efficiency for convergence in **5000 episodes**.

### Step 3: Q-Table Initialization
The Q-Table is the agent's "memory". It is a two-dimensional table where "quality" or expected reward values are stored.

- **Dimensions**: `(env.observation_space.n, env.action_space.n)` where the wrapper converts the 2D space to a unique index.
- **For MountainCar**: Between 1,200 and 1,875 total Q values (discrete states × 3 actions).
- **Initialization**: `q_table = np.zeros(...)` - all Q(s,a) values start at zero.

## The Training Loop: Episodes and Steps
**Implemented configuration**: 5000 episodes per trial, with initial epsilon = 1.0 and decay optimized by Optuna between 0.995-0.9999.

### Step 4: Action Selection ($\epsilon$-Greedy)
**Implementation**: Initial `epsilon = 1.0`, decay `epsilon = max(0.05, epsilon * epsilon_decay)` where `epsilon_decay` is optimized by Optuna in [0.995, 0.9999]. Maintains 5% minimum exploration.

### Step 5: Environment Interaction
The agent executes the selected action ($a$) in the current state ($s$). The environment returns:

1. A new state ($s'$).
2. A reward ($R$): For MountainCar, it's $-1$ per step (penalty) and $0$ if it reaches the goal.
3. Whether the episode has ended (done).

### Step 6: Q-Table Update
**Implemented equation**: `new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)`

**Hyperparameters optimized by Optuna:**
- **α (learning rate)**: [0.05, 0.5] - balance between speed and stability
- **γ (discount factor)**: [0.9, 0.999] - importance of future rewards for MountainCar

### Step 7: Transition
The new state $s'$ becomes the current state $s$, and the process repeats from Step 4 until the episode ends.

## 3. Optuna: Finding the Best Hyperparameters
**Study configuration**: 30 trials, maximizing the mean reward of the last 100 episodes.

**Hyperparameters**
- `alpha = trial.suggest_float('alpha', 0.05, 0.5)`
- `gamma = trial.suggest_float('gamma', 0.9, 0.999)` 
- `n_bins = trial.suggest_int('n_bins', 20, 25)`
- `epsilon_decay = trial.suggest_float('epsilon_decay', 0.995, 0.9999)`

**Process**: Each trial trains for 5000 episodes, evaluates with `np.mean(rewards_history[-100:])`, and reports for intelligent pruning. Optuna learns from previous trials to suggest better combinations using the TPE algorithm.

### Optimization Results
The results of the three most successful combinations during hyperparameter exploration are as follows:

| Combination | Alpha | Gamma | N_bins | Epsilon_decay |
|-------------|-------|-------|--------|---------------|
| 1st | 0.1436 | 0.9972 | 23 | 0.9953 |
| 2nd | 0.2237 | 0.9690 | 20 | 0.9974 |
| 3rd | 0.1695 | 0.9794 | 22 | 0.9990 |