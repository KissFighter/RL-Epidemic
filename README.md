# SIR Epidemic Control: Q-Learning vs SARSA

This project compares **off-policy Q-learning** and **on-policy SARSA** algorithms for controlling epidemic spread using the SIR (Susceptible-Infected-Recovered) epidemiological model.

## Project Overview

The project consists of four core Python files:

- **`environment.py`**: SIR epidemic environment implementation
- **`q_learning.py`**: Q-learning algorithm (off-policy) with fixed epsilon
- **`sarsa.py`**: SARSA algorithm (on-policy) with fixed epsilon
- **`inverse_rl.py`**: Inverse Reinforcement Learning using Maximum Margin IRL

## Key Features

- **Clean Implementation**: Simple, readable code focused on core concepts
- **Fixed Epsilon**: No epsilon decay for clear policy analysis
- **Explicit Policy Distinction**: Clear separation of behavior and target policies in Q-learning
- **Reproducible Results**: Fixed random seeds ensure identical outcomes across runs
- **Inverse RL**: Learn reward weights from expert demonstrations
- **Educational Focus**: Designed for understanding fundamental RL concepts

## Environment Setup

### Using Python venv (Recommended)
```bash
# Create virtual environment
python3 -m venv epidemic_rl_env

# Activate environment
source epidemic_rl_env/bin/activate  # On macOS/Linux
# or
epidemic_rl_env\Scripts\activate.bat  # On Windows

# Install required packages
pip install numpy matplotlib scipy
```

## Quick Start

### 1. Test Environment
```python
from environment import SIREpidemicEnv

# Create environment with reproducible seed
env = SIREpidemicEnv(population=5000, max_steps=100, seed=42)
state = env.reset()
print(f"Initial state: {state}")

# Take action
next_state, reward, done, info = env.step(1)  # Partial isolation
```

### 2. Train Q-Learning (Off-Policy)
```python
from q_learning import train_q_learning

# Train Q-learning agent with reproducible seed
agent, env = train_q_learning(episodes=500, max_steps=100, seed=42)
```

### 3. Train SARSA (On-Policy)
```python
from sarsa import train_sarsa

# Train SARSA agent with reproducible seed
agent, env = train_sarsa(episodes=500, max_steps=100, seed=42)
```

### 4. Test Trained Models
```python
from q_learning import test_q_learning
from sarsa import test_sarsa
from inverse_rl import train_irl_from_expert

# Test both algorithms with reproducible seeds
test_q_learning(seed=42)
test_sarsa(seed=42)

# Train and test Inverse RL
irl = train_irl_from_expert('results/q_learning_model.pkl', seed=42)
```

## Core Components

### Environment (`environment.py`)

**SIR Epidemic Model:**
- **State Space**: [S%, I%, R%] - 3D normalized values [0,1]
- **Action Space**: 3 discrete actions
  - 0: No isolation (β unchanged, no cost)
  - 1: Partial isolation (50% β reduction, 20% economic cost)
  - 2: Full isolation (90% β reduction, 50% economic cost)
- **Reward Function**: Balances infection penalty and economic costs

**Key Parameters:**
- `population`: Total population (default: 5000)
- `beta`: Base transmission rate (default: 0.3)
- `gamma`: Recovery rate (default: 0.1)
- `max_steps`: Simulation length (default: 100)
- `seed`: Random seed for reproducibility (default: None)

### Q-Learning (`q_learning.py`)

**Off-Policy Characteristics:**
- **Behavior Policy**: ε-greedy (ε=0.1, fixed) for action selection
- **Target Policy**: Greedy for Q-value updates
- **Update Rule**: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

**Key Features:**
- Explicit `behavior_policy()` and `target_policy()` methods
- Fixed epsilon (no decay) for consistent analysis
- Policy information analysis tools

### SARSA (`sarsa.py`)

**On-Policy Characteristics:**
- **Single Policy**: ε-greedy (ε=0.1, fixed) for both action selection and updates
- **Update Rule**: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
- **Consistency**: Same policy used throughout learning

**Key Features:**
- Single `epsilon_greedy_policy()` method
- Fixed epsilon for fair comparison with Q-learning
- Policy comparison tools
- Reproducible training with seed parameter

### Inverse RL (`inverse_rl.py`)

**Maximum Margin IRL Characteristics:**
- **Goal**: Learn reward weights from expert demonstrations
- **Algorithm**: Maximum Margin IRL with linear features
- **Features**: [infection_cost, economic_cost]
- **Output**: Learned weight vector balancing infection control vs economic cost

**Key Features:**
- Expert demonstration generation from trained Q-learning agents
- Diverse policy generation for comparison
- Lightweight numerical diagnostics (no plotting)
- Complete save/load functionality with metadata
- Reproducible results with seed management

## Algorithm Comparison

| Feature | Q-Learning (Off-Policy) | SARSA (On-Policy) |
|---------|-------------------------|-------------------|
| **Policy Structure** | Behavior ≠ Target | Single Policy |
| **Action Selection** | ε-greedy | ε-greedy |
| **Q-Update** | Uses max Q(s',a') | Uses actual Q(s',a') |
| **Exploration Impact** | Learning unaffected by exploration | Learning reflects exploration |
| **Convergence** | To optimal policy | To ε-greedy policy |
| **Reproducibility** | Fixed seed support | Fixed seed support |

## Reproducibility and Random Seeds

### Why Seeds Matter
- **Stochastic Components**: ε-greedy policies, episode initialization, IRL optimization
- **Research Requirements**: Reproducible results for fair algorithm comparison
- **Debugging**: Consistent behavior for easier troubleshooting

### Seed Implementation

**Environment Seeds:**
```python
# Environment with reproducible dynamics
env = SIREpidemicEnv(population=5000, seed=42)
```

**Agent Seeds:**
```python
# Q-Learning agent with reproducible exploration
q_agent = QLearningAgent(state_size=3, action_size=3, seed=42)

# SARSA agent with reproducible policy
s_agent = SARSAAgent(state_size=3, action_size=3, seed=42)
```

**Training with Seeds:**
```python
# Reproducible training runs
agent, env = train_q_learning(episodes=500, seed=42)
agent, env = train_sarsa(episodes=500, seed=42)
irl = train_irl_from_expert('results/q_learning_model.pkl', seed=42)
```

**Seed Storage:**
- All models save seed information in metadata
- IRL models include complete seed history for demonstration generation
- Load functions automatically restore seed context

### Default Seeds
- **Main Training**: 42 (Q-Learning, SARSA, IRL)
- **Testing**: 42 (consistent with training)
- **IRL Testing**: 123 (different from main to test generalization)

## Usage Examples

### Environment Interaction
```python
from environment import SIREpidemicEnv

# Create environment with seed for reproducibility
env = SIREpidemicEnv(population=10000, initial_infected=50, seed=42)
state = env.reset()

for step in range(100):
    action = 1  # Partial isolation
    state, reward, done, info = env.step(action)
    if done:
        break

# Visualize results
env.render(save_path="epidemic_simulation.png")

# Get statistics
stats = env.get_statistics()
print(f"Peak infections: {stats['peak_infections']}")
print(f"Attack rate: {stats['attack_rate']:.1%}")
```

### Policy Analysis
```python
from q_learning import QLearningAgent
from sarsa import SARSAAgent

# Create agents with seeds for reproducible behavior
q_agent = QLearningAgent(state_size=3, action_size=3, epsilon=0.1, seed=42)
s_agent = SARSAAgent(state_size=3, action_size=3, epsilon=0.1, seed=42)

# Analyze policies
state = [0.7, 0.2, 0.1]  # Example state

q_info = q_agent.get_policy_info(state)
print(f"Q-learning - Target action: {q_info['target_action']}")
print(f"Q-learning - Behavior probs: {q_info['behavior_probabilities']}")

s_info = s_agent.get_policy_info(state)
print(f"SARSA - Best action: {s_info['best_action']}")
print(f"SARSA - Policy probs: {s_info['policy_probabilities']}")
```

## Training and Results

Both algorithms save:
- **Models**: `results/{algorithm}_model.pkl`
- **Training curves**: `results/{algorithm}_training.png`
- **Test results**: `results/{algorithm}_test.png`

**Expected Performance:**
- Q-learning: Learns optimal policy for epidemic control
- SARSA: Learns ε-greedy policy balancing exploration and control

## Project Structure

```
RL+Epidemic/
├── environment.py              # SIR epidemic environment
├── q_learning.py               # Q-learning implementation (off-policy)
├── sarsa.py                    # SARSA implementation (on-policy)
├── inverse_rl.py               # Inverse Reinforcement Learning
├── README.md                   # This file
├── CLAUDE.md                   # Development guidance
├── environment详解.md           # Environment explanation (Chinese)
├── q_learning详解.md            # Q-learning explanation (Chinese)
├── sarsa详解.md                 # SARSA explanation (Chinese)
├── epidemic_rl_env/            # Python virtual environment
├── results/                    # Q-Learning and SARSA outputs
│   ├── q_learning_model.pkl    # Trained Q-learning model (with seed)
│   ├── sarsa_model.pkl         # Trained SARSA model (with seed)
│   ├── *_training.png          # Training curves
│   └── *_test.png              # Test visualizations
└── training_results/           # IRL outputs
    ├── irl_model.pkl           # Trained IRL model (with seed metadata)
    └── irl_test.pkl            # IRL test results
```

## Documentation

- **English**: `README.md` (this file), `CLAUDE.md`
- **Chinese**: `environment详解.md`, `q_learning详解.md`, `sarsa详解.md`

## Key Design Decisions

### Fixed Epsilon
- **Q-learning**: ε=0.1 (fixed) for clear off-policy demonstration
- **SARSA**: ε=0.1 (fixed) for fair comparison
- **Benefit**: Easier analysis of policy differences without decay complexity

### State Discretization
- 8 bins per dimension → 8³ = 512 total states
- Balance between resolution and learning speed
- Adjustable via `state_bins` parameter

### Simplified Reward
- Infection penalty: -(I/population) × 100
- Economic penalty: -economic_cost × 20
- Clean trade-off without complex weighting

## Customization

### Modify Environment Parameters
```python
env = SIREpidemicEnv(
    population=10000,     # Larger population
    beta=0.4,            # Higher transmission
    gamma=0.15,          # Faster recovery
    max_steps=200        # Longer episodes
)
```

### Adjust Learning Parameters
```python
agent = QLearningAgent(
    state_size=3,
    action_size=3,
    learning_rate=0.05,   # Slower learning
    epsilon=0.15,         # More exploration
    state_bins=10,        # Finer discretization
    seed=42               # Reproducible behavior
)
```

### Custom Reward Function
Modify `_calculate_reward()` in `environment.py`:
```python
def _calculate_reward(self, economic_cost: float) -> float:
    infection_penalty = -(self.I / self.population) * 50  # Reduce infection weight
    economic_penalty = -economic_cost * 30                # Increase economic weight
    return infection_penalty + economic_penalty
```

## Next Steps

1. **Parameter Tuning**: Experiment with learning rates, epsilon values
2. **Extended Environment**: Add vaccination, multiple populations
3. **Advanced Algorithms**: Implement Expected SARSA, Double Q-learning
4. **Deep RL**: Replace tabular methods with neural networks
5. **Multi-Agent**: Compare policies across different regions

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
- Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics
- Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine learning*