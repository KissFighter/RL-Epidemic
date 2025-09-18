# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **simplified** reinforcement learning project that implements epidemic control using the SIR (Susceptible-Infected-Recovered) epidemiological model. The current implementation focuses on a streamlined approach with:

- **Simplified SIR Environment**: 3-dimensional state space [S, I, R] only
- **Simple Q-Learning Agent**: Tabular Q-learning with state discretization
- **Streamlined Training**: Focused on core epidemic control concepts

The system balances infection control with economic costs through Q-learning, providing a clean educational implementation of RL for epidemic control.

## Environment Setup and Development Commands

### Environment Activation
```bash
# The project uses a Python virtual environment
source epidemic_rl_env/bin/activate

# Install dependencies if needed
pip install numpy matplotlib scipy
```

### Key Development Commands

```bash
# Q-Learning training (with reproducible seed)
python q_learning.py

# SARSA training (with reproducible seed)
python sarsa.py

# Inverse Reinforcement Learning (IRL)
python inverse_rl.py

# All training now uses fixed random seeds for reproducibility
# Default seed: 42 for consistency across runs
```

### Model Management
```bash
# Q-Learning models
results/q_learning_model.pkl

# SARSA models
results/sarsa_model.pkl

# IRL models
training_results/irl_model.pkl

# All models include seed information for reproducibility
# Load a model programmatically:
# agent.load_model('results/q_learning_model.pkl')

# View training outputs
ls results/ training_results/
```

## Architecture Overview

### Core Components Integration

The system follows a simplified RL environment-agent pattern focused on educational clarity:

1. **Simple SIR Environment** (`simple_sir_environment.py`): 
   - Implements discrete-time SIR epidemiological dynamics
   - State: [S%, I%, R%] (3D normalized) - simplified from original 5D
   - Actions: {0: no isolation, 1: partial isolation, 2: full isolation}
   - Reward: Balances infection penalty and economic cost

2. **Simple Q-Learning Agent** (`simple_train.py`):
   - **Single Implementation**: Tabular Q-learning with state discretization (default 8 bins per dimension)
   - Uses epsilon-greedy exploration with decay
   - Direct learning from environment interactions
   - Focused on core Q-learning concepts without complexity

3. **Training Framework** (`simple_train.py`):
   - Single-file implementation with training and testing functions
   - Implements basic checkpointing and visualization
   - Streamlined for educational purposes and quick experimentation

### Key Architectural Decisions

- **State Discretization**: Continuous SIR states are binned for tabular Q-learning. State bins parameter (default 8) controls resolution vs. memory trade-off.
- **Reward Design**: Simplified function balancing infection control and economic costs. Modify `_calculate_reward()` in SimpleSIREpidemicEnv for different optimization objectives.
- **Action Space**: Three discrete intervention levels (none, partial, full isolation) for simplicity and clarity.
- **Model Persistence**: Uses pickle for Q-table serialization, includes epsilon and state_bins parameters.

### Current Implementation Focus

The current simplified implementation uses **model-free Q-learning** for educational clarity:

**Advantages of Current Approach:**
- Simple implementation and debugging
- Clear understanding of core RL concepts
- Direct learning from environment interactions
- Lower computational overhead per step
- Well-established theoretical foundations
- Easy to modify and experiment with

### Extension Points

- **Custom Epidemiological Models**: Modify SimpleSIREpidemicEnv to implement SEIR, SIRV, or spatial models
- **Advanced RL Algorithms**: Replace SimpleQLearningAgent with deep RL implementations (DQN, PPO, A3C, etc.)
- **Model-Based Methods**: Add model-based approaches like Dyna-Q or Monte Carlo Tree Search (MCTS)
- **Multi-Agent Systems**: Extend for multiple regions/decision makers
- **Continuous Control**: Modify action space for fine-grained intervention levels
- **Enhanced State Space**: Add more variables like vaccination rates, mobility, demographics

### Training and Evaluation Workflow

#### Current Simplified Workflow:
1. **Environment Initialization**: Set population size (5000), disease parameters (beta=0.3, gamma=0.1), episode length (100 steps), **random seed**
2. **Agent Configuration**: Set learning rate (0.1), exploration schedule (epsilon=0.1 fixed), state discretization (8 bins), **random seed**
3. **Training Loop**: Episode execution → Q-learning/SARSA updates → periodic progress reporting → model saving
4. **Policy Extraction**: Trained Q-table provides state→action policy mapping
5. **Testing & Visualization**: Test trained model, generate epidemic curves and action sequences
6. **IRL Training**: Learn reward weights from expert demonstrations using Maximum Margin IRL

#### Reproducibility Features:
- **Fixed Random Seeds**: All components (environment, agents, IRL) use configurable seeds (default: 42)
- **Deterministic Training**: Same seed produces identical results across runs
- **Seed Storage**: Seeds saved in model metadata for experiment reproducibility
- **Consistent Evaluation**: Test runs use same seeds as training for fair comparison

### Performance Considerations

**Current Implementation:**
- Training time scales with state space size (state_bins^3 for 3D state = 8^3 = 512 states)
- Lower memory usage due to simplified state space
- Faster training due to reduced complexity
- Default 500 episodes provides good convergence
- **Reproducible Results**: Fixed seeds ensure identical outcomes across runs
- **Multiple Algorithms**: Q-Learning (off-policy), SARSA (on-policy), IRL (inverse learning)

**Optimization Tips:**
- Adjust state_bins (default 8) to balance memory vs. learning quality
- Modify learning_rate and epsilon_decay for different learning speeds
- Shorter max_steps (default 100) for faster episodes
- Visualization can be disabled during training for speed

### State and Action Spaces

**State Vector** (normalized to [0,1]):
- S/population: Fraction susceptible
- I/population: Fraction infected  
- R/population: Fraction recovered

**Action Encoding**:
- 0: No isolation (β unchanged, no economic cost)
- 1: Partial isolation (β reduced to 50%, 20% economic cost)
- 2: Full isolation (β reduced to 10%, 50% economic cost)

The reward function balances infection penalty (based on infected population) and economic penalty (based on isolation level).

### Random Seed Usage

**Reproducibility Features**:
- **Environment**: `SIREpidemicEnv(seed=42)` - Controls episode initialization and dynamics
- **Q-Learning**: `QLearningAgent(seed=42)` - Controls ε-greedy action selection
- **SARSA**: `SARSAAgent(seed=42)` - Controls ε-greedy policy behavior
- **IRL**: `MaxMarginIRL(seed=42)` - Controls demonstration generation and optimization

**Usage Examples**:
```python
# Q-Learning with reproducible seed
agent, env = train_q_learning(episodes=500, seed=42)

# SARSA with reproducible seed
agent, env = train_sarsa(episodes=500, seed=42)

# IRL with reproducible seed
irl = train_irl_from_expert('results/q_learning_model.pkl', seed=42)

# Load and analyze IRL results
loaded_irl = load_and_analyze_irl('training_results/irl_model.pkl')
```

**Seed Management**:
- Default seeds: Q-Learning/SARSA (42), IRL (42), Testing (42)
- Seeds saved in model metadata for experiment reproduction
- Consistent seeding across environment, agent, and IRL components
- All random operations (action selection, episode initialization) are deterministic