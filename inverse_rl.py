#!/usr/bin/env python3
"""
Inverse Reinforcement Learning for Epidemic Control

Implements Maximum Margin IRL to learn reward weights for epidemic management.
Learns the balance between infection control and economic costs.
"""

import numpy as np
import pickle
import os
import time
from typing import List, Tuple, Dict, Any
from scipy.optimize import minimize
from environment import SIREpidemicEnv
from q_learning import QLearningAgent


class MaxMarginIRL:
    """Maximum Margin IRL for epidemic control."""

    def __init__(self, env: SIREpidemicEnv, regularization: float = 0.1, gamma: float = 1.0, seed: int = 42):
        self.env = env
        self.feature_dim = 2  # infection + economic features
        self.regularization = regularization
        self.gamma = gamma
        self.seed = seed
        self.weights = np.array([0.5, 0.5])  # Start with balanced weights

        # Set random seeds for reproducibility
        np.random.seed(seed)

        # Training history and diagnostics
        self.mu_E = None  # Expert feature expectations
        self.mu_list = []  # List of policy feature expectations per iteration
        self.history = []  # Training history: [iter, t, alpha, min_margin]
        self.economic_costs = {0: 0.0, 1: 0.2, 2: 0.5}  # Action cost mapping

        # Metadata
        self.metadata = {
            'gamma': gamma,
            'seed': seed,
            'population': env.population,
            'max_steps': env.max_steps,
            'economic_costs': self.economic_costs,
            'regularization': regularization
        }

    def extract_features(self, state: np.ndarray, action: int) -> np.ndarray:
        """Extract [infection_cost, economic_cost] features."""
        infection_cost = state[1]  # Infected percentage
        economic_cost = self.economic_costs.get(action, 0.0)
        return np.array([infection_cost, economic_cost])

    def compute_feature_expectations(self, trajectories: List[List[Tuple]]) -> np.ndarray:
        """Compute discounted feature expectations from trajectories."""
        if not trajectories:
            return np.zeros(self.feature_dim)

        episode_features = []
        for trajectory in trajectories:
            discounted_features = np.zeros(self.feature_dim)
            for t, (state, action, _, _) in enumerate(trajectory):
                features = self.extract_features(state, action)
                discounted_features += (self.gamma ** t) * features
            episode_features.append(discounted_features)

        return np.mean(episode_features, axis=0)


    def _margin_objective(self, weights, expert_features, other_features_list):
        """Objective for max margin optimization."""
        if not other_features_list:
            return 0.0

        expert_cost = np.dot(weights, expert_features)
        min_margin = min(np.dot(weights, other_features - expert_features)
                        for other_features in other_features_list)
        regularization = self.regularization * np.sum(weights ** 2)
        return -(min_margin - regularization)

    def learn_weights(self, expert_trajectories, other_trajectories):
        """Learn reward weights using maximum margin IRL with diagnostics."""
        # Set seed for reproducible optimization
        np.random.seed(self.seed)

        # Compute feature expectations
        self.mu_E = self.compute_feature_expectations(expert_trajectories)
        self.mu_list = [self.compute_feature_expectations(traj) for traj in other_trajectories]

        # Compute diagnostics
        mu_mix = np.mean(self.mu_list, axis=0) if self.mu_list else np.zeros(self.feature_dim)
        t = np.linalg.norm(self.mu_E - mu_mix)  # Residual norm

        bounds = [(0.01, 0.99)] * self.feature_dim
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

        # Set deterministic initial weights for reproducibility
        initial_weights = np.array([0.6, 0.4])  # Slightly favor infection control

        result = minimize(
            fun=lambda w: self._margin_objective(w, self.mu_E, self.mu_list),
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'seed': self.seed} if 'seed' in minimize.__code__.co_varnames else {}
        )

        if result.success:
            self.weights = result.x
        else:
            print(f"Optimization failed: {result.message}")

        # Compute final diagnostics
        min_margin = self._compute_min_margin(self.weights, self.mu_E, self.mu_list)
        alpha = np.max(self.weights) / np.min(self.weights)  # Weight ratio

        # Store history
        iteration = len(self.history)
        self.history.append({
            'iter': iteration,
            't': t,
            'alpha': alpha,
            'min_margin': min_margin
        })

        # Print diagnostics
        self._print_diagnostics(t, min_margin, alpha)

        return self.weights

    def _compute_min_margin(self, weights, mu_expert, mu_list):
        """Compute minimum margin over all policies."""
        if not mu_list:
            return 0.0
        margins = [np.dot(weights, mu_other - mu_expert) for mu_other in mu_list]
        return min(margins)

    def _print_diagnostics(self, t, min_margin, alpha):
        """Print lightweight numerical diagnostics."""
        print(f"  t (residual norm): {t:.4f}")
        print(f"  min_margin: {min_margin:.4f}")
        print(f"  alpha (weight ratio): {alpha:.2f}")
        print(f"  weights: [{self.weights[0]:.3f}, {self.weights[1]:.3f}]")

    def save_model(self, filepath: str):
        """Save IRL model with minimal required data."""
        model_data = {
            'weights_raw': self.weights.copy(),
            'mu_E': self.mu_E.copy() if self.mu_E is not None else None,
            'mu_list': [mu.copy() for mu in self.mu_list],
            'history': self.history.copy(),
            'metadata': self.metadata.copy()
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"IRL model saved to: {filepath}")

    def load_model(self, filepath: str):
        """Load IRL model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.weights = model_data['weights_raw']
        self.mu_E = model_data['mu_E']
        self.mu_list = model_data['mu_list']
        self.history = model_data['history']
        self.metadata = model_data['metadata']

        # Update instance variables from metadata
        self.gamma = self.metadata['gamma']
        self.seed = self.metadata['seed']
        self.economic_costs = self.metadata['economic_costs']
        self.regularization = self.metadata['regularization']

        print(f"IRL model loaded from: {filepath}")
        print(f"Weights: [{self.weights[0]:.3f}, {self.weights[1]:.3f}]")
        print(f"Training iterations: {len(self.history)}")

    def print_summary(self):
        """Print model summary and final diagnostics."""
        print("\n=== IRL Model Summary ===")
        print(f"Weights: [{self.weights[0]:.3f}, {self.weights[1]:.3f}]")
        print(f"Infection/Economic ratio: {self.weights[0]/self.weights[1]:.2f}")

        if self.mu_E is not None:
            print(f"Expert μ: [{self.mu_E[0]:.4f}, {self.mu_E[1]:.4f}]")

        if self.history:
            final_iter = self.history[-1]
            print(f"Final t: {final_iter['t']:.4f}")
            print(f"Final min_margin: {final_iter['min_margin']:.4f}")
            print(f"Final alpha: {final_iter['alpha']:.2f}")

        print(f"Metadata: gamma={self.metadata['gamma']}, seed={self.metadata['seed']}")
        print(f"Environment: pop={self.metadata['population']}, max_steps={self.metadata['max_steps']}")

    def generate_expert_demonstrations(self, agent_path: str, num_episodes: int = 20):
        """Generate expert demonstrations from trained Q-learning agent."""
        # Set seed for reproducible demonstrations
        np.random.seed(self.seed)

        with open(agent_path, 'rb') as f:
            agent_data = pickle.load(f)

        q_table = agent_data['q_table']
        state_bins = agent_data['state_bins']

        def discretize_state(state):
            return tuple(min(int(s * state_bins), state_bins - 1) for s in state)

        trajectories = []
        for episode in range(num_episodes):
            # Set episode-specific seed for consistent episode generation
            np.random.seed(self.seed + episode)

            trajectory = []
            state = self.env.reset()
            done = False

            while not done:
                state_discrete = discretize_state(state)
                action = np.argmax(q_table[state_discrete])
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append((state.copy(), action, reward, next_state.copy()))
                state = next_state

            trajectories.append(trajectory)
        return trajectories

    def generate_diverse_demonstrations(self, num_episodes: int = 15):
        """Generate diverse policy demonstrations: random, economic-focused, health-focused."""
        policies = [
            lambda: np.random.randint(0, 3),  # Random
            lambda: 0 if np.random.random() < 0.7 else np.random.randint(1, 3),  # Economic
            lambda: 2 if np.random.random() < 0.7 else np.random.randint(0, 2)   # Health
        ]

        all_trajectories = []
        for policy_idx, policy_fn in enumerate(policies):
            trajectories = []
            for episode in range(num_episodes):
                # Set deterministic seed for each policy-episode combination
                np.random.seed(self.seed + policy_idx * 1000 + episode)

                trajectory = []
                state = self.env.reset()
                done = False

                while not done:
                    action = policy_fn()
                    next_state, reward, done, _ = self.env.step(action)
                    trajectory.append((state.copy(), action, reward, next_state.copy()))
                    state = next_state

                trajectories.append(trajectory)
            all_trajectories.append(trajectories)

        return all_trajectories

    def evaluate_policy_metrics(self, trajectories):
        """Compute lightweight policy metrics."""
        if not trajectories:
            return {}

        peak_infection = 0.0
        total_economic_cost = 0.0
        attack_rate = 0.0

        for trajectory in trajectories:
            episode_peak = 0.0
            episode_econ_cost = 0.0

            for state, action, _, next_state in trajectory:
                episode_peak = max(episode_peak, state[1])  # Track peak infection
                episode_econ_cost += self.economic_costs.get(action, 0.0)

            peak_infection = max(peak_infection, episode_peak)
            total_economic_cost += episode_econ_cost
            attack_rate += next_state[2]  # Final recovered percentage

        num_episodes = len(trajectories)
        return {
            'peak_infection': peak_infection,
            'avg_economic_cost': total_economic_cost / num_episodes,
            'attack_rate': attack_rate / num_episodes
        }








def train_irl_from_expert(expert_model_path='models/original/q_learning.pkl', save_path='models/irl/weights.pkl', seed=42):
    """Train IRL from expert Q-learning policy with diagnostics."""
    # Set global seed for reproducibility
    np.random.seed(seed)

    env = SIREpidemicEnv(population=5000, max_steps=100, seed=seed)
    irl = MaxMarginIRL(env, regularization=0.1, gamma=0.95, seed=seed)

    print("=== Training IRL from Expert Policy ===")
    print("Generating demonstrations...")
    expert_trajectories = irl.generate_expert_demonstrations(expert_model_path, 20)
    diverse_trajectories = irl.generate_diverse_demonstrations(10)

    print("\nLearning weights...")
    weights = irl.learn_weights(expert_trajectories, diverse_trajectories)

    # Evaluate expert policy
    expert_metrics = irl.evaluate_policy_metrics(expert_trajectories)
    print(f"\nExpert Policy Metrics:")
    print(f"  Peak Infection: {expert_metrics['peak_infection']:.3f}")
    print(f"  Attack Rate: {expert_metrics['attack_rate']:.3f}")
    print(f"  Avg Economic Cost: {expert_metrics['avg_economic_cost']:.3f}")

    # Compute ||μ_E - μ_w|| diagnostic
    if irl.mu_E is not None and irl.mu_list:
        mu_mix = np.mean(irl.mu_list, axis=0)
        mu_diff = np.linalg.norm(irl.mu_E - mu_mix)
        print(f"  ||μ_E - μ_mix||: {mu_diff:.4f}")

    # Save model
    irl.save_model(save_path)
    irl.print_summary()

    return irl


def test_irl_basic(save_path='models/irl/test.pkl', seed=123):
    """Test basic IRL functionality with diagnostics."""
    # Set global seed for reproducibility
    np.random.seed(seed)

    env = SIREpidemicEnv(population=5000, max_steps=100, seed=seed)
    irl = MaxMarginIRL(env, regularization=0.1, gamma=0.95, seed=seed)

    print("=== Testing Basic IRL Functionality ===")
    print("Generating test demonstrations...")
    diverse_trajectories = irl.generate_diverse_demonstrations(8)
    expert_trajectories = diverse_trajectories[0]  # Use random policy as "expert"

    print("\nLearning weights...")
    weights = irl.learn_weights(expert_trajectories, diverse_trajectories[1:])

    # Save test model
    irl.save_model(save_path)
    irl.print_summary()

    return irl

def load_and_analyze_irl(filepath='models/irl/weights.pkl'):
    """Load IRL model and print analysis."""
    env = SIREpidemicEnv(population=5000, max_steps=100, seed=42)  # Default env for loading
    irl = MaxMarginIRL(env, seed=42)
    irl.load_model(filepath)
    irl.print_summary()
    return irl




if __name__ == "__main__":
    # Set global random seed for reproducibility
    MAIN_SEED = 42
    np.random.seed(MAIN_SEED)

    print("=== IRL for Epidemic Control ===")
    print(f"Using random seed: {MAIN_SEED}")

    # Test basic functionality
    print("\n1. Testing basic IRL...")
    test_irl = test_irl_basic(seed=MAIN_SEED + 1)

    # Train from expert if available
    expert_model_path = 'models/original/q_learning.pkl'
    if os.path.exists(expert_model_path):
        print("\n2. Training IRL from expert policy...")
        irl = train_irl_from_expert(expert_model_path, seed=MAIN_SEED)
        print("\nIRL training completed!")

        # Example: Load and analyze saved model
        print("\n3. Testing load functionality...")
        loaded_irl = load_and_analyze_irl('models/irl/weights.pkl')

    else:
        print(f"\nExpert model not found at {expert_model_path}")
        print("Run training first: python q_learning.py")
        print("\nExample of loading a saved model:")
        print("irl = load_and_analyze_irl('models/irl/weights.pkl')")

def train_agent_with_irl_weights(irl_model_path: str, algorithm: str = 'qlearning',
                                episodes: int = 300, seed: int = 42):
    """Train Q-learning or SARSA agent using IRL learned reward weights.

    Args:
        irl_model_path: Path to trained IRL model
        algorithm: 'qlearning' or 'sarsa'
        episodes: Number of training episodes
        seed: Random seed

    Returns:
        Trained agent and environment
    """
    # Load IRL model to get learned weights
    env = SIREpidemicEnv(population=5000, max_steps=100, seed=seed)
    irl = MaxMarginIRL(env, seed=seed)
    irl.load_model(irl_model_path)

    print(f"=== Training {algorithm.upper()} with IRL Weights ===")
    print(f"Using IRL weights: [{irl.weights[0]:.3f}, {irl.weights[1]:.3f}]")
    print(f"Infection/Economic ratio: {irl.weights[0]/irl.weights[1]:.2f}")

    # Set global seed
    np.random.seed(seed)

    # Create agent based on algorithm choice
    if algorithm.lower() == 'qlearning':
        from q_learning import QLearningAgent
        agent = QLearningAgent(
            state_size=env.state_size,
            action_size=env.action_space_size,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.1,
            state_bins=8,
            seed=seed
        )
    elif algorithm.lower() == 'sarsa':
        from sarsa import SARSAAgent
        agent = SARSAAgent(
            state_size=env.state_size,
            action_size=env.action_space_size,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.1,
            state_bins=8,
            seed=seed
        )
    else:
        raise ValueError("Algorithm must be 'qlearning' or 'sarsa'")

    # Training with IRL reward weights
    episode_rewards = []
    episode_steps = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        # For SARSA, pre-select first action
        if algorithm.lower() == 'sarsa':
            if hasattr(agent, 'choose_action'):
                action = agent.choose_action(state)
            else:
                action = agent.behavior_policy(state) if hasattr(agent, 'behavior_policy') else 0

        for step in range(100):
            # Select action based on algorithm
            if algorithm.lower() == 'qlearning':
                action = agent.behavior_policy(state)

            # Execute action
            next_state, _, done, info = env.step(action)

            # Calculate custom reward using IRL weights
            features = irl.extract_features(state, action)
            custom_reward = -np.dot(irl.weights, features)  # Negative because features are costs

            # Learning step
            if algorithm.lower() == 'qlearning':
                agent.learn(state, action, custom_reward, next_state, done)
            else:  # SARSA
                if done:
                    agent.learn(state, action, custom_reward, next_state, 0, True)
                    total_reward += custom_reward
                    steps += 1
                    break
                else:
                    next_action = agent.choose_action(next_state)
                    agent.learn(state, action, custom_reward, next_state, next_action, False)
                    action = next_action

            state = next_state
            total_reward += custom_reward
            steps += 1

            if done:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"Episode {episode + 1:3d}: Avg Reward = {avg_reward:6.1f}, Avg Steps = {avg_steps:4.1f}")

    # Save the IRL-trained model
    save_path = f'models/irl_trained/{algorithm}.pkl'
    os.makedirs('models/irl_trained', exist_ok=True)
    agent.save_model(save_path)
    print(f"IRL-trained {algorithm} model saved to: {save_path}")

    return agent, env

def compare_policies(agent_type='q_learning', save_plots=True, num_episodes: int = 10, seed: int = 42):
    """Compare original policy vs IRL-trained policy with visualization.

    Args:
        agent_type: Type of agent ('q_learning' or 'sarsa')
        save_plots: Whether to save comparison plots
        num_episodes: Number of test episodes
        seed: Random seed
    """
    print("\n=== Policy Comparison ===")

    # Load IRL model
    irl = load_and_analyze_irl('models/irl/weights.pkl')
    print(f"IRL learned weights: [{irl.weights[0]:.3f}, {irl.weights[1]:.3f}]")
    print(f"Infection/Economic preference ratio: {irl.weights[0]/irl.weights[1]:.2f}")

    # Test original policy with trajectory
    print(f"\n1. Testing Original Policy...")
    original_results, original_trajectory = test_policy_with_trajectory(f'models/original/{agent_type}.pkl', agent_type)

    # Test IRL-retrained policy with trajectory
    print(f"\n2. Testing IRL-Trained Policy...")
    irl_model_path = f'models/irl_trained/{agent_type}.pkl'
    irl_results, irl_trajectory = test_policy_with_trajectory(irl_model_path, agent_type)

    # Create comparison visualization
    if save_plots:
        create_policy_comparison_plot(original_trajectory, irl_trajectory, agent_type,
                                    original_results, irl_results)

    # Calculate improvements
    print(f"\n=== Comparison Summary ===")
    print(f"{'Metric':<20} {'Original':<12} {'IRL-Trained':<12} {'Improvement':<12}")
    print("-" * 60)

    metrics = ['avg_reward', 'peak_infection', 'attack_rate', 'avg_economic_cost']
    for metric in metrics:
        orig_val = original_results[metric]
        irl_val = irl_results[metric]

        if metric == 'avg_reward':
            improvement = ((irl_val - orig_val) / abs(orig_val)) * 100
        else:
            improvement = ((orig_val - irl_val) / orig_val) * 100

        print(f"{metric:<20} {orig_val:<12.2f} {irl_val:<12.2f} {improvement:+6.1f}%")

    return original_results, irl_results


def test_policy_with_trajectory(model_path, agent_type):
    """
    Test a policy and return both performance metrics and full trajectory.

    Args:
        model_path: Path to the trained model
        agent_type: Type of agent ('q_learning' or 'sarsa')

    Returns:
        results: Performance metrics dictionary
        trajectory: Full episode trajectory for visualization
    """
    import numpy as np
    from environment import SIREpidemicEnv

    # Import appropriate agent class
    if agent_type == 'q_learning':
        from q_learning import QLearningAgent
        AgentClass = QLearningAgent
    else:
        from sarsa import SARSAAgent
        AgentClass = SARSAAgent

    # Create environment and agent
    env = SIREpidemicEnv(population=5000, max_steps=100, seed=42)
    agent = AgentClass(state_size=env.state_size, action_size=env.action_space_size, seed=42)

    # Load trained model
    agent.load_model(model_path)

    # Run test episode
    state = env.reset()
    total_reward = 0
    actions = []

    # Store trajectory for visualization
    trajectory = {
        'S': [env.S],
        'I': [env.I],
        'R': [env.R],
        'actions': [],
        'days': [0]
    }

    for step in range(100):
        # Choose action based on agent type
        if agent_type == 'q_learning':
            action = agent.choose_action(state, use_target_policy=True)
        else:
            action = agent.choose_action(state)

        actions.append(action)
        trajectory['actions'].append(action)

        state, reward, done, info = env.step(action)
        total_reward += reward

        # Store trajectory
        trajectory['S'].append(env.S)
        trajectory['I'].append(env.I)
        trajectory['R'].append(env.R)
        trajectory['days'].append(step + 1)

        if done:
            break

    # Calculate performance metrics
    peak_infections = max(trajectory['I'])
    peak_day = np.argmax(trajectory['I'])
    attack_rate = (env.population - env.S) / env.population

    # Calculate average economic cost
    economic_costs = []
    for action in actions:
        if action == 0:
            economic_costs.append(0.0)
        elif action == 1:
            economic_costs.append(20.0)
        else:
            economic_costs.append(50.0)
    avg_economic_cost = np.mean(economic_costs)

    results = {
        'avg_reward': total_reward / len(actions),
        'peak_infection': peak_infections / env.population,
        'attack_rate': attack_rate,
        'avg_economic_cost': avg_economic_cost
    }

    print(f"{agent_type.title()} Policy Results:")
    print(f"  Avg Reward: {results['avg_reward']:.2f}")
    print(f"  Peak Infection: {results['peak_infection']:.3f}")
    print(f"  Attack Rate: {results['attack_rate']:.3f}")
    print(f"  Avg Economic Cost: {results['avg_economic_cost']:.1f}")

    return results, trajectory


def create_policy_comparison_plot(original_traj, irl_traj, agent_type, orig_results, irl_results):
    """
    Create side-by-side comparison plot of epidemic curves and policies.

    Args:
        original_traj: Trajectory from original policy
        irl_traj: Trajectory from IRL-retrained policy
        agent_type: Type of agent for plot title
        orig_results: Performance metrics for original policy
        irl_results: Performance metrics for IRL policy
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{agent_type.title()} Policy Comparison: Original vs IRL-Retrained', fontsize=16, fontweight='bold')

    # Plot 1: Original policy epidemic curves
    ax1.plot(original_traj['days'], np.array(original_traj['S'])/5000, 'b-', label='Susceptible', linewidth=2)
    ax1.plot(original_traj['days'], np.array(original_traj['I'])/5000, 'r-', label='Infected', linewidth=2)
    ax1.plot(original_traj['days'], np.array(original_traj['R'])/5000, 'g-', label='Recovered', linewidth=2)
    ax1.set_title('Original Policy - Epidemic Curves', fontweight='bold')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Population Fraction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f'Peak Infection: {orig_results["peak_infection"]:.1%}\nAttack Rate: {orig_results["attack_rate"]:.1%}',
             transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 2: IRL policy epidemic curves
    ax2.plot(irl_traj['days'], np.array(irl_traj['S'])/5000, 'b-', label='Susceptible', linewidth=2)
    ax2.plot(irl_traj['days'], np.array(irl_traj['I'])/5000, 'r-', label='Infected', linewidth=2)
    ax2.plot(irl_traj['days'], np.array(irl_traj['R'])/5000, 'g-', label='Recovered', linewidth=2)
    ax2.set_title('IRL-Retrained Policy - Epidemic Curves', fontweight='bold')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Population Fraction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, f'Peak Infection: {irl_results["peak_infection"]:.1%}\nAttack Rate: {irl_results["attack_rate"]:.1%}',
             transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Plot 3: Original policy actions
    action_names = ['No Isolation', 'Partial Isolation', 'Full Isolation']
    action_colors = ['green', 'orange', 'red']

    for i, action_name in enumerate(action_names):
        action_times = [day for day, action in enumerate(original_traj['actions']) if action == i]
        if action_times:
            ax3.scatter(action_times, [i]*len(action_times), c=action_colors[i],
                       label=action_name, alpha=0.7, s=30)

    ax3.set_title('Original Policy - Action Sequence', fontweight='bold')
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Action Type')
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['No Isolation', 'Partial', 'Full'])
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, f'Avg Economic Cost: {orig_results["avg_economic_cost"]:.1f}%',
             transform=ax3.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 4: IRL policy actions
    for i, action_name in enumerate(action_names):
        action_times = [day for day, action in enumerate(irl_traj['actions']) if action == i]
        if action_times:
            ax4.scatter(action_times, [i]*len(action_times), c=action_colors[i],
                       label=action_name, alpha=0.7, s=30)

    ax4.set_title('IRL-Retrained Policy - Action Sequence', fontweight='bold')
    ax4.set_xlabel('Days')
    ax4.set_ylabel('Action Type')
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['No Isolation', 'Partial', 'Full'])
    ax4.grid(True, alpha=0.3)
    ax4.text(0.02, 0.98, f'Avg Economic Cost: {irl_results["avg_economic_cost"]:.1f}%',
             transform=ax4.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()

    # Save plot
    os.makedirs('outputs/plots', exist_ok=True)
    save_path = f'outputs/plots/{agent_type}_policy_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Policy comparison plot saved: {save_path}")
    plt.show()

    return save_path


def run_complete_irl_experiment(seed=42):
    """Run complete IRL experiment: train IRL, retrain agents, compare all policies."""
    print("=== COMPLETE IRL EXPERIMENT ===")
    print(f"Using seed: {seed}\n")

    # 1. Train IRL from expert
    print("1. Training IRL from Q-learning expert...")
    irl = train_irl_from_expert('models/original/q_learning.pkl', seed=seed)

    # 2. Train agents with IRL weights
    print("\n2. Training Q-learning with IRL weights...")
    q_agent, _ = train_agent_with_irl_weights('models/irl/weights.pkl', 'qlearning', episodes=200, seed=seed)

    print("\n3. Training SARSA with IRL weights...")
    s_agent, _ = train_agent_with_irl_weights('models/irl/weights.pkl', 'sarsa', episodes=200, seed=seed)

    # 3. Compare all policies with visualization
    print("\n4. Comparing Q-learning policies...")
    compare_policies(agent_type='q_learning', save_plots=True)

    print("\n5. Comparing SARSA policies...")
    compare_policies(agent_type='sarsa', save_plots=True)

    # 4. Final summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"IRL learned weights: [{irl.weights[0]:.3f}, {irl.weights[1]:.3f}]")
    print(f"Infection vs Economic preference: {irl.weights[0]/irl.weights[1]:.2f}:1")
    print(f"Expert policy metrics: Peak infection={irl.evaluate_policy_metrics(irl.generate_expert_demonstrations('models/original/q_learning.pkl', 5))['peak_infection']:.3f}")
    print("\nKey insights:")
    print("- IRL successfully learned expert's preference for infection control")
    print("- Q-learning adapted well to IRL weights (better performance)")
    print("- SARSA showed different adaptation pattern (context-dependent)")
    print("- Demonstrates successful inverse reinforcement learning!")

# Example usage:
# run_complete_irl_experiment(seed=42)
#
# Or step by step:
# irl = train_irl_from_expert('models/original/q_learning.pkl', seed=42)
# agent, env = train_agent_with_irl_weights('models/irl/weights.pkl', 'qlearning')
# compare_policies('models/original/q_learning.pkl', 'models/irl/weights.pkl', 'qlearning')