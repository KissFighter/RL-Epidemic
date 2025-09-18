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
        self.env.seed(seed) if hasattr(self.env, 'seed') else None

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








def train_irl_from_expert(expert_model_path='training_results/simple_q_model.pkl', save_path='training_results/irl_model.pkl', seed=42):
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


def test_irl_basic(save_path='training_results/irl_test.pkl', seed=123):
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

def load_and_analyze_irl(filepath):
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
    expert_model_path = 'training_results/simple_q_model.pkl'
    if os.path.exists(expert_model_path):
        print("\n2. Training IRL from expert policy...")
        irl = train_irl_from_expert(expert_model_path, seed=MAIN_SEED)
        print("\nIRL training completed!")

        # Example: Load and analyze saved model
        print("\n3. Testing load functionality...")
        loaded_irl = load_and_analyze_irl('training_results/irl_model.pkl')

    else:
        print(f"\nExpert model not found at {expert_model_path}")
        print("Run training first: python simple_train.py")
        print("\nExample of loading a saved model:")
        print("irl = load_and_analyze_irl('training_results/irl_model.pkl')")

# Example usage with specific seeds:
# irl = train_irl_from_expert('training_results/simple_q_model.pkl', seed=42)
# test_irl = test_irl_basic(seed=123)
# loaded_irl = load_and_analyze_irl('training_results/irl_model.pkl')