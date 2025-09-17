#!/usr/bin/env python3
"""
Inverse Reinforcement Learning (IRL) Module for Epidemic Control

This module implements Maximum Margin IRL to learn the optimal weights
between infection control and economic cost in epidemic management.

The learned reward function is a linear combination:
R(s,a) = w1 * infection_penalty + w2 * economic_penalty

Key features:
- Non-invasive design: doesn't modify existing code
- Maximum Margin IRL algorithm
- Feature extraction for infection and economic factors
- Expert demonstration generation from trained policies
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from typing import List, Tuple, Dict, Any
from scipy.optimize import minimize
from environment import SIREpidemicEnv
from q_learning import QLearningAgent


class MaxMarginIRL:
    """
    Maximum Margin Inverse Reinforcement Learning implementation.

    Learns reward function weights by maximizing the margin between
    expert policy value and other policies' values.
    """

    def __init__(self,
                 env: SIREpidemicEnv,
                 feature_dim: int = 2,
                 regularization: float = 1.0):
        """
        Initialize Maximum Margin IRL.

        Args:
            env: SIR epidemic environment
            feature_dim: Number of features (2: infection + economic)
            regularization: L2 regularization strength
        """
        self.env = env
        self.feature_dim = feature_dim
        self.regularization = regularization
        self.weights = np.ones(feature_dim) / feature_dim  # Initialize uniform weights (standardized space)
        self.weights_raw = np.ones(feature_dim) / feature_dim  # Initialize uniform weights (raw space)

    def extract_features(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Extract feature vector from state-action pair.

        Features are positive costs (no scaling factors):
        - Infection cost: infected population ratio [0,1]
        - Economic cost: economic cost of action [0,0.5]

        Args:
            state: Environment state [S%, I%, R%]
            action: Action taken {0, 1, 2}

        Returns:
            Feature vector [infection_cost, economic_cost]
        """
        # Feature 1: Infection cost (positive, normalized)
        infection_cost = float(state[1])  # I% (infected percentage)

        # Feature 2: Economic cost (positive, from action mapping)
        economic_costs = {0: 0.0, 1: 0.2, 2: 0.5}  # From environment action mapping
        economic_cost = economic_costs.get(action, 0.0)

        return np.array([infection_cost, economic_cost], dtype=np.float64)

    def compute_feature_expectations(self, trajectories: List[List[Tuple]], gamma: float = 0.99) -> np.ndarray:
        """
        Compute discounted feature expectations from trajectories.

        Args:
            trajectories: List of trajectories, each containing (state, action, reward, next_state)
            gamma: Discount factor for temporal weighting

        Returns:
            Feature expectations vector (averaged across episodes)
        """
        episode_features = []

        for trajectory in trajectories:
            discounted_features = np.zeros(self.feature_dim, dtype=np.float64)
            for t, (state, action, _, _) in enumerate(trajectory):
                features = self.extract_features(state, action)
                discounted_features += (gamma ** t) * features
            episode_features.append(discounted_features)

        return np.mean(episode_features, axis=0) if episode_features else np.zeros(self.feature_dim, dtype=np.float64)

    def compute_policy_value(self, agent: QLearningAgent, num_episodes: int = 10) -> float:
        """
        Compute expected value of a policy over multiple episodes.

        Args:
            agent: Trained agent whose policy to evaluate
            num_episodes: Number of episodes for evaluation

        Returns:
            Average episode return
        """
        total_return = 0.0

        for _ in range(num_episodes):
            state = self.env.reset()
            episode_return = 0.0
            done = False

            while not done:
                # Use agent's greedy policy (no exploration during evaluation)
                state_discrete = agent.discretize_state(state)
                action = np.argmax(agent.q_table[state_discrete])

                # Compute reward using current weights (negative of cost)
                features = self.extract_features(state, action)
                reward = -np.dot(self.weights_raw, features)

                state, _, done, _ = self.env.step(action)
                episode_return += reward

            total_return += episode_return

        return total_return / num_episodes

    def margin_objective(self, weights: np.ndarray,
                        expert_features: np.ndarray,
                        other_policies_features: List[np.ndarray]) -> float:
        """
        Objective function for Maximum Margin IRL.

        Maximize: min_policy (other_cost - expert_cost) - λ||w||²
        Since features are costs, we want other policies to have higher costs.

        Args:
            weights: Current weight vector
            expert_features: Expert policy feature expectations
            other_policies_features: List of other policies' feature expectations

        Returns:
            Negative margin (for minimization)
        """
        expert_cost = np.dot(weights, expert_features)  # Expert's cost

        # Find minimum margin over all other policies
        min_margin = float('inf')
        for other_features in other_policies_features:
            other_cost = np.dot(weights, other_features)
            margin = other_cost - expert_cost  # Other - Expert (want this > 0)
            min_margin = min(min_margin, margin)

        # If no other policies, set neutral baseline
        if min_margin == float('inf'):
            min_margin = 0.0

        # Add L2 regularization
        regularization_term = self.regularization * np.sum(weights ** 2)

        # Return negative (since we're minimizing)
        return -(min_margin - regularization_term)

    def learn_weights(self,
                     expert_trajectories: List[List[Tuple]],
                     other_trajectories: List[List[List[Tuple]]]) -> np.ndarray:
        """
        Learn reward function weights using Maximum Margin IRL.

        Args:
            expert_trajectories: Expert demonstration trajectories
            other_trajectories: List of other policies' trajectories

        Returns:
            Learned weight vector
        """
        # Compute feature expectations
        expert_features = self.compute_feature_expectations(expert_trajectories)
        other_features = [self.compute_feature_expectations(traj)
                         for traj in other_trajectories]

        # Feature standardization for better numerical stability
        all_features = np.vstack([expert_features] + other_features)
        feature_mean = all_features.mean(axis=0)
        feature_std = all_features.std(axis=0) + 1e-8

        expert_features_std = (expert_features - feature_mean) / feature_std
        other_features_std = [(f - feature_mean) / feature_std for f in other_features]

        # Optimization constraints and bounds
        bounds = [(1e-3, 1.0)] * self.feature_dim  # Prevent corner solutions
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # sum = 1
            {'type': 'ineq', 'fun': lambda w: w}  # w >= 0
        ]

        # Initial guess
        initial_weights = np.ones(self.feature_dim) / self.feature_dim

        # Optimize with standardized features
        result = minimize(
            fun=lambda w: self.margin_objective(w, expert_features_std, other_features_std),
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )

        if result.success:
            self.weights = result.x  # Weights in standardized space
            # Restore to original feature space
            self._feat_mean = feature_mean
            self._feat_std = feature_std
            self.weights_raw = self.weights / self._feat_std
            # Normalize to sum to 1
            self.weights_raw = self.weights_raw / np.sum(self.weights_raw)
        else:
            print(f"Optimization failed: {result.message}")
            # Fallback to uniform weights
            self.weights_raw = np.ones(self.feature_dim) / self.feature_dim

        return self.weights_raw

    def generate_expert_demonstrations(self,
                                     agent_path: str,
                                     num_episodes: int = 20) -> List[List[Tuple]]:
        """
        Generate expert demonstrations using a trained Q-learning agent.

        Args:
            agent_path: Path to saved Q-learning model
            num_episodes: Number of demonstration episodes

        Returns:
            List of expert trajectories
        """
        # Load trained agent data
        with open(agent_path, 'rb') as f:
            agent_data = pickle.load(f)

        # Extract Q-table and parameters
        q_table = agent_data['q_table']
        state_bins = agent_data['state_bins']

        def discretize_state(state):
            """Discretize state for Q-table lookup"""
            discretized = []
            for s in state:
                bin_idx = min(int(s * state_bins), state_bins - 1)
                discretized.append(bin_idx)
            return tuple(discretized)

        trajectories = []

        for _ in range(num_episodes):
            trajectory = []
            state = self.env.reset()
            done = False

            while not done:
                # Use greedy policy from Q-table
                state_discrete = discretize_state(state)
                action = np.argmax(q_table[state_discrete])

                next_state, reward, done, _ = self.env.step(action)
                trajectory.append((state.copy(), action, reward, next_state.copy()))
                state = next_state

            trajectories.append(trajectory)

        return trajectories

    def generate_diverse_demonstrations(self, num_episodes: int = 20) -> List[List[List[Tuple]]]:
        """
        Generate diverse policy demonstrations for better comparison.

        Returns trajectories from:
        1. Random policy
        2. Economic-focused policy (prefers action 0)
        3. Health-focused policy (prefers action 2)

        Args:
            num_episodes: Number of episodes per policy

        Returns:
            List of trajectory lists for different policies
        """
        all_policies_trajectories = []

        # Policy 1: Random
        random_trajectories = []
        for _ in range(num_episodes):
            trajectory = []
            state = self.env.reset()
            done = False
            while not done:
                action = np.random.randint(0, self.env.action_space_size)
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append((state.copy(), action, reward, next_state.copy()))
                state = next_state
            random_trajectories.append(trajectory)
        all_policies_trajectories.append(random_trajectories)

        # Policy 2: Economic-focused (prefers no isolation)
        economic_trajectories = []
        for _ in range(num_episodes):
            trajectory = []
            state = self.env.reset()
            done = False
            while not done:
                # 70% chance action 0, 30% chance others
                if np.random.random() < 0.7:
                    action = 0
                else:
                    action = np.random.randint(1, self.env.action_space_size)
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append((state.copy(), action, reward, next_state.copy()))
                state = next_state
            economic_trajectories.append(trajectory)
        all_policies_trajectories.append(economic_trajectories)

        # Policy 3: Health-focused (prefers full isolation)
        health_trajectories = []
        for _ in range(num_episodes):
            trajectory = []
            state = self.env.reset()
            done = False
            while not done:
                # 70% chance action 2, 30% chance others
                if np.random.random() < 0.7:
                    action = 2
                else:
                    action = np.random.randint(0, 2)
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append((state.copy(), action, reward, next_state.copy()))
                state = next_state
            health_trajectories.append(trajectory)
        all_policies_trajectories.append(health_trajectories)

        return all_policies_trajectories

    def train_policy_with_custom_weights(self, weights: np.ndarray,
                                       episodes: int = 200,
                                       max_steps: int = 100,
                                       seed: int = 123) -> 'QLearningAgent':
        """
        Train a Q-learning policy using custom reward weights.

        Args:
            weights: Reward weights [infection_weight, economic_weight]
            episodes: Number of training episodes
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility

        Returns:
            Trained Q-learning agent
        """
        np.random.seed(seed)

        # Create a fresh Q-learning agent
        agent = QLearningAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_space_size,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.1,
            state_bins=8
        )

        for episode in range(episodes):
            state = self.env.reset()
            done = False
            step = 0

            while not done and step < max_steps:
                # Choose action using behavior policy
                action = agent.behavior_policy(state)

                # Take environment step
                next_state, _, done, info = self.env.step(action)

                # Calculate custom reward using learned weights
                features = self.extract_features(state, action)
                custom_reward = -np.dot(weights, features)  # Negative because features are costs

                # Learn with custom reward
                agent.learn(state, action, custom_reward, next_state, done)

                state = next_state
                step += 1

        return agent

    def generate_expert_by_planning(self, weights: np.ndarray,
                                  num_episodes: int = 20,
                                  max_steps: int = 100) -> List[List[Tuple]]:
        """
        Generate expert trajectories using Q-learning with given reward weights.

        Args:
            weights: True reward weights to use for planning
            num_episodes: Number of demonstration episodes
            max_steps: Maximum steps per episode

        Returns:
            List of expert trajectories from planning
        """
        # Train agent with true weights
        agent = self.train_policy_with_custom_weights(weights)

        # Generate trajectories using trained policy
        trajectories = []
        for _ in range(num_episodes):
            trajectory = []
            state = self.env.reset()
            done = False
            step = 0

            while not done and step < max_steps:
                # Use greedy policy (target policy)
                state_discrete = agent._discretize_state(state)
                action = np.argmax(agent.q_table[state_discrete])

                next_state, reward, done, _ = self.env.step(action)
                trajectory.append((state.copy(), action, reward, next_state.copy()))

                state = next_state
                step += 1

            trajectories.append(trajectory)

        return trajectories

    def iterate_max_margin(self, expert_trajectories: List[List[Tuple]],
                         initial_adversaries: List[List[List[Tuple]]],
                         max_iterations: int = 4) -> Tuple[np.ndarray, List]:
        """
        Iterative Max-Margin IRL with adversarial policy generation.

        Args:
            expert_trajectories: Expert demonstration trajectories
            initial_adversaries: Initial set of adversarial policies
            max_iterations: Number of outer loop iterations

        Returns:
            Final learned weights and adversary set
        """
        adversaries = initial_adversaries[:]
        weights_history = []

        print(f"Starting iterative Max-Margin IRL ({max_iterations} iterations)...")

        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

            # Learn weights with current adversary set
            weights = self.learn_weights(expert_trajectories, adversaries)
            weights_history.append(weights.copy())

            print(f"Learned weights: [{weights[0]:.3f}, {weights[1]:.3f}]")

            # Train new adversarial policy using current weights
            print("Training new adversarial policy...")
            adversarial_agent = self.train_policy_with_custom_weights(
                weights, episodes=150, seed=iteration * 42
            )

            # Generate trajectories from new adversarial policy
            new_adversarial_trajectories = []
            for _ in range(10):
                trajectory = []
                state = self.env.reset()
                done = False
                step = 0

                while not done and step < 100:
                    # Use greedy policy
                    state_discrete = adversarial_agent._discretize_state(state)
                    action = np.argmax(adversarial_agent.q_table[state_discrete])

                    next_state, reward, done, _ = self.env.step(action)
                    trajectory.append((state.copy(), action, reward, next_state.copy()))

                    state = next_state
                    step += 1

                new_adversarial_trajectories.append(trajectory)

            # Add new adversary to set
            adversaries.append(new_adversarial_trajectories)
            print(f"Added new adversarial policy (total adversaries: {len(adversaries)})")

            # Check convergence
            if len(weights_history) >= 2:
                weight_change = np.linalg.norm(weights_history[-1] - weights_history[-2])
                print(f"Weight change: {weight_change:.6f}")
                if weight_change < 0.01:
                    print("Converged!")
                    break

        return weights, adversaries

    def plot_feature_space(self, expert_trajectories: List[List[Tuple]],
                          adversarial_trajectories: List[List[List[Tuple]]],
                          weights: np.ndarray, title: str = "Feature Expectation Space"):
        """
        Visualize feature expectations in 2D space with learned weights.

        Args:
            expert_trajectories: Expert demonstration trajectories
            adversarial_trajectories: List of adversarial policies' trajectories
            weights: Learned weight vector
            title: Plot title
        """
        # Compute feature expectations
        mu_expert = self.compute_feature_expectations(expert_trajectories)
        mu_adversaries = [self.compute_feature_expectations(traj) for traj in adversarial_trajectories]

        # Create visualization
        plt.figure(figsize=(8, 6))

        # Plot adversarial policies
        adv_xs = [mu[0] for mu in mu_adversaries]
        adv_ys = [mu[1] for mu in mu_adversaries]
        plt.scatter(adv_xs, adv_ys, c='lightblue', s=60, alpha=0.7, label='Adversarial Policies')

        # Plot expert policy
        plt.scatter([mu_expert[0]], [mu_expert[1]], c='red', s=100, label='Expert Policy', zorder=3)

        # Draw weight vector pointing toward better (lower cost) direction
        x_range = max(adv_xs + [mu_expert[0]]) - min(adv_xs + [mu_expert[0]])
        y_range = max(adv_ys + [mu_expert[1]]) - min(adv_ys + [mu_expert[1]])
        scale = 0.15 * max(x_range, y_range)

        # Arrow points toward lower cost direction (negative gradient)
        plt.arrow(mu_expert[0], mu_expert[1], -weights[0]*scale, -weights[1]*scale,
                 head_width=0.02*scale, head_length=0.03*scale, fc='red', ec='red',
                 linewidth=2, length_includes_head=True, label=f'Preference Direction (w=[{weights[0]:.3f}, {weights[1]:.3f}])')

        # Draw decision boundary through expert point
        xs = np.linspace(min(adv_xs + [mu_expert[0]])-0.5, max(adv_xs + [mu_expert[0]])+0.5, 100)
        if abs(weights[1]) > 1e-8:
            ys = (-weights[0]*(xs - mu_expert[0]) / weights[1]) + mu_expert[1]
            plt.plot(xs, ys, 'r--', alpha=0.7, label='Decision Boundary')

        plt.xlabel('Infection Cost (discounted sum)')
        plt.ylabel('Economic Cost (discounted sum)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Print numerical info
        print(f"\nFeature Space Analysis:")
        print(f"Expert μ: [{mu_expert[0]:.4f}, {mu_expert[1]:.4f}]")
        for i, mu_adv in enumerate(mu_adversaries):
            margin = np.dot(weights, mu_adv - mu_expert)  # Other - Expert (positive is good)
            print(f"Adversary {i+1} μ: [{mu_adv[0]:.4f}, {mu_adv[1]:.4f}], Margin (other-expert): {margin:.4f}")

        plt.tight_layout()
        plt.show()

    def evaluate_policy_performance(self, trajectories: List[List[Tuple]],
                                  policy_name: str = "Policy") -> Dict[str, float]:
        """
        Evaluate policy performance metrics.

        Args:
            trajectories: Policy trajectories
            policy_name: Name for logging

        Returns:
            Performance metrics dictionary
        """
        metrics = {
            'peak_infection': 0.0,
            'attack_rate': 0.0,
            'total_economic_cost': 0.0,
            'total_episodes': len(trajectories)
        }

        for trajectory in trajectories:
            episode_peak = 0.0
            episode_econ_cost = 0.0
            final_recovered = 0.0

            for state, action, _, next_state in trajectory:
                # Track peak infection in this episode
                infection_rate = state[1]  # I%
                episode_peak = max(episode_peak, infection_rate)

                # Accumulate economic cost
                economic_costs = {0: 0.0, 1: 0.2, 2: 0.5}
                episode_econ_cost += economic_costs.get(action, 0.0)

                # Final state recovered percentage
                final_recovered = next_state[2]  # R%

            metrics['peak_infection'] = max(metrics['peak_infection'], episode_peak)
            metrics['attack_rate'] += final_recovered  # Average across episodes
            metrics['total_economic_cost'] += episode_econ_cost

        # Average metrics
        if metrics['total_episodes'] > 0:
            metrics['attack_rate'] /= metrics['total_episodes']
            metrics['avg_economic_cost'] = metrics['total_economic_cost'] / metrics['total_episodes']

        print(f"\n{policy_name} Performance:")
        print(f"  Peak Infection: {metrics['peak_infection']:.3f}")
        print(f"  Attack Rate: {metrics['attack_rate']:.3f}")
        print(f"  Avg Economic Cost: {metrics['avg_economic_cost']:.3f}")

        return metrics

    def comprehensive_analysis(self, expert_trajectories: List[List[Tuple]],
                             adversarial_trajectories: List[List[List[Tuple]]],
                             weights: np.ndarray):
        """
        Comprehensive analysis of IRL results with visualizations and metrics.

        Args:
            expert_trajectories: Expert demonstration trajectories
            adversarial_trajectories: List of adversarial policies' trajectories
            weights: Final learned weights
        """
        print("=" * 60)
        print("COMPREHENSIVE IRL ANALYSIS")
        print("=" * 60)

        # 1. Feature space visualization
        self.plot_feature_space(expert_trajectories, adversarial_trajectories, weights)

        # 2. Policy performance comparison
        print("\n" + "=" * 40)
        print("POLICY PERFORMANCE COMPARISON")
        print("=" * 40)

        expert_metrics = self.evaluate_policy_performance(expert_trajectories, "Expert Policy")

        policy_names = ["Random", "Economic-focused", "Health-focused"]
        for i, adv_traj in enumerate(adversarial_trajectories[:3]):  # First 3 are diverse policies
            if i < len(policy_names):
                self.evaluate_policy_performance(adv_traj, policy_names[i] + " Policy")

        # 3. Weight sensitivity analysis
        print("\n" + "=" * 40)
        print("WEIGHT SENSITIVITY ANALYSIS")
        print("=" * 40)

        test_weights = [
            np.array([0.9, 0.1]),  # Health-focused
            np.array([0.7, 0.3]),  # Moderate health focus
            np.array([0.5, 0.5]),  # Balanced
            np.array([0.3, 0.7]),  # Moderate economic focus
            np.array([0.1, 0.9])   # Economic-focused
        ]

        print("Testing different weight combinations:")
        for i, test_w in enumerate(test_weights):
            expert_val = np.dot(test_w, self.compute_feature_expectations(expert_trajectories))
            print(f"  Weights [{test_w[0]:.1f}, {test_w[1]:.1f}]: Expert value = {expert_val:.4f}")


def train_irl_from_expert(expert_model_path: str = 'results/q_learning_model.pkl',
                         num_expert_demos: int = 20,
                         num_random_demos: int = 20) -> MaxMarginIRL:
    """
    Complete IRL training pipeline using expert Q-learning policy.

    Args:
        expert_model_path: Path to trained Q-learning model
        num_expert_demos: Number of expert demonstration episodes
        num_random_demos: Number of random comparison episodes

    Returns:
        Trained IRL model with learned weights
    """
    # Create environment
    env = SIREpidemicEnv(population=5000, max_steps=100)

    # Initialize IRL
    irl = MaxMarginIRL(env, feature_dim=2, regularization=0.1)

    print("Generating expert demonstrations...")
    expert_trajectories = irl.generate_expert_demonstrations(
        expert_model_path, num_expert_demos
    )

    print("Generating diverse policy demonstrations...")
    diverse_trajectories = irl.generate_diverse_demonstrations(num_random_demos // 3)

    # Learn weights
    print("Learning reward weights using Maximum Margin IRL...")
    learned_weights = irl.learn_weights(expert_trajectories, diverse_trajectories)

    # Display results
    print(f"\nLearned weights:")
    print(f"  Infection penalty weight: {learned_weights[0]:.3f}")
    print(f"  Economic penalty weight:  {learned_weights[1]:.3f}")
    print(f"  Weight ratio (infection/economic): {learned_weights[0]/learned_weights[1]:.3f}")

    return irl


def test_irl_with_known_weights():
    """
    Test IRL algorithm with synthetic data from known reward weights.
    Uses Q-learning planning to generate proper expert demonstrations.
    """
    print("Testing IRL with known reward weights...")

    # Create environment and IRL
    env = SIREpidemicEnv(population=5000, max_steps=100)
    irl = MaxMarginIRL(env, feature_dim=2, regularization=1.0)

    # Set known true weights based on original environment (100:20 ratio)
    true_weights = np.array([100, 20])
    true_weights = true_weights / np.sum(true_weights)  # Normalize: [0.833, 0.167]

    print(f"True weights: infection={true_weights[0]:.3f}, economic={true_weights[1]:.3f}")

    # Generate expert trajectories using Q-learning planning with true weights
    print("Training expert policy with true weights...")
    expert_trajectories = irl.generate_expert_by_planning(true_weights, num_episodes=30)

    # Generate diverse comparison trajectories
    print("Generating diverse comparison policies...")
    diverse_trajectories = irl.generate_diverse_demonstrations(10)

    # Learn weights
    print("Learning weights from demonstrations...")
    learned_weights = irl.learn_weights(expert_trajectories, diverse_trajectories)

    # Compare results
    print(f"Learned weights: infection={learned_weights[0]:.3f}, economic={learned_weights[1]:.3f}")
    print(f"Weight difference: {np.abs(learned_weights - true_weights)}")
    print(f"Recovery success: {np.allclose(learned_weights, true_weights, atol=0.1)}")


def visualize_irl_results(irl: MaxMarginIRL, save_path: str = 'results/irl_analysis.png'):
    """
    Visualize IRL learning results and feature analysis.

    Args:
        irl: Trained IRL model
        save_path: Path to save visualization
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Learned weights visualization
    weights = irl.weights
    labels = ['Infection\nPenalty', 'Economic\nPenalty']
    colors = ['red', 'blue']

    ax1.bar(labels, weights, color=colors, alpha=0.7)
    ax1.set_title('Learned Reward Weights')
    ax1.set_ylabel('Weight')
    ax1.set_ylim(0, 1)

    # Add weight values on bars
    for i, (label, weight) in enumerate(zip(labels, weights)):
        ax1.text(i, weight + 0.02, f'{weight:.3f}', ha='center', va='bottom')

    # 2. Feature distribution analysis
    # Sample some states and actions to analyze feature distribution
    states = []
    features_infection = []
    features_economic = []

    for _ in range(1000):
        # Sample random state
        s = np.random.random()  # S%
        i = np.random.random() * (1 - s)  # I%
        r = 1 - s - i  # R%
        state = np.array([s, i, r])

        # Sample random action
        action = np.random.randint(0, 3)

        features = irl.extract_features(state, action)
        states.append(state)
        features_infection.append(features[0])
        features_economic.append(features[1])

    ax2.scatter(features_infection, features_economic, alpha=0.5, s=10)
    ax2.set_xlabel('Infection Feature')
    ax2.set_ylabel('Economic Feature')
    ax2.set_title('Feature Space Distribution')
    ax2.grid(True, alpha=0.3)

    # 3. Policy comparison under learned reward
    # Compare different actions across infection levels
    infection_levels = np.linspace(0, 0.5, 50)
    action_values = {0: [], 1: [], 2: []}

    for i_level in infection_levels:
        state = np.array([1-i_level, i_level, 0])  # Simple state: only S and I

        for action in range(3):
            features = irl.extract_features(state, action)
            value = np.dot(weights, features)
            action_values[action].append(value)

    for action in range(3):
        action_names = ['No Isolation', 'Partial Isolation', 'Full Isolation']
        ax3.plot(infection_levels, action_values[action],
                label=action_names[action], linewidth=2)

    ax3.set_xlabel('Infected Population %')
    ax3.set_ylabel('Action Value (Learned Reward)')
    ax3.set_title('Policy Analysis Under Learned Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Weight sensitivity analysis
    weight_ratios = np.linspace(0.1, 0.9, 20)
    optimal_actions = []

    for ratio in weight_ratios:
        temp_weights = np.array([ratio, 1-ratio])

        # Test with medium infection state
        state = np.array([0.7, 0.2, 0.1])
        best_action = 0
        best_value = float('-inf')

        for action in range(3):
            features = irl.extract_features(state, action)
            value = np.dot(temp_weights, features)
            if value > best_value:
                best_value = value
                best_action = action

        optimal_actions.append(best_action)

    ax4.plot(weight_ratios, optimal_actions, 'o-', linewidth=2, markersize=4)
    ax4.set_xlabel('Infection Weight Ratio')
    ax4.set_ylabel('Optimal Action')
    ax4.set_title('Policy Sensitivity to Weight Changes')
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['No Isolation', 'Partial', 'Full'])
    ax4.grid(True, alpha=0.3)

    # Mark learned weight ratio
    learned_ratio = weights[0] / weights.sum()
    ax4.axvline(learned_ratio, color='red', linestyle='--', alpha=0.7,
                label=f'Learned ratio: {learned_ratio:.3f}')
    ax4.legend()

    plt.tight_layout()

    # Ensure results directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"IRL analysis visualization saved to: {save_path}")


if __name__ == "__main__":
    # Example usage
    print("Testing IRL with known weights...")
    test_irl_with_known_weights()

    print("\nTraining IRL from expert policy...")
    expert_model_path = 'results/q_learning_model.pkl'
    if os.path.exists(expert_model_path):
        # Basic IRL training
        irl = train_irl_from_expert(expert_model_path)
        print(f"Basic IRL weights (raw): {irl.weights_raw}")

        # Iterative Max-Margin IRL for more robust results
        print("\nRunning iterative Max-Margin IRL...")
        expert_trajectories = irl.generate_expert_demonstrations(expert_model_path, 20)
        initial_adversaries = irl.generate_diverse_demonstrations(7)

        final_weights, final_adversaries = irl.iterate_max_margin(expert_trajectories, initial_adversaries, max_iterations=3)
        print(f"Final iterative weights (raw): {final_weights}")

        # Comprehensive analysis with visualizations (use raw weights)
        irl.comprehensive_analysis(expert_trajectories, final_adversaries, irl.weights_raw)

    else:
        print(f"Expert model not found at {expert_model_path}")
        print("Run 'python q_learning.py' first to train expert.")