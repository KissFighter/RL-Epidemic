#!/usr/bin/env python3
"""
Q-Learning Algorithm Implementation (Off-Policy)

This module implements the Q-learning algorithm for epidemic control.
Q-learning is an off-policy temporal difference learning method that learns
the optimal action-value function regardless of the policy being followed.

Key features:
- Off-policy: behavior policy (epsilon-greedy) ≠ target policy (greedy)
- Fixed epsilon for clear policy distinction
- Tabular Q-learning with state discretization
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from environment import SIREpidemicEnv


class QLearningAgent:
    """
    Q-Learning agent for epidemic control with explicit off-policy structure.
    
    The agent uses two distinct policies:
    - Behavior Policy: epsilon-greedy (for action selection)
    - Target Policy: greedy (for Q-value updates)
    """
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,  # Fixed epsilon - no decay
                 state_bins: int = 8,
                 seed: int = None):
        """
        Initialize Q-learning agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of available actions
            learning_rate: Learning rate alpha
            discount_factor: Discount factor gamma
            epsilon: Fixed exploration rate (no decay)
            state_bins: Number of bins for state discretization
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Fixed value
        self.state_bins = state_bins
        
        # Initialize Q-table: state_bins^state_size × action_size
        q_shape = [state_bins] * state_size + [action_size]
        self.q_table = np.zeros(q_shape)
        
        print(f"Q-Learning Agent initialized:")
        print(f"  State space: {state_size}D with {state_bins} bins each")
        print(f"  Action space: {action_size} actions")
        print(f"  Q-table shape: {q_shape}")
        print(f"  Fixed epsilon: {epsilon} (no decay)")
    
    def _discretize_state(self, state: np.ndarray) -> tuple:
        """
        Discretize continuous state into bins for tabular Q-learning.
        
        Args:
            state: Continuous state array [0,1]
            
        Returns:
            Discretized state as tuple of bin indices
        """
        discretized = []
        for s in state:
            # Map [0,1] to bin index [0, state_bins-1]
            bin_idx = min(int(s * self.state_bins), self.state_bins - 1)
            discretized.append(bin_idx)
        return tuple(discretized)
    
    def behavior_policy(self, state: np.ndarray) -> int:
        """
        Behavior Policy: epsilon-greedy policy for action selection.
        
        This policy is used during environment interaction to collect experience.
        It balances exploration (random actions) and exploitation (greedy actions).
        
        Args:
            state: Current state observation
            
        Returns:
            Selected action using epsilon-greedy policy
        """
        discrete_state = self._discretize_state(state)
        
        if np.random.random() <= self.epsilon:
            # Exploration: select random action
            return np.random.choice(self.action_size)
        else:
            # Exploitation: select greedy action
            return np.argmax(self.q_table[discrete_state])
    
    def target_policy(self, state: np.ndarray) -> int:
        """
        Target Policy: greedy policy for Q-value updates.
        
        This policy represents what we want to learn - the optimal policy.
        It always selects the action with highest Q-value.
        
        Args:
            state: Current state observation
            
        Returns:
            Greedy action (highest Q-value)
        """
        discrete_state = self._discretize_state(state)
        return np.argmax(self.q_table[discrete_state])
    
    def get_target_value(self, state: np.ndarray) -> float:
        """
        Get the value of a state under the target policy.
        
        This is used in Q-learning updates: max_a Q(s', a)
        
        Args:
            state: State to evaluate
            
        Returns:
            Maximum Q-value for the state
        """
        discrete_state = self._discretize_state(state)
        return np.max(self.q_table[discrete_state])
    
    def choose_action(self, state: np.ndarray, use_target_policy: bool = False) -> int:
        """
        Choose action using specified policy.
        
        Args:
            state: Current state
            use_target_policy: If True, use target policy (greedy)
                             If False, use behavior policy (epsilon-greedy)
            
        Returns:
            Selected action
        """
        if use_target_policy:
            return self.target_policy(state)
        else:
            return self.behavior_policy(state)
    
    def learn(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool):
        """
        Q-learning update rule (off-policy).
        
        Uses behavior policy to collect experience but updates toward target policy.
        This is the key characteristic of off-policy learning.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a Q(s',a) - Q(s,a)]
                                    ^^^^^^^^^^^^
                                    Target policy value
        
        Args:
            state: Current state
            action: Action taken (from behavior policy)
            reward: Received reward
            next_state: Resulting next state
            done: Whether episode terminated
        """
        discrete_state = self._discretize_state(state)
        
        # Current Q-value
        current_q = self.q_table[discrete_state][action]
        
        if done:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Use target policy to evaluate next state
            # This makes Q-learning off-policy
            target_q = reward + self.discount_factor * self.get_target_value(next_state)
        
        # Q-learning update
        td_error = target_q - current_q
        self.q_table[discrete_state][action] += self.learning_rate * td_error
    
    def get_policy_info(self, state: np.ndarray) -> dict:
        """
        Get detailed policy information for analysis.
        
        Args:
            state: State to analyze
            
        Returns:
            Dictionary with policy analysis information
        """
        discrete_state = self._discretize_state(state)
        q_values = self.q_table[discrete_state].copy()
        
        # Target policy action (greedy)
        target_action = np.argmax(q_values)
        
        # Behavior policy probabilities (epsilon-greedy)
        behavior_probs = np.ones(self.action_size) * (self.epsilon / self.action_size)
        behavior_probs[target_action] += (1 - self.epsilon)
        
        return {
            'discrete_state': discrete_state,
            'q_values': q_values,
            'target_action': target_action,
            'behavior_probabilities': behavior_probs,
            'epsilon': self.epsilon
        }
    
    def save_model(self, filepath: str):
        """Save Q-table and parameters to file."""
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'state_bins': self.state_bins,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load Q-table and parameters from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.epsilon = model_data['epsilon']
        self.state_bins = model_data['state_bins']
        self.learning_rate = model_data.get('learning_rate', 0.1)
        self.discount_factor = model_data.get('discount_factor', 0.95)
        print(f"Model loaded from {filepath}")


def train_q_learning(episodes: int = 500, max_steps: int = 100, seed: int = 42):
    """
    Train Q-learning agent on epidemic control task.

    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility

    Returns:
        Trained agent and environment
    """
    print("=== Q-Learning Training (Off-Policy) ===")
    print(f"Using random seed: {seed}")

    # Set global random seed
    np.random.seed(seed)

    # Create environment and agent with seeds
    env = SIREpidemicEnv(population=5000, max_steps=max_steps, seed=seed)
    agent = QLearningAgent(
        state_size=env.state_size,
        action_size=env.action_space_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1,  # Fixed epsilon
        state_bins=8,
        seed=seed
    )
    
    # Training history
    episode_rewards = []
    episode_steps = []
    
    print(f"\nTraining for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Use behavior policy (epsilon-greedy) for action selection
            action = agent.behavior_policy(state)
            
            # Execute action in environment
            next_state, reward, done, info = env.step(action)
            
            # Learn using off-policy Q-learning update
            agent.learn(state, action, reward, next_state, done)
            
            # Update for next iteration
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"Episode {episode + 1:4d}: "
                  f"Avg Reward = {avg_reward:6.1f}, "
                  f"Avg Steps = {avg_steps:4.1f}")
    
    # Save trained model
    os.makedirs('results', exist_ok=True)
    agent.save_model('results/q_learning_model.pkl')
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Q-Learning Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_steps)
    plt.title('Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/q_learning_training.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTraining completed!")
    return agent, env


def test_q_learning(seed: int = 42):
    """Test trained Q-learning agent."""
    print("\n=== Testing Q-Learning Agent ===")
    print(f"Using random seed: {seed}")

    # Set global random seed
    np.random.seed(seed)

    # Create environment and agent with seeds
    env = SIREpidemicEnv(population=5000, max_steps=100, seed=seed)
    agent = QLearningAgent(
        state_size=env.state_size,
        action_size=env.action_space_size,
        seed=seed
    )
    
    try:
        # Load trained model
        agent.load_model('results/q_learning_model.pkl')
        
        # Test episode using target policy (greedy)
        state = env.reset()
        total_reward = 0
        actions = []
        
        print("\nTesting with target policy (greedy)...")
        
        for step in range(100):
            # Use target policy for testing (pure exploitation)
            action = agent.choose_action(state, use_target_policy=True)
            actions.append(action)
            
            # Get policy information
            if step % 25 == 0:
                policy_info = agent.get_policy_info(state)
                print(f"Step {step:2d}: Q-values = {policy_info['q_values']}, "
                      f"Target action = {policy_info['target_action']}")
            
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        print(f"\nTest Results:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Episode length: {step + 1} steps")
        print(f"  Actions taken: {actions[:10]}..." if len(actions) > 10 else f"  Actions taken: {actions}")
        
        # Visualize results
        env.render(save_path='results/q_learning_test.png')
        
        # Print statistics
        stats = env.get_statistics()
        print(f"\nEpidemic Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
    
    except FileNotFoundError:
        print("No trained model found. Please run training first.")


if __name__ == "__main__":
    # Set main random seed
    MAIN_SEED = 42
    np.random.seed(MAIN_SEED)

    # Train the agent
    agent, env = train_q_learning(episodes=500, max_steps=100, seed=MAIN_SEED)

    # Test the trained agent
    test_q_learning(seed=MAIN_SEED)
    
    print(f"\nQ-Learning (Off-Policy) Summary:")
    print(f"- Behavior Policy: ε-greedy (ε={agent.epsilon})")
    print(f"- Target Policy: Greedy")
    print(f"- Learning: Updates toward target policy using behavior policy data")