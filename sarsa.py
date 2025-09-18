#!/usr/bin/env python3
"""
SARSA Algorithm Implementation (On-Policy)

This module implements the SARSA algorithm for epidemic control.
SARSA is an on-policy temporal difference learning method that learns
the action-value function for the policy being followed.

Key features:
- On-policy: single epsilon-greedy policy for both action selection and updates
- Fixed epsilon for consistent policy behavior
- Tabular SARSA with state discretization
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from environment import SIREpidemicEnv


class SARSAAgent:
    """
    SARSA agent for epidemic control with on-policy learning.
    
    Unlike Q-learning, SARSA uses the same policy for both:
    - Action selection during environment interaction
    - Value estimation in learning updates
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
        Initialize SARSA agent.
        
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
        
        print(f"SARSA Agent initialized:")
        print(f"  State space: {state_size}D with {state_bins} bins each")
        print(f"  Action space: {action_size} actions")
        print(f"  Q-table shape: {q_shape}")
        print(f"  Fixed epsilon: {epsilon} (no decay)")
    
    def _discretize_state(self, state: np.ndarray) -> tuple:
        """
        Discretize continuous state into bins for tabular SARSA.
        
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
    
    def epsilon_greedy_policy(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy policy for both action selection and learning.
        
        This single policy is used throughout SARSA (on-policy characteristic).
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
    
    def choose_action(self, state: np.ndarray) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        return self.epsilon_greedy_policy(state)
    
    def learn(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, next_action: int, done: bool):
        """
        SARSA update rule (on-policy).
        
        Uses the same epsilon-greedy policy for both action selection and updates.
        This is the key characteristic of on-policy learning.
        
        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
                                    ^^^^^^
                                    Actual next action from policy
        
        Args:
            state: Current state
            action: Action taken
            reward: Received reward
            next_state: Resulting next state
            next_action: Next action (from same policy)
            done: Whether episode terminated
        """
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        
        # Current Q-value
        current_q = self.q_table[discrete_state][action]
        
        if done:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Use actual next action from the same policy
            # This makes SARSA on-policy
            target_q = reward + self.discount_factor * self.q_table[discrete_next_state][next_action]
        
        # SARSA update
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
        
        # Best action (greedy)
        best_action = np.argmax(q_values)
        
        # Epsilon-greedy policy probabilities
        policy_probs = np.ones(self.action_size) * (self.epsilon / self.action_size)
        policy_probs[best_action] += (1 - self.epsilon)
        
        return {
            'discrete_state': discrete_state,
            'q_values': q_values,
            'best_action': best_action,
            'policy_probabilities': policy_probs,
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


def train_sarsa(episodes: int = 500, max_steps: int = 100, seed: int = 42):
    """
    Train SARSA agent on epidemic control task.
    
    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        
    Returns:
        Trained agent and environment
    """
    print("=== SARSA Training (On-Policy) ===")
    print(f"Using random seed: {seed}")

    # Set global random seed
    np.random.seed(seed)

    # Create environment and agent with seeds
    env = SIREpidemicEnv(population=5000, max_steps=max_steps, seed=seed)
    agent = SARSAAgent(
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
        
        # Choose first action using epsilon-greedy policy
        action = agent.choose_action(state)
        
        for step in range(max_steps):
            # Execute action in environment
            next_state, reward, done, info = env.step(action)
            
            if done:
                # Terminal state: learn without next action
                agent.learn(state, action, reward, next_state, 0, True)
                total_reward += reward
                steps += 1
                break
            else:
                # Choose next action using same policy (on-policy)
                next_action = agent.choose_action(next_state)
                
                # SARSA update using current and next actions
                agent.learn(state, action, reward, next_state, next_action, False)
                
                # Update for next iteration
                state = next_state
                action = next_action  # Key: use the chosen next_action
                total_reward += reward
                steps += 1
        
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
    os.makedirs('models/original', exist_ok=True)
    agent.save_model('models/original/sarsa.pkl')
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('SARSA Training Rewards')
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
    # Save training plots
    os.makedirs('outputs/plots', exist_ok=True)
    plt.savefig('outputs/plots/sarsa_training.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTraining completed!")
    return agent, env


def test_sarsa(seed: int = 42):
    """Test trained SARSA agent."""
    print("\n=== Testing SARSA Agent ===")
    print(f"Using random seed: {seed}")

    # Set global random seed
    np.random.seed(seed)

    # Create environment and agent with seeds
    env = SIREpidemicEnv(population=5000, max_steps=100, seed=seed)
    agent = SARSAAgent(
        state_size=env.state_size,
        action_size=env.action_space_size,
        seed=seed
    )
    
    try:
        # Load trained model
        agent.load_model('models/original/sarsa.pkl')
        
        # Test episode using the learned policy
        state = env.reset()
        total_reward = 0
        actions = []
        
        print("\nTesting with learned epsilon-greedy policy...")
        
        for step in range(100):
            # Use the learned epsilon-greedy policy
            action = agent.choose_action(state)
            actions.append(action)
            
            # Get policy information
            if step % 25 == 0:
                policy_info = agent.get_policy_info(state)
                print(f"Step {step:2d}: Q-values = {policy_info['q_values']}, "
                      f"Best action = {policy_info['best_action']}")
            
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        print(f"\nTest Results:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Episode length: {step + 1} steps")
        print(f"  Actions taken: {actions[:10]}..." if len(actions) > 10 else f"  Actions taken: {actions}")
        
        # Visualize results
        env.render(save_path='outputs/plots/sarsa_test.png')
        
        # Print statistics
        stats = env.get_statistics()
        print(f"\nEpidemic Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
    
    except FileNotFoundError:
        print("No trained model found. Please run training first.")


def compare_with_greedy_policy():
    """Compare SARSA policy with pure greedy policy."""
    print("\n=== Comparing SARSA vs Greedy Policy ===")
    
    try:
        # Load trained model
        env = SIREpidemicEnv(population=5000, max_steps=100)
        agent = SARSAAgent(state_size=env.state_size, action_size=env.action_space_size)
        agent.load_model('models/original/sarsa.pkl')
        
        # Test with epsilon-greedy (learned policy)
        print("Testing with epsilon-greedy policy (ε=0.1)...")
        state = env.reset()
        sarsa_reward = 0
        for step in range(100):
            action = agent.choose_action(state)  # epsilon-greedy
            state, reward, done, _ = env.step(action)
            sarsa_reward += reward
            if done:
                break
        
        # Test with pure greedy policy (epsilon=0)
        print("Testing with pure greedy policy (ε=0)...")
        old_epsilon = agent.epsilon
        agent.epsilon = 0  # Temporarily set to 0
        
        state = env.reset()
        greedy_reward = 0
        for step in range(100):
            action = agent.choose_action(state)  # pure greedy
            state, reward, done, _ = env.step(action)
            greedy_reward += reward
            if done:
                break
        
        agent.epsilon = old_epsilon  # Restore original epsilon
        
        print(f"\nComparison Results:")
        print(f"  SARSA policy (ε={old_epsilon}): {sarsa_reward:.2f}")
        print(f"  Greedy policy (ε=0): {greedy_reward:.2f}")
        print(f"  Difference: {greedy_reward - sarsa_reward:.2f}")
        
    except FileNotFoundError:
        print("No trained model found. Please run training first.")


if __name__ == "__main__":
    # Set main random seed
    MAIN_SEED = 42
    np.random.seed(MAIN_SEED)

    # Train the agent
    agent, env = train_sarsa(episodes=500, max_steps=100, seed=MAIN_SEED)

    # Test the trained agent
    test_sarsa(seed=MAIN_SEED)

    # Compare policies
    compare_with_greedy_policy()
    
    print(f"\nSARSA (On-Policy) Summary:")
    print(f"- Single Policy: ε-greedy (ε={agent.epsilon})")
    print(f"- Learning: Updates based on actions actually taken by the policy")
    print(f"- Consistency: Action selection and value updates use same policy")