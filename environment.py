#!/usr/bin/env python3
"""
SIR Epidemic Environment for Reinforcement Learning

This module implements a simplified SIR (Susceptible-Infected-Recovered) 
epidemic model as a reinforcement learning environment.

State space: [S%, I%, R%] - 3D normalized values
Action space: {0: no isolation, 1: partial isolation, 2: full isolation}
Reward: Balance between infection control and economic costs
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any


class SIREpidemicEnv:
    """
    Simplified SIR epidemic model environment for reinforcement learning.
    
    The environment simulates the spread of an epidemic using the classic
    SIR model with controllable intervention measures.
    """
    
    def __init__(self, 
                 population: int = 5000,
                 initial_infected: int = 10,
                 beta: float = 0.3,  # transmission rate
                 gamma: float = 0.1,  # recovery rate
                 max_steps: int = 100):
        """
        Initialize the SIR epidemic environment.
        
        Args:
            population: Total population size
            initial_infected: Initial number of infected individuals
            beta: Base transmission rate
            gamma: Recovery rate
            max_steps: Maximum simulation steps (days)
        """
        self.population = population
        self.initial_infected = initial_infected
        self.beta = beta
        self.gamma = gamma
        self.max_steps = max_steps
        
        # Action space: 3 isolation levels
        self.action_space_size = 3
        
        # State space: S, I, R variables only
        self.state_size = 3
        
        # Initialize environment
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state as normalized numpy array [S%, I%, R%]
        """
        self.S = self.population - self.initial_infected
        self.I = self.initial_infected
        self.R = 0
        self.day = 0
        
        # Record history for visualization
        self.history = {
            'S': [self.S],
            'I': [self.I],
            'R': [self.R],
            'actions': []
        }
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        Returns:
            Normalized state array [S%, I%, R%]
        """
        return np.array([
            self.S / self.population,  # Susceptible percentage
            self.I / self.population,  # Infected percentage
            self.R / self.population   # Recovered percentage
        ], dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: 0=no isolation, 1=partial isolation, 2=full isolation
            
        Returns:
            next_state: Next state observation
            reward: Reward for the action
            done: Whether episode is finished
            info: Additional information dictionary
        """
        if self.day >= self.max_steps:
            return self._get_state(), 0, True, {}
        
        # Apply action effects on transmission rate and economic cost
        effective_beta, economic_cost = self._apply_action(action)
        
        # Update SIR dynamics using discrete-time equations
        self._update_sir_dynamics(effective_beta)
        
        self.day += 1
        
        # Calculate reward
        reward = self._calculate_reward(economic_cost)
        
        # Record history
        self._record_history(action)
        
        # Check termination conditions
        done = (self.day >= self.max_steps) or (self.I < 1)
        
        info = {
            'S': self.S,
            'I': self.I,
            'R': self.R,
            'day': self.day,
            'economic_cost': economic_cost
        }
        
        return self._get_state(), reward, done, info
    
    def _apply_action(self, action: int) -> Tuple[float, float]:
        """
        Apply action to get effective transmission rate and economic cost.
        
        Args:
            action: Action to apply
            
        Returns:
            effective_beta: Modified transmission rate
            economic_cost: Economic cost of the action
        """
        if action == 0:  # No isolation
            effective_beta = self.beta
            economic_cost = 0.0
        elif action == 1:  # Partial isolation
            effective_beta = self.beta * 0.5  # 50% reduction
            economic_cost = 0.2  # 20% economic impact
        else:  # Full isolation (action == 2)
            effective_beta = self.beta * 0.1  # 90% reduction
            economic_cost = 0.5  # 50% economic impact
            
        return effective_beta, economic_cost
    
    def _update_sir_dynamics(self, effective_beta: float):
        """
        Update SIR populations using discrete-time dynamics.
        
        Args:
            effective_beta: Current effective transmission rate
        """
        # SIR differential equations (discretized)
        dS = -effective_beta * self.S * self.I / self.population
        dI = effective_beta * self.S * self.I / self.population - self.gamma * self.I
        dR = self.gamma * self.I
        
        # Update populations
        self.S = max(0, self.S + dS)
        self.I = max(0, self.I + dI)
        self.R = max(0, self.R + dR)
        
        # Ensure population conservation
        total = self.S + self.I + self.R
        if total > 0:
            self.S = self.S * self.population / total
            self.I = self.I * self.population / total
            self.R = self.R * self.population / total
    
    def _calculate_reward(self, economic_cost: float) -> float:
        """
        Calculate reward balancing infection control and economic costs.
        
        Args:
            economic_cost: Economic cost from current action
            
        Returns:
            Reward value (negative, to be minimized)
        """
        # Infection penalty: higher infections = higher penalty
        infection_penalty = -(self.I / self.population) * 100
        
        # Economic cost penalty
        economic_penalty = -economic_cost * 20
        
        # Total reward
        return infection_penalty + economic_penalty
    
    def _record_history(self, action: int):
        """Record current state and action for visualization."""
        self.history['S'].append(self.S)
        self.history['I'].append(self.I)
        self.history['R'].append(self.R)
        self.history['actions'].append(action)
    
    def render(self, save_path: str = None):
        """
        Visualize epidemic curves and policy decisions.
        
        Args:
            save_path: Optional path to save the plot
        """
        days = range(len(self.history['S']))
        
        plt.figure(figsize=(12, 8))
        
        # Plot SIR curves
        plt.subplot(2, 1, 1)
        plt.plot(days, self.history['S'], 'b-', label='Susceptible(S)', linewidth=2)
        plt.plot(days, self.history['I'], 'r-', label='Infected(I)', linewidth=2)
        plt.plot(days, self.history['R'], 'g-', label='Recovered(R)', linewidth=2)
        plt.xlabel('Days')
        plt.ylabel('Population')
        plt.title('SIR Epidemic Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot action sequence
        if self.history['actions']:
            plt.subplot(2, 1, 2)
            action_names = ['No Isolation', 'Partial Isolation', 'Full Isolation']
            colors = ['green', 'orange', 'red']
            
            for i, action in enumerate(self.history['actions']):
                plt.bar(i, action, color=colors[action], alpha=0.7)
            
            plt.xlabel('Days')
            plt.ylabel('Isolation Level')
            plt.title('Policy Decisions')
            plt.yticks([0, 1, 2], action_names)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get epidemic statistics for analysis.
        
        Returns:
            Dictionary containing key epidemic metrics
        """
        if not self.history['I']:
            return {}
            
        peak_infections = max(self.history['I'])
        peak_day = np.argmax(self.history['I'])
        
        return {
            'peak_infections': peak_infections,
            'peak_day': peak_day,
            'final_susceptible': self.S,
            'final_recovered': self.R,
            'attack_rate': (self.population - self.S) / self.population
        }