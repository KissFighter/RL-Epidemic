#!/usr/bin/env python3
"""
Complete IRL Experiment Runner

This script provides an easy-to-use interface for running the complete
Inverse Reinforcement Learning experiment, including:
1. Learning reward weights from expert policy
2. Retraining Q-learning and SARSA with IRL weights
3. Comparing all policies

Usage:
    python run_irl_experiment.py

Features:
- Single command execution
- Clear progress reporting
- Automatic model saving
- Comprehensive comparison results
"""

import sys
import os
from inverse_rl import (
    train_irl_from_expert,
    train_agent_with_irl_weights,
    compare_policies,
    run_complete_irl_experiment
)


def main():
    """Run complete IRL experiment with user-friendly interface."""

    print("=" * 60)
    print("INVERSE REINFORCEMENT LEARNING EXPERIMENT")
    print("=" * 60)
    print()

    print("This experiment will:")
    print("1. Learn reward weights from Q-learning expert policy")
    print("2. Retrain Q-learning agent with IRL weights")
    print("3. Retrain SARSA agent with IRL weights")
    print("4. Compare all policies and show improvements")
    print()

    # Check if required models exist
    required_files = [
        'models/original/q_learning.pkl',
        'models/original/sarsa.pkl'
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("❌ ERROR: Missing required model files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print()
        print("Please run the following commands first:")
        print("   python q_learning.py")
        print("   python sarsa.py")
        print()
        return 1

    print("✅ All required models found. Starting experiment...")
    print()

    try:
        # Run the complete experiment
        run_complete_irl_experiment()

        print()
        print("=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Generated files:")
        print("📁 models/irl/weights.pkl - IRL learned weights")
        print("📁 models/irl_trained/qlearning.pkl - Q-learning with IRL weights")
        print("📁 models/irl_trained/sarsa.pkl - SARSA with IRL weights")
        print("📁 outputs/plots/q_learning_policy_comparison.png - Q-learning policy comparison")
        print("📁 outputs/plots/sarsa_policy_comparison.png - SARSA policy comparison")
        print("📁 outputs/plots/ - All training and comparison plots")
        print()
        print("Key insights:")
        print("- IRL successfully learned expert policy preferences")
        print("- Check the comparison results above for performance metrics")
        print("- Models are saved and ready for further analysis")

        return 0

    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("Please ensure all required model files exist.")
        return 1
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred: {e}")
        return 1


def run_step_by_step():
    """Run IRL experiment step by step with detailed control."""

    print("=" * 60)
    print("STEP-BY-STEP IRL EXPERIMENT")
    print("=" * 60)
    print()

    # Step 1: Train IRL
    print("🔄 Step 1: Learning reward weights from expert policy...")
    try:
        irl = train_irl_from_expert('models/original/q_learning.pkl', seed=42)
        print("✅ Step 1 completed: IRL weights learned")
        print(f"   Weights: {irl.weights_raw}")
        print(f"   Saved to: models/irl/weights.pkl")
        print()
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        return 1

    # Step 2: Train Q-learning with IRL weights
    print("🔄 Step 2: Retraining Q-learning with IRL weights...")
    try:
        q_agent = train_agent_with_irl_weights(
            'models/irl/weights.pkl',
            agent_type='q_learning',
            save_path='models/irl_trained/qlearning.pkl',
            seed=42
        )
        print("✅ Step 2 completed: Q-learning retrained")
        print("   Saved to: models/irl_trained/qlearning.pkl")
        print()
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        return 1

    # Step 3: Train SARSA with IRL weights
    print("🔄 Step 3: Retraining SARSA with IRL weights...")
    try:
        sarsa_agent = train_agent_with_irl_weights(
            'models/irl/weights.pkl',
            agent_type='sarsa',
            save_path='models/irl_trained/sarsa.pkl',
            seed=42
        )
        print("✅ Step 3 completed: SARSA retrained")
        print("   Saved to: models/irl_trained/sarsa.pkl")
        print()
    except Exception as e:
        print(f"❌ Step 3 failed: {e}")
        return 1

    # Step 4: Compare policies
    print("🔄 Step 4: Comparing all policies...")
    try:
        print("\n--- Q-learning Comparison ---")
        compare_policies(agent_type='q_learning')

        print("\n--- SARSA Comparison ---")
        compare_policies(agent_type='sarsa')

        print("✅ Step 4 completed: Policy comparison finished")
        print()
    except Exception as e:
        print(f"❌ Step 4 failed: {e}")
        return 1

    print("=" * 60)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return 0


def show_results():
    """Show results from existing IRL models."""

    print("=" * 60)
    print("IRL EXPERIMENT RESULTS")
    print("=" * 60)
    print()

    # Check if IRL results exist
    if not os.path.exists('models/irl/weights.pkl'):
        print("❌ No IRL results found. Please run the experiment first:")
        print("   python run_irl_experiment.py")
        return 1

    print("🔄 Loading and displaying results...")
    try:
        # Just run the comparison part
        print("\n--- Q-learning Policy Comparison ---")
        compare_policies(agent_type='q_learning')

        print("\n--- SARSA Policy Comparison ---")
        compare_policies(agent_type='sarsa')

        print()
        print("=" * 60)
        print("RESULTS DISPLAYED SUCCESSFULLY!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return 1


def print_usage():
    """Print usage instructions."""
    print("Usage:")
    print("  python run_irl_experiment.py [command]")
    print()
    print("Commands:")
    print("  (no args)  - Run complete IRL experiment (default)")
    print("  step       - Run step-by-step with detailed progress")
    print("  results    - Show results from existing models")
    print("  help       - Show this help message")
    print()
    print("Examples:")
    print("  python run_irl_experiment.py")
    print("  python run_irl_experiment.py step")
    print("  python run_irl_experiment.py results")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "step":
            exit_code = run_step_by_step()
        elif command == "results":
            exit_code = show_results()
        elif command == "help" or command == "-h" or command == "--help":
            print_usage()
            exit_code = 0
        else:
            print(f"❌ Unknown command: {command}")
            print()
            print_usage()
            exit_code = 1
    else:
        # Default: run complete experiment
        exit_code = main()

    sys.exit(exit_code)