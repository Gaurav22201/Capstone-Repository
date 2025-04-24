#!/usr/bin/env python3
import os
import subprocess
import time
import argparse
from analyze_results import main as analyze_main

def run_environment(env_index, env_name, max_steps):
    """Run training for a specific environment"""
    print("=" * 60)
    print(f"Starting training for environment {env_index}: {env_name}")
    print("=" * 60)
    
    # Create command to run the training script
    cmd = [
        "python", "main.py",
        "--EnvIdex", str(env_index),
        "--Max_train_steps", str(max_steps),
        "--write", "True",
        "--record_video", "True",  # Enable video recording
        "--eval_interval", "10000",  # Changed to 10K steps to match save_interval
        "--save_interval", "10000",  # Save every 10K steps
        "--dvc", "cpu"  # Use CPU instead of CUDA
    ]
    
    # Run the command
    process = subprocess.Popen(cmd)
    process.wait()
    
    print(f"Training completed for {env_name}")
    print("=" * 60)

def main():
    # Check if we're in the correct directory
    current_dir = os.path.basename(os.path.abspath(os.getcwd()))
    if current_dir != "TD3-Analysis":
        print(f"Warning: Current directory is '{current_dir}', expected 'TD3-Analysis'")
        print("Changing to the TD3-Analysis directory...")
        os.chdir("TD3-Analysis")
    
    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    os.makedirs("videos", exist_ok=True)  # Add videos directory
    
    # Define environments with reduced steps to 50,000
    base_steps = 50000  # Reduced to 50K steps
    environments = [
        (0, "Pendulum-v1", base_steps),  # 50K steps
        (1, "LunarLanderContinuous-v3", base_steps),  # 50K steps
        (2, "Humanoid-v5", base_steps)  # 50K steps
    ]
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run DDPG on multiple environments")
    parser.add_argument("--envs", type=str, default="0,1,2", 
                        help="Comma-separated list of environment indices to run (default: 0,1,2)")
    parser.add_argument("--steps", type=str, default=f"{base_steps},{base_steps},{base_steps}",
                        help="Comma-separated list of max steps for each environment")
    args = parser.parse_args()
    
    # Parse environment indices and steps
    env_indices = [int(idx) for idx in args.envs.split(",")]
    max_steps_list = [int(steps) for steps in args.steps.split(",")]
    
    # Ensure we have the same number of environments and step counts
    if len(env_indices) != len(max_steps_list):
        print("Error: Number of environments and step counts must match")
        return
    
    # Run each environment
    for i, env_idx in enumerate(env_indices):
        env_name = ["Pendulum-v1", "LunarLanderContinuous-v3", "Humanoid-v5"][env_idx]
        run_environment(env_idx, env_name, max_steps_list[i])
    
    print("All environments have been trained!")
    print("Results are saved in the results directory")
    print("Models are saved in the model directory")
    print("Videos are saved in the videos directory")
    
    print("\nRunning analysis...")
    # Run the analysis
    analyze_main()

if __name__ == "__main__":
    main() 