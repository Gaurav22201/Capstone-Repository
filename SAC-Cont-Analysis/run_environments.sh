#!/bin/bash

# Create necessary directories
mkdir -p results
mkdir -p model
mkdir -p videos

# Function to run training for a specific environment
run_environment() {
    local env_index=$1
    local env_name=$2
    local max_steps=$3
    
    echo "====================================================="
    echo "Starting training for environment $env_index: $env_name"
    echo "====================================================="
    
    # Run the training script with appropriate parameters
    python main.py --EnvIdex $env_index --Max_train_steps $max_steps --write True --record_video True
    
    echo "Training completed for $env_name"
    echo "====================================================="
}

# Run each environment with appropriate max steps
# Pendulum-v1 (Env 0) - typically needs fewer steps
run_environment 0 "Pendulum-v1" 500000

# LunarLanderContinuous-v2 (Env 1) - medium complexity
run_environment 1 "LunarLanderContinuous-v2" 1000000

# Humanoid-v4 (Env 2) - more complex, needs more steps
run_environment 2 "Humanoid-v4" 2000000

echo "All environments have been trained!"
echo "Results are saved in the results directory"
echo "Videos are saved in the videos directory"
echo "Models are saved in the model directory" 