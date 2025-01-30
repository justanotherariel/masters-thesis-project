#!/usr/bin/env bash

# Function to run a command for specified duration (in hours)
run_command() {
    local cmd="$1"
    local duration_hours="$2"
    
    # Start the command in background
    eval "$cmd" &
    local pid=$!
    
    # Sleep for specified duration (convert hours to seconds)
    sleep $(($duration_hours * 3600))
    # sleep 60
    
    # Kill all python processes
    pkill -f python
    
    # Ensure our specific process is killed
    kill -9 $pid 2>/dev/null
    
    # Wait a bit to ensure cleanup
    sleep 5
}

# List of commands with their durations (in hours)
run_command "wandb agent a-ebersberger-tu-delft/Thesis/ok9mfhxm" 3    # U-Net Rebalance Loss
run_command "wandb agent a-ebersberger-tu-delft/Thesis/fkinrfy0" 2    # U-Net CE Loss
run_command "wandb agent a-ebersberger-tu-delft/Thesis/mubff62j" 3    # Transformer Rebalance Loss
run_command "wandb agent a-ebersberger-tu-delft/Thesis/r5n2mxex" 3    # Transformer CE Loss

echo "All commands completed"