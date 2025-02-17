#!/usr/bin/env bash

# Function to run a command for specified duration (in hours and optional minutes)
run_command() {
    local id="$1"
    local hours="$2"
    local minutes="${3:-0}"
    
    # Convert time to seconds unless in test mode
    if [[ $test_mode == true ]]; then
        duration=60
    else
        duration=$((hours * 3600 + minutes * 60))
    fi
    
    # Start the command in background
    eval "$base_command$id" &
    local pid=$!
    
    # Sleep for specified duration
    sleep $duration
    
    # Kill all python processes
    pkill -f python
    
    # Ensure our specific process is killed
    kill -9 $pid 2>/dev/null
    
    # Wait a bit to ensure cleanup
    sleep 10
}

# Create
if [[ $1 == "create" ]]; then
    wandb sweep conf/sweep/unet.yaml
    wandb sweep conf/sweep/transformer.yaml
    exit
fi

# Set test mode based on argument
test_mode=false
if [[ $1 == "test" ]]; then
    test_mode=true
fi

# Base command to run
base_command="wandb agent a-ebersberger-tu-delft/Thesis/"

# List of commands with their durations
run_command "ypo2nbbv" 5    # Transformer
run_command "miu2s9va" 5    # U-Net

echo "All commands completed"