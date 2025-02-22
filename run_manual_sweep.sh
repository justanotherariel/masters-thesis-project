#!/usr/bin/env bash

# Define the base command
base_command="python train.py n_trials=5 trial_idx=0"

# Define the argument lists as arrays
model_args=("transformer_extensive" "unet_extensive")
env_args=("MiniGrid-LavaCrossingS9N3-v0" "MiniGrid-SimpleCrossingS9N3-v0" "MiniGrid-LavaCrossingS11N5-v0" "MiniGrid-SimpleCrossingS11N5-v0")

# Loop through all combinations
for model_arg in "${model_args[@]}"; do
    for env_arg in "${env_args[@]}"; do
        # if [[ "$model_arg" == "transformer_extensive" && "$env_arg" == "MiniGrid-LavaCrossingS9N3-v0" ]]; then
        #     continue
        # fi

        $base_command "model=$model_arg" "model.env_sys.steps.0.environment=$env_arg"
    done
done