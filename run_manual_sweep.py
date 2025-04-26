#!/usr/bin/env python3
import subprocess
import itertools
import os
import time
import sys
import json
import argparse
import re
from typing import List, Dict, Any
import datetime
from pathlib import Path

BASE_COMMAND = ["python", "train.py"]
LOG_DIR = Path("data/logs")

def generate_permutations(sweep_config: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Generate all permutations of parameters from a sweep config."""
    keys = list(sweep_config.keys())
    values = [sweep_config[key] for key in keys]
    
    permutations = []
    for combination in itertools.product(*values):
        permutation = {}
        for key, value in zip(keys, combination):
            permutation[key] = value
        permutations.append(permutation)
    
    return permutations

def create_command(params: Dict[str, str]) -> List[str]:
    """Create the command list with the given parameters."""
    base_command = BASE_COMMAND.copy()
    
    # Add each parameter to the command
    for key, value in params.items():
        base_command.append(f"{key}={value}")
    
    return base_command

def display_progress(completed, total_runs, running_processes):
    """Display the current progress and running processes."""
    print(f"\nProgress: {completed}/{total_runs} completed | Running {len(running_processes)} processes:")
    
    for process, (idx, _, params) in running_processes.items():
        param_str = " ".join([f"{k}={v}" for k, v in params.items()])
        print(f"  Run {idx+1}: {param_str}")

def extract_group_id_from_log(log_file: Path) -> str:
    """
    Extract the group ID from a log file.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        The group ID if found, None otherwise
    """
    if not log_file.exists():
        return None
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Look for the group ID pattern in each line
                group_id_match = re.search(r"Group ID: ([a-z0-9]+)", line)
                if group_id_match:
                    return group_id_match.group(1)
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return None

def monitor_log_files(log_params_map: Dict[Path, Dict[str, str]], summary_file_path: Path) -> None:
    """
    Monitor log files for group IDs and append them to the group ID file.
    
    Args:
        log_params_map: Dictionary mapping log file paths to parameter dictionaries
    """
    # Make a copy of the keys to avoid modifying the dictionary during iteration
    log_files = list(log_params_map.keys())
    
    for idx, log_file in enumerate(log_files):
        if not log_file.exists():
            continue
            
        group_id = extract_group_id_from_log(log_file)
        if group_id:
            # Get the parameters for this log file
            params = log_params_map.pop(log_file)
                        
            # Append to the group ID file
            with open(summary_file_path, 'a') as f:
                f.write(f"# Run {idx}\n")
                f.write("".join([f"{k}: {v}\n" for k, v in params.items()]))
                f.write(f"Group ID: [{group_id}](https://wandb.ai/a-ebersberger-tu-delft/Thesis/groups/{group_id}/workspace)\n")
                f.write("\n")
            
            print(f"Found group ID: {group_id}")

def run_parameter_sweep(sweep_configs: List[Dict[str, List[str]]], concurrent_runs: int = 1) -> None:
    """Run parameter sweep with specified concurrency."""
    # Generate all permutations for each sweep config
    all_permutations = []
    for config in sweep_configs:
        all_permutations.extend(generate_permutations(config))
    
    total_runs = len(all_permutations)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create or clear the results file
    results_file = LOG_DIR / f"{run_name}_results.md"
    with open(results_file, 'w') as f:
        f.write(f"# Results for parameter sweep started at {timestamp}\n")
    
    print(f"Parameter sweep will run {total_runs} configurations:")
    for idx, params in enumerate(all_permutations):
        base_cmd_str = " ".join(BASE_COMMAND)
        param_str = " ".join([f"{k}={v}" for k, v in params.items()])
        print(f"  {idx+1}: {base_cmd_str} {param_str}")
    
    print(f"\nStarting parameter sweep with {concurrent_runs} concurrent runs")
    print(f"Note: Runs will be started at least 5 seconds apart to prevent Hydra output directory collisions")
    print(f"Results will be saved to {results_file}")
    
    completed = 0
    running_processes = {}  # Map process to its index, output file, and params
    log_params_map = {}  # Map log files to parameters for group ID extraction
    next_idx = 0
    status_changed = True
    last_process_start_time = 0  # Track when we last started a process
    last_monitor_time = 0  # Track when we last monitored log files
    
    try:
        while completed < total_runs:
            current_time = time.time()
            
            # Periodically check for group IDs in log files (every 10 seconds)
            if current_time - last_monitor_time >= 10:
                monitor_log_files(log_params_map, results_file)
                last_monitor_time = current_time
            
            # Start new processes if we have capacity and enough time has passed since last start
            process_started = False
            while (len(running_processes) < concurrent_runs and 
                   next_idx < total_runs and 
                   (current_time - last_process_start_time >= 5 or completed == 0)):
                
                params = all_permutations[next_idx]
                command = create_command(params)
                
                # Create a unique output file name
                output_file_name = f"{run_name}_{next_idx+1}_of_{total_runs}.log"
                output_file = LOG_DIR / output_file_name
                
                print(f"Starting run {next_idx+1}/{total_runs}: {' '.join(command)}")
                
                # Run the command and redirect output to file
                with open(output_file, 'w') as f:
                    process = subprocess.Popen(
                        command, 
                        stdout=f, 
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    running_processes[process] = (next_idx, output_file, params)
                    
                    # Add to log file to parameters mapping for group ID extraction
                    log_params_map[output_file] = params.copy()
                
                next_idx += 1
                process_started = True
                status_changed = True
                last_process_start_time = time.time()
                
                # Wait 5 seconds before starting the next process
                if next_idx < total_runs and len(running_processes) < concurrent_runs:
                    print(f"Waiting 5 seconds before starting next run to avoid Hydra directory collision...")
                    time.sleep(5)
            
            # Check for completed processes
            process_completed = False
            for process in list(running_processes.keys()):
                if process.poll() is not None:  # Process has terminated
                    idx, output_file, _ = running_processes.pop(process)
                    completed += 1
                    return_code = process.returncode
                    print(f"Completed run {idx+1}/{total_runs} with return code {return_code}")
                    process_completed = True
                    status_changed = True
            
            # Display progress when status changes
            if status_changed:
                display_progress(completed, total_runs, running_processes)
                status_changed = False
            
            # If nothing changed, wait before checking again
            if not process_started and not process_completed:
                time.sleep(1)
        
        # Final check for any remaining group IDs
        monitor_log_files(log_params_map, results_file)
        
        print(f"\nParameter sweep completed. All {total_runs} configurations have been run.")
        print(f"Group IDs and configurations have been saved to {results_file}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Terminating running processes...")
        for process in running_processes:
            process.terminate()
        
        # Final check for any remaining group IDs before exiting
        monitor_log_files(log_params_map, results_file)
        
        print(f"Completed {completed}/{total_runs} configurations before interruption.")
        print(f"Group IDs and configurations have been saved to {results_file}")
        sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run parameter sweep for training jobs.')
    parser.add_argument(
        '--config', 
        type=str,
        default=None,
        help='Path to JSON config file with sweep configurations.'
    )
    parser.add_argument(
        '--concurrent-runs', 
        type=int, 
        default=1,
        help='Number of concurrent runs to execute.'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    common_config = {
        "n_trials": [10], 
        "trial_idx": [-1],
    }
    
    # Default sweep configuration if no config file is provided
    default_sweep_configs = [
        {
            'model': ['transformer'],
            'model.train_sys.steps.0.model.model_cls._target_': 
                [
                    'src.modules.training.models.transformer.classic.CombAction',
                    'src.modules.training.models.transformer.classic_zero_self.CombAction'
                ],
            'model.env_sys.steps.1.train_envs': ['4', '8', '12', '16'],
            'model.env_sys.steps.1.train_keep_perc': ['0.2', '0.4', '0.6']
        },
        {
            'model': ['transformer_sparse'],
            'model.env_sys.steps.1.train_envs': ['4', '8', '12', '16'],
            'model.env_sys.steps.1.train_keep_perc': ['0.2', '0.4', '0.6']
        }
    ]
    
    # Merge common config with sweep configs
    for config in default_sweep_configs:
        config.update(common_config)
    
    if args.config:
        try:
            with open(args.config, 'r') as f:
                sweep_configs = json.load(f)
            print(f"Loaded sweep configuration from {args.config}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading config file: {e}")
            print("Using default sweep configuration.")
            sweep_configs = default_sweep_configs
    else:
        sweep_configs = default_sweep_configs
    
    run_parameter_sweep(sweep_configs, args.concurrent_runs)