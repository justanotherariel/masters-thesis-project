#!/usr/bin/env python3

import os
from pathlib import Path

RESULTS_DIR = Path("model-data/")

def parse_result_file(file_path):
    """Parse result file to extract runs, configurations, and metrics."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split the content by runs
    run_sections = content.split("# Run")[1:]  # Skip the first empty part
    
    results = []
    
    for run_section in run_sections:
        run_data = {}
        
        # Extract run number, arguments and results
        lines = run_section.strip().split('\n')
        
        # Initialize section pointers
        args_start = None
        args_end = None
        results_start = None
        results_end = len(lines)
        
        # Find section boundaries
        for i, line in enumerate(lines):
            if line.startswith("## Arguments"):
                args_start = i
            elif line.startswith("## Results"):
                results_start = i
                if args_start is not None and args_end is None:
                    args_end = i
        
        # Parse arguments
        arguments = {}
        if args_start is not None and args_end is not None:
            for line in lines[args_start+1:args_end]:
                if line.startswith("- "):
                    key_val = line[2:].split(": ", 1)
                    if len(key_val) == 2:
                        key, val = key_val
                        arguments[key] = val
        
        run_data["arguments"] = arguments
        
        # Parse results
        metrics = {}
        if results_start is not None:
            for line in lines[results_start+1:results_end]:
                if line.startswith("- "):
                    key_val = line[2:].split(": ", 1)
                    if len(key_val) == 2:
                        key, val = key_val
                        metrics[key] = val
        
        run_data["metrics"] = metrics
        results.append(run_data)
    
    return results

def sort_config_key(config):
    """Sort configurations by envs and keep percentage."""
    parts = config.split(", ")
    envs = int(parts[0].split(": ")[1])
    keep = float(parts[1].split(": ")[1])
    return (envs, keep)

def organize_data_by_metric(result_files, configs: dict):
    """Organize data by metric, configuration, and model."""
    
    def get_config_str(arguments):
        config_str = ""
        for config_short, config_long in configs.items():
            if config_long in arguments:
                config_str += f"{config_short}: {arguments[config_long]}, "
            else:
                raise ValueError(f"Configuration key '{config_long}' not found in arguments.")
        return config_str.strip(", ")
    
    # First, extract all unique metrics and configurations
    all_metrics = set()
    all_configs = set()
    
    model_results = {}
    
    for model_name, file_path in result_files.items():
        results = parse_result_file(file_path)
        model_results[model_name] = results
        
        for run in results:
            metrics = run["metrics"]
            arguments = run["arguments"]
            all_configs.add(get_config_str(arguments))
            all_metrics.update(metrics.keys())
    
    # Organize data by metric
    metric_data = {}
    for metric in all_metrics:
        metric_data[metric] = {}
        
        for config in all_configs:
            metric_data[metric][config] = {}
            
            for model_name in result_files.keys():
                metric_data[metric][config][model_name] = None
    
    # Fill in the data
    for model_name, results in model_results.items():
        for run in results:
            metrics = run["metrics"]
            arguments = run["arguments"]
            
            config = get_config_str(arguments)                
            for metric, value in metrics.items():
                if metric in metric_data:
                    metric_data[metric][config][model_name] = value
    
    return metric_data

def generate_comparison_tables(metric_data):
    """Generate comparison tables for each metric."""
    tables = []
    
    for metric, configs in sorted(metric_data.items()):
        # Skip metrics that don't have data or don't start with "Validation/"
        if not configs or not metric.startswith("Validation/"):
            continue
        
        table = f"## {metric}\n\n"
        
        # Get all model names
        model_names = set()
        for config_data in configs.values():
            model_names.update(config_data.keys())
        model_names = sorted(model_names)
        
        # Create header
        header = "|    Configuration   |" + "|".join(f"      {model}      " for model in model_names) + "|\n"
        separator = "|" + "-" * 20 + "|" + "|".join("-" * 19 for _ in model_names) + "|\n"
        
        table += header + separator
        
        # Sort configurations by envs and keep
        sorted_configs = sorted(configs.keys(), key=sort_config_key)
        
        # Add rows
        for config in sorted_configs:
            row = f"| {config} |"
            
            for model in model_names:
                value = configs[config].get(model, None)
                if value:
                    row += f" {value} |"
                else:
                    row += " N/A |"
            
            table += row + "\n"
        
        tables.append(table)
    
    return "\n".join(tables)

def main():
    # Define the result files (hardcoded as requested)
    result_files = {
        # "Classic Comb": "results_comb_classic.md",
        # "Classic Sep": "results_sep_classic.md",
        "Sparse Comb": "results_comb_sparse.md",
        "Sparse Sep": "results_sep_sparse.md",
    }
    config = {
        'envs': 'model.env_sys.steps.1.train_envs',
        'keep': 'model.env_sys.steps.1.train_keep_perc',
        # 'weight': 'model.train_sys.steps.0.loss.eta_loss_fn.weight',
        # 'threshold': 'model.train_sys.steps.0.model.threshold',
    }

    output_file_name = "comparison_sparse_sep_comb_test.md"
    
    # Check if files exist
    existing_files = {}
    for model, file_name in result_files.items():
        if os.path.exists(RESULTS_DIR / file_name):
            existing_files[model] = RESULTS_DIR / file_name
        else:
            print(f"Warning: {file_name} does not exist. Skipping.")
    
    if not existing_files:
        print("No input files found. Exiting.")
        return
    
    # Organize data by metric
    metric_data = organize_data_by_metric(existing_files, config)
    
    # Generate comparison tables
    tables = generate_comparison_tables(metric_data)
        
    output_path = os.path.join(RESULTS_DIR, output_file_name)
    with open(output_path, "w") as f:
        f.write(tables)
    
    print(f"Comparison tables written to {output_path}")

if __name__ == "__main__":
    main()