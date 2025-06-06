#!/usr/bin/env python3
"""
Script to extract and display metrics from experimental results file.

Usage:
    python extract_metrics.py <file_path> <metric1> [metric2] [metric3] ...

Example:
    python extract_metrics.py results.txt "Test/Agent Accuracy" "Test/Agent Forward Accuracy"
"""

import sys
import re
from typing import List, Dict, Tuple, Optional


def parse_run_data(content: str) -> List[Dict]:
    """Parse the content and extract run data."""
    runs = []
    
    # Split by run headers
    run_sections = re.split(r'^# Run \d+', content, flags=re.MULTILINE)
    
    for i, section in enumerate(run_sections[1:], 1):  # Skip first empty section
        run_data = {'run_number': i, 'arguments': {}, 'results': {}}
        
        # Extract arguments section
        args_match = re.search(r'## Arguments\n(.*?)(?=## Results)', section, re.DOTALL)
        if args_match:
            args_text = args_match.group(1)
            for line in args_text.strip().split('\n'):
                if line.startswith('- '):
                    if ':' in line:
                        key, value = line[2:].split(':', 1)
                        run_data['arguments'][key.strip()] = value.strip()
        
        # Extract results section
        results_match = re.search(r'## Results\n(.*)', section, re.DOTALL)
        if results_match:
            results_text = results_match.group(1)
            for line in results_text.strip().split('\n'):
                if line.startswith('- '):
                    if ':' in line:
                        key, value = line[2:].split(':', 1)
                        run_data['results'][key.strip()] = value.strip()
        
        runs.append(run_data)
    
    return runs


def extract_metric_value(metric_string: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract mean and std from metric string like '0.4184 (± 0.0734)'."""
    if not metric_string or metric_string == 'nan (± nan)':
        return None, None
    
    # Handle special cases
    if 'nan' in metric_string.lower():
        return None, None
    
    # Extract mean and std using regex
    match = re.search(r'([\d.-]+)\s*\(±\s*([\d.-]+)\)', metric_string)
    if match:
        try:
            mean = float(match.group(1))
            std = float(match.group(2))
            return mean, std
        except ValueError:
            return None, None
    
    # Try to extract just the number if no std deviation
    try:
        return float(metric_string.strip()), None
    except ValueError:
        return None, None


def print_metrics(runs: List[Dict], metrics: List[str], show_args: List[str] = None):
    """Print the specified metrics for all runs."""
    if not runs:
        print("No runs found in the file.")
        return
    
    # Prepare all data first to calculate column widths
    all_rows = []
    
    # Prepare header
    header = ["Run"]
    if show_args:
        header.extend([x.split('.')[-1] for x in show_args])  # Show only last part of each argument
    header.extend(metrics)
    all_rows.append(header)
    
    # Prepare data rows
    for run in runs:
        row = [str(run['run_number'])]
        
        # Add argument values if requested
        if show_args:
            for arg in show_args:
                value = run['arguments'].get(arg, 'N/A')
                row.append(str(value))
        
        # Add metric values
        for metric in metrics:
            if metric in run['results']:
                mean, std = extract_metric_value(run['results'][metric])
                if mean is not None:
                    if std is not None:
                        metric_str = f"{mean:.4f} (±{std:.4f})"
                    else:
                        metric_str = f"{mean:.4f}"
                else:
                    metric_str = "N/A"
            else:
                metric_str = "N/A"
            row.append(metric_str)
        
        all_rows.append(row)
    
    # Calculate column widths
    col_widths = []
    for col_idx in range(len(header)):
        max_width = max(len(str(row[col_idx])) for row in all_rows)
        col_widths.append(max_width + 2)  # Add 2 for padding
    
    # Print formatted table
    for row_idx, row in enumerate(all_rows):
        formatted_row = []
        for col_idx, cell in enumerate(row):
            formatted_row.append(str(cell).ljust(col_widths[col_idx]))
        print("".join(formatted_row))
        
        # Print separator after header
        if row_idx == 0:
            print("-" * sum(col_widths))


def print_available_metrics(runs: List[Dict]):
    """Print all available metrics from the first run."""
    if not runs:
        print("No runs found.")
        return
    
    print("Available metrics:")
    for metric in sorted(runs[0]['results'].keys()):
        print(f"  - {metric}")


def print_available_arguments(runs: List[Dict]):
    """Print all available arguments from the first run."""
    if not runs:
        print("No runs found.")
        return
    
    print("Available arguments:")
    for arg in sorted(runs[0]['arguments'].keys()):
        print(f"  - {arg}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_metrics.py <file_path> <metric1> [metric2] ...")
        print("       python extract_metrics.py <file_path> --list-metrics")
        print("       python extract_metrics.py <file_path> --list-args")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    runs = parse_run_data(content)
    
    if len(sys.argv) == 3 and sys.argv[2] == '--list-metrics':
        print_available_metrics(runs)
        return
    
    if len(sys.argv) == 3 and sys.argv[2] == '--list-args':
        print_available_arguments(runs)
        return
    
    if len(sys.argv) < 3:
        print("Please specify at least one metric name.")
        print("Use --list-metrics to see available metrics.")
        sys.exit(1)
    
    metrics = sys.argv[2:]
    
    # Hardcoded arguments to display (modify these as needed)
    show_args = [
        'model.env_sys.steps.1.train_envs',
        'model.env_sys.steps.1.train_keep_perc'
    ]
    
    print(f"Extracting metrics from {len(runs)} runs:")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Arguments shown: {', '.join(show_args)}")
    print()
    
    print_metrics(runs, metrics, show_args)


if __name__ == "__main__":
    main()