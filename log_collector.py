#!/usr/bin/env python3

import os
import re
import glob
import argparse
from collections import defaultdict
from pathlib import Path

# Configure which metrics to include in the output - empty list means include all metrics
# Add specific metrics here if you want to filter, for example:
# WHITELIST_METRICS = ["Train/Transition Accuracy", "Validation/Transition Accuracy"]
WHITELIST_METRICS = []

def parse_log_file(log_file_path):
    """Parse a single log file and extract relevant information."""
    with open(log_file_path, 'r') as f:
        content = f.readlines()
    
    # Extract the lines we need
    first_100_lines = content[:100]
    last_200_lines = content[-200:] if len(content) > 200 else content
    
    # Parse parameters
    params = {}
    output_dir = None
    group_id = None
    
    for line in first_100_lines:
        line = line.strip()
        
        # Extract parameters
        if "INFO model" in line or "INFO n_trials" in line or "INFO trial_idx" in line:
            parts = line.split("INFO ", 1)[1].split("=", 1)
            if len(parts) == 2:
                key, value = parts
                params[key] = value
        
        # Extract output directory
        elif "INFO Output directory:" in line:
            output_dir = line.split("INFO Output directory:", 1)[1].strip()
        
        # Extract Group ID
        elif "INFO Group ID:" in line:
            group_id = line.split("INFO Group ID:", 1)[1].strip()
    
    # Parse results
    results = {}
    in_results_section = False
    
    for line in last_200_lines:
        line = line.strip()
        
        if "INFO --- Results ---" in line:
            in_results_section = True
            continue
        
        if "INFO --- Results End ---" in line:
            in_results_section = False
            continue
        
        if in_results_section:
            # Extract metric
            parts = line.split("INFO ", 1)[1]
            match = re.match(r'(.+): (.+) \(± (.+)\)', parts)
            if match:
                metric, value, std = match.groups()
                results[metric] = (value, std)
    
    return {
        'params': params,
        'output_dir': output_dir,
        'group_id': group_id,
        'results': results
    }

def generate_markdown(parsed_data_list):
    """Generate markdown content for a run with multiple log files."""
    md_content = ""
    
    for idx, (log_file, current_idx, max_idx, parsed_data) in enumerate(parsed_data_list):
        md_content += f"# Run {current_idx}\n"
        
        # Add output directory
        if parsed_data['output_dir']:
            md_content += f"Output_dir: {parsed_data['output_dir']}\n"
        
        # Add Group ID with link
        if parsed_data['group_id']:
            group_id = parsed_data['group_id']
            md_content += f"Group ID: [{group_id}](https://wandb.ai/a-ebersberger-tu-delft/Thesis/groups/{group_id}/workspace)\n"
        
        # Add arguments
        md_content += "## Arguments\n"
        for key, value in sorted(parsed_data['params'].items()):
            md_content += f"- {key}: {value}\n"
        
        # Add results
        md_content += "\n## Results\n"
        # Filter metrics if whitelist is not empty
        filtered_results = parsed_data['results']
        if WHITELIST_METRICS:
            filtered_results = {k: v for k, v in filtered_results.items() if k in WHITELIST_METRICS}
        
        # Sort metrics for consistent output
        for metric, (value, std) in sorted(filtered_results.items()):
            md_content += f"- {metric}: {value} (± {std})\n"
        
        # Add a blank line between runs, unless it's the last run
        if idx < len(parsed_data_list) - 1:
            md_content += "\n\n"
    
    return md_content

def clean_log_file(log_file_path):
    """Clean a log file by removing incomplete progress bar lines."""
    with open(log_file_path, 'r') as f:
        content = f.readlines()
    
    cleaned_content = []
    i = 0
    
    while i < len(content):
        line = content[i]
        
        # Check if this is a progress bar line
        match = re.search(r'(Epoch \d+ (?:Train|Valid)(?:\s+\([^)]+\))?:)', line)
        
        if match:
            # This is a progress bar line - find the last consecutive line for this epoch/phase
            epoch_phase = match.group(1)
            j = i
            while j + 1 < len(content) and re.search(re.escape(epoch_phase), content[j + 1]):
                j += 1
            
            # Keep only the last line of this sequence
            cleaned_content.append(content[j])
            i = j + 2  # Skip to the next line after this sequence
        else:
            # Not a progress bar line, keep it
            cleaned_content.append(line)
            i += 1
    
    # Write cleaned content back to file
    with open(log_file_path, 'w') as f:
        f.writelines(cleaned_content)
    
    print(f"Cleaned {log_file_path}")

def is_log_file_clean(log_file_path):
    """
    Check if a log file has been cleaned by looking for multiple 'Epoch 0 Valid' lines 
    in the first 100 lines.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        bool: False if the log contains two or more 'Epoch 0 Valid' lines (uncleaned),
              True otherwise (cleaned)
    """
    try:
        with open(log_file_path, 'r') as file:
            # Read first 100 lines
            lines = [file.readline() for _ in range(100)]
            
            # Count 'Epoch 0 Valid' occurrences
            epoch_0_valid_count = sum(1 for line in lines if 'Epoch 0 Valid' in line)
            
            # If there are 2 or more occurrences, the file is not cleaned
            return epoch_0_valid_count < 2
    except Exception as e:
        print(f"Error processing file {log_file_path}: {e}")
        return None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process log files and optionally clean them.')
    parser.add_argument('--clean', action='store_true', help='Clean log files by removing incomplete progress bars')
    args = parser.parse_args()
    
    # Find all log files
    log_files = glob.glob("data/logs/run_*.log")
    
    # Clean log files if requested
    if args.clean:
        print("Cleaning log files...")
        for log_file in log_files:
            if not is_log_file_clean(log_file):
                print(f"Cleaning {log_file}...")
                clean_log_file(log_file)
        print("Cleaning complete.")
    
    # Group log files by run date and time
    runs = defaultdict(list)
    
    for log_file in log_files:
        # Extract run date and time from filename
        match = re.search(r'run_(\d{8})_(\d{6})_(\d+)_of_(\d+)\.log', log_file)
        if match:
            run_date, run_time, current_idx, max_idx = match.groups()
            key = (run_date, run_time)
            runs[key].append((log_file, int(current_idx), int(max_idx)))
    
    # Process each run
    for (run_date, run_time), log_files_info in runs.items():
        results_file = Path(f"data/logs/run_{run_date}_{run_time}_results.md")
        
        if results_file.exists():
            print(f"Skipping {results_file} as it already exists.")
            continue
        
        # Sort log files by current_idx
        log_files_info.sort(key=lambda x: x[1])
        
        # Parse each log file
        parsed_data_list = []
        
        for log_file, current_idx, max_idx in log_files_info:
            parsed_data = parse_log_file(log_file)
            parsed_data_list.append((log_file, current_idx, max_idx, parsed_data))
        
        # Generate markdown content
        md_content = generate_markdown(parsed_data_list)
        
        # Write markdown file
        with open(results_file, 'w') as f:
            f.write(md_content)
        
        print(f"Generated {results_file}")

if __name__ == "__main__":
    main()