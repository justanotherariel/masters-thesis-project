from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from typing import Dict, Tuple, List
from scipy import stats
import math

DATA_DIR = Path("data")


def test_statistical_significance(mean1, std1, mean2, std2, n1=10, n2=10, alpha=0.05):
    """
    Test statistical significance between two models using Welch's t-test.
    
    Returns:
        tuple: (is_significant, p_value)
    """
    # Perform two-sample t-test using summary statistics
    t_stat, p_value = stats.ttest_ind_from_stats(
        mean1, std1, n1,
        mean2, std2, n2,
        equal_var=False  # Welch's t-test
    )
    
    return p_value < alpha, p_value


def parse_run_file(file_path: Path) -> dict:
    """
    Parse a run summary file and extract structured data.
    
    Args:
        file_path: Path to the run summary file
        
    Returns:
        Dictionary with structured data from all runs
    """
    # Read file content
    content = file_path.read_text()
    
    # Find all run blocks
    run_pattern = r'# Run (\d+)(.*?)(?=# Run \d+|\Z)'
    runs = re.findall(run_pattern, content, re.DOTALL)
    
    all_runs = {}
    
    for run_num, run_content in runs:
        run_data = {}
        
        # Extract output directory
        output_match = re.search(r'Output_dir: (.+)', run_content)
        if output_match:
            run_data['output_dir'] = output_match.group(1).strip()
        
        # Extract group ID and link
        group_match = re.search(r'Group ID: \[(.+?)\]\((.+?)\)', run_content)
        if group_match:
            run_data['group_id'] = group_match.group(1)
            run_data['group_link'] = group_match.group(2)
        
        # Extract arguments
        arguments = {}
        args_section = re.search(r'## Arguments(.*?)(?=##|\Z)', run_content, re.DOTALL)
        if args_section:
            args_lines = args_section.group(1).strip().split('\n')
            for line in args_lines:
                if line.startswith('- '):
                    parts = line[2:].split(': ', 1)
                    if len(parts) == 2:
                        key, value = parts
                        arguments[key] = value
        run_data['arguments'] = arguments
        
        # Extract results
        results = {}
        results_section = re.search(r'## Results(.*?)(?=##|\Z)', run_content, re.DOTALL)
        if results_section:
            results_lines = results_section.group(1).strip().split('\n')
            for line in results_lines:
                if line.startswith('- '):
                    parts = line[2:].split(': ', 1)
                    if len(parts) == 2:
                        key, value = parts
                        # Parse value with mean and std dev
                        value_match = re.match(r'([\d\.]+) \(± ([\d\.]+)\)', value)
                        if value_match:
                            mean, std = value_match.groups()
                            results[key] = {
                                'value': float(mean),
                                'std_dev': float(std)
                            }
        run_data['results'] = results
        
        # Add run to dictionary
        all_runs[f'run_{run_num}'] = run_data
    
    return all_runs


def plot_model_comparison_by_grid_count(model1_data: Dict, model2_data: Dict, model3_data: Dict,
                                      model1_name: str, model2_name: str, model3_name: str,
                                      metric: str = "Validation/Transition Accuracy",
                                      output_prefix: str = None,
                                      show_significance: bool = True,
                                      alpha: float = 0.05) -> List[plt.Figure]:
    """
    Create separate bar plots, one for each environment grid count,
    comparing three models across different data percentages.
    
    Args:
        model1_data: Dictionary with data for the first model (from parse_run_file)
        model2_data: Dictionary with data for the second model (from parse_run_file)
        model3_data: Dictionary with data for the third model (from parse_run_file)
        model1_name: Name of the first model for the legend
        model2_name: Name of the second model for the legend
        model3_name: Name of the third model for the legend (reference model for significance testing)
        metric: Which metric to plot (default: "Validation/Transition Accuracy")
        output_prefix: Optional prefix for saving figures (will append env_count)
        show_significance: Whether to show statistical significance indicators (default: True)
        alpha: Significance level for statistical tests (default: 0.05)
        
    Returns:
        List of Matplotlib figures, one for each environment grid count
    """
    # Extract and organize data for plotting
    organized_data = organize_data_for_plot(
        model1_data, model2_data, model3_data,
        model1_name, model2_name, model3_name, 
        metric
    )
    
    # Get unique environment counts and percentages
    env_counts = sorted(set(item['env_count'] for item in organized_data))
    percentages = sorted(set(item['data_perc'] for item in organized_data))
    
    # Set colors for the models
    model_colors = {
        model1_name: '#3366CC',  # Blue
        model2_name: '#DC3912',  # Red
        model3_name: '#109618'   # Green
    }
    
    # Create a list to store all figures
    figures = []
    
    # Find global y-axis limits for consistency across plots
    all_values = [item['value'] for item in organized_data]
    all_std_devs = [item['std_dev'] for item in organized_data]
    global_y_min = max(0, min(all_values) - max(all_std_devs) * 2) * 0.95
    global_y_max = min(1.1, max(all_values) + max(all_std_devs) * 2) * 1.05
    
    # Increase default font size for all plot elements
    plt.rcParams.update({'font.size': 14})
    
    # Create a separate plot for each environment grid count
    for env_count in env_counts:
        # Filter data for this environment count
        env_data = [item for item in organized_data if item['env_count'] == env_count]
        
        # Set up the figure and axes
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set bar width and positions
        bar_width = 0.25
        group_positions = np.arange(len(percentages))
        
        # Plot the bars for each model
        for i, model_name in enumerate([model1_name, model2_name, model3_name]):
            # Extract data points for this model
            model_points = [next((item for item in env_data 
                                if item['data_perc'] == perc 
                                and item['model'] == model_name), None) 
                          for perc in percentages]
            
            # Prepare data for plotting
            values = []
            errors = []
            positions = []
            
            for j, point in enumerate(model_points):
                if point:
                    values.append(point['value'])
                    errors.append(point['std_dev'])
                    positions.append(group_positions[j] - bar_width + i*bar_width)
            
            # Plot the bars with error bars that have horizontal caps
            bars = ax.bar(positions, values, bar_width,
                         yerr=errors,
                         error_kw={'capsize': 5},
                         color=model_colors[model_name],
                         label=model_name)
            
            # Add value labels above the error bars
            for pos, val, err in zip(positions, values, errors):
                label_height = val + err + 0.015
                ax.text(pos, label_height, f"{val:.3f}", 
                       ha='center', va='bottom', rotation=0, size=14,
                       fontweight='bold')
                
                # Add significance indicators if enabled and not the reference model
                if show_significance and model_name != model3_name:
                    # Find corresponding sparse transformer (model3) data
                    perc_idx = positions.index(pos) if pos in positions else None
                    if perc_idx is not None:
                        sparse_point = next((item for item in env_data 
                                           if item['data_perc'] == percentages[perc_idx] 
                                           and item['model'] == model3_name), None)
                        
                        if sparse_point:
                            # Test statistical significance
                            is_significant, p_value = test_statistical_significance(
                                val, err, 
                                sparse_point['value'], sparse_point['std_dev'],
                                n1=10, n2=10, alpha=alpha
                            )
                            
                            if is_significant:
                                # Add asterisk above the value label
                                asterisk_height = label_height + 0.04
                                significance_marker = '*'
                                if p_value < 0.001:
                                    significance_marker = '***'
                                elif p_value < 0.01:
                                    significance_marker = '**'
                                
                                ax.text(pos, asterisk_height, significance_marker, 
                                       ha='center', va='bottom', size=16,
                                       fontweight='bold', color='black' if val < sparse_point['value'] else 'red')
        
        # Set the x-axis labels and ticks
        ax.set_xticks(group_positions)
        ax.set_xticklabels([f"{int(perc*100)}%" for perc in percentages], fontsize=14)
        
        # Set consistent y-axis range across all plots
        ax.set_ylim(global_y_min, global_y_max)
        ax.tick_params(axis='y', labelsize=14)
        
        # Add grid lines for readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set labels and title with increased font sizes
        ax.set_xlabel('Training Data Percentage', fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        # ax.set_title(f'Model Comparison with {env_count} Environment Grids', fontsize=18)
        
        # Add legend with larger font
        legend = ax.legend(loc='lower right', fontsize=14)
        
        # Add significance notation to legend if enabled
        if show_significance:
            # Create a text box for significance notation
            sig_text = "* p < 0.05, ** p < 0.01, *** p < 0.001"
            ax.text(0.02, 0.95, sig_text, transform=ax.transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path is provided
        if output_prefix:
            output_path = DATA_DIR / f"{output_prefix}_{env_count}envs.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Add to list of figures
        figures.append(fig)
    
    return figures


def organize_data_for_plot(model1_data: Dict, model2_data: Dict, model3_data: Dict,
                           model1_name: str, model2_name: str, model3_name: str,
                           metric: str) -> List[Dict]:
    """
    Organize the data from three models into a format suitable for plotting.
    
    Args:
        model1_data: Dictionary with data for the first model
        model2_data: Dictionary with data for the second model
        model3_data: Dictionary with data for the third model
        model1_name: Name of the first model
        model2_name: Name of the second model
        model3_name: Name of the third model
        metric: Which metric to extract
        
    Returns:
        List of dictionaries with organized data
    """
    organized_data = []
    
    # Process first model data
    for run_key, run_info in model1_data.items():
        if 'arguments' in run_info and 'results' in run_info:
            if 'model.env_sys.steps.1.train_envs' in run_info['arguments'] and \
               'model.env_sys.steps.1.train_keep_perc' in run_info['arguments'] and \
               metric in run_info['results']:
                
                env_count = int(run_info['arguments']['model.env_sys.steps.1.train_envs'])
                data_perc = float(run_info['arguments']['model.env_sys.steps.1.train_keep_perc'])
                value = run_info['results'][metric]['value']
                std_dev = run_info['results'][metric]['std_dev']
                
                organized_data.append({
                    'model': model1_name,
                    'env_count': env_count,
                    'data_perc': data_perc,
                    'value': value,
                    'std_dev': std_dev
                })
    
    # Process second model data
    for run_key, run_info in model2_data.items():
        if 'arguments' in run_info and 'results' in run_info:
            if 'model.env_sys.steps.1.train_envs' in run_info['arguments'] and \
               'model.env_sys.steps.1.train_keep_perc' in run_info['arguments'] and \
               metric in run_info['results']:
                
                env_count = int(run_info['arguments']['model.env_sys.steps.1.train_envs'])
                data_perc = float(run_info['arguments']['model.env_sys.steps.1.train_keep_perc'])
                value = run_info['results'][metric]['value']
                std_dev = run_info['results'][metric]['std_dev']
                
                organized_data.append({
                    'model': model2_name,
                    'env_count': env_count,
                    'data_perc': data_perc,
                    'value': value,
                    'std_dev': std_dev
                })
    
    # Process third model data
    for run_key, run_info in model3_data.items():
        if 'arguments' in run_info and 'results' in run_info:
            if 'model.env_sys.steps.1.train_envs' in run_info['arguments'] and \
               'model.env_sys.steps.1.train_keep_perc' in run_info['arguments'] and \
               metric in run_info['results']:
                
                env_count = int(run_info['arguments']['model.env_sys.steps.1.train_envs'])
                data_perc = float(run_info['arguments']['model.env_sys.steps.1.train_keep_perc'])
                value = run_info['results'][metric]['value']
                std_dev = run_info['results'][metric]['std_dev']
                
                organized_data.append({
                    'model': model3_name,
                    'env_count': env_count,
                    'data_perc': data_perc,
                    'value': value,
                    'std_dev': std_dev
                })
    
    return organized_data


# Example usage:
if __name__ == "__main__":
    # Load data from three different model runs
    model1_file = Path("model-data/results_unet.md")
    model2_file = Path("model-data/results_comb_classic.md")
    model3_file = Path("model-data/results_comb_sparse.md")
    
    # model1_file = Path("model-data/results_unet.md")
    # model2_file = Path("model-data/results_comb_classic_sin1d.md")
    # model3_file = Path("model-data/results_comb_sparse_sin1d.md")

    model1_data = parse_run_file(model1_file)
    model2_data = parse_run_file(model2_file)
    model3_data = parse_run_file(model3_file)
    
    # Create and save the separate plots for each data percentage with significance indicators
    figs = plot_model_comparison_by_grid_count(
        model1_data,
        model2_data, 
        model3_data,
        "U-Net",
        "Transformer",
        "Sparse Transformer",
        metric="Validation/Transition Accuracy",
        output_prefix="model_comparison",
        show_significance=True,  # Enable significance testing
        alpha=0.05  # Significance level
    )
