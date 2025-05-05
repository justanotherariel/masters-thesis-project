import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import pickle
from tqdm import tqdm
from pathlib import Path

# Configuration Constants
ENTITY = "a-ebersberger-tu-delft"  # Your wandb username or team name
PROJECT = "Thesis"  # Your project name

CACHE_FILE = Path("cache.pkl")  # Path to the cache file

def get_group_metric_data(group_id, metric_name):
    """
    Download metric data for all runs in a specified group.
    
    Args:
        group_id (str): The group ID to filter runs by
        metric_name (str): The name of the metric to download (e.g., 'Train/Validation Accuracy')
        
    Returns:
        list of tuples: Each tuple contains (step_indices, metric_values) for a run
    """
    # Initialize the API
    api = wandb.Api()
    
    # Get runs from the specified project and filter by group
    runs = api.runs(f"{ENTITY}/{PROJECT}", {"group": group_id})
    
    # List to store metrics for each run
    all_run_data = []
    
    # Iterate through runs and get the metric history for each
    for run in tqdm(runs, desc=f"Downloading data for group {group_id}", unit="run"):
        # Get the metrics for this run
        steps = []
        metric_values = []
        step_idx = 0
        
        for row in run.scan_history(keys=["_step", metric_name]):
            # Use the provided step if available, otherwise increment counter
            if "_step" in row:
                step_idx = row["_step"]
            
            if metric_name in row:
                steps.append(step_idx)
                metric_values.append(row[metric_name])
            
            # Increment for next iteration if _step not provided
            if "_step" not in row:
                step_idx += 1
        
        # Add the data for this run if we have any points
        if steps:
            all_run_data.append((steps, metric_values))
    
    return all_run_data

def plot_group_metrics(group_data_dict, metric_name, save_path=None):
    """
    Plot average metrics with standard deviation for multiple groups,
    handling interpolation for missing values.
    
    Args:
        group_data_dict (dict): Dictionary mapping group IDs to data from get_group_metric_data()
        metric_name (str): Name of the metric (used for the y-axis label)
        save_path (str, optional): Path to save the figure. If None, the plot is displayed instead.
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']  # Add more if needed
    
    for i, (group_id, group_data) in enumerate(group_data_dict.items()):
        color = colors[i % len(colors)]
        
        if not group_data:
            print(f"No data available for group {group_id}")
            continue
            
        # Find the maximum step across all runs to determine the x-axis range
        max_step = 0
        for steps, _ in group_data:
            if steps and max(steps) > max_step:
                max_step = max(steps)
        
        # Create a common x-axis for all runs
        common_steps = np.arange(max_step + 1)
        
        # Interpolate each run's data to the common step range
        interpolated_runs = []
        for steps, values in group_data:
            # Filter out None values
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            valid_steps = [steps[i] for i in valid_indices]
            valid_values = [values[i] for i in valid_indices]
            
            if len(valid_steps) < 2:  # Need at least 2 points for interpolation
                print(f"Skipping a run in {group_id} with insufficient valid data points")
                continue
                
            # Interpolate to the common step range
            interp_values = np.interp(
                common_steps,
                valid_steps,
                valid_values,
                left=valid_values[0],    # Use first value for steps before first measurement
                right=valid_values[-1]   # Use last value for steps after last measurement
            )
            
            interpolated_runs.append(interp_values)
        
        if not interpolated_runs:
            print(f"No valid data available for group {group_id} after interpolation")
            continue
            
        # Convert to numpy array for easier calculations
        data_array = np.array(interpolated_runs)
        
        # Calculate mean and std dev at each step
        mean_values = np.mean(data_array, axis=0)
        std_values = np.std(data_array, axis=0)
        
        # Plot the mean line
        plt.plot(common_steps, mean_values, color=color, label=f"{group_id} (n={len(interpolated_runs)})")
        
        # Shade the standard deviation area
        rgba_color = to_rgba(color, 0.2)  # Transparent version of the color
        plt.fill_between(common_steps, mean_values - std_values, mean_values + std_values, color=rgba_color)
    
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} by Group (with ±1σ)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    metric_name = "Validation/Transition Accuracy"
    
    # Download data for two groups
    group1_id = "5k3jal0y"
    group2_id = "wklveq3u"
    
    cache = None
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
    
    if cache is None or group1_id not in cache:
        data = get_group_metric_data(group1_id, metric_name)
        cache = {} if cache is None else cache
        cache[group1_id] = data
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
            
    if group2_id not in cache:
        data = get_group_metric_data(group2_id, metric_name)
        cache[group2_id] = data
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
        
    # Plot the data
    plot_group_metrics({
        "Transformer": cache[group1_id],
        "Sparse Transformer": cache[group2_id]
    }, metric_name, save_path="group_metrics_plot.png")