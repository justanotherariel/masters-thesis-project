import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

MODEL_DATA_PATH = Path("model-data")
DATA_PATH = Path("data")

def extract_eta_data(file_path, run_id, dataset):
    """
    Extract Eta statistics from the file for a specific run and dataset.
    
    Args:
        file_path: Path to the input file
        run_id: Run number (e.g., 1, 2, 3...)
        dataset: One of 'Test', 'Train', or 'Validation'
    
    Returns:
        Dictionary with eta numbers as keys and (value, stderr) tuples
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the specific run section
    run_pattern = f"# Run {run_id}\n"
    run_start = content.find(run_pattern)
    if run_start == -1:
        raise ValueError(f"Run {run_id} not found")
    
    # Find the next run or end of file
    next_run_start = content.find(f"# Run {run_id + 1}\n", run_start)
    if next_run_start == -1:
        run_content = content[run_start:]
    else:
        run_content = content[run_start:next_run_start]
    
    # Extract Eta data for the specified dataset
    eta_pattern = rf"- {dataset}/Eta (\d+\+?): ([\d.]+) \(± ([\d.]+)\)"
    eta_matches = re.findall(eta_pattern, run_content)

    # Find eta_loss_fn.weight
    eta_loss_fn_pattern = rf"- model.train_sys.steps.0.loss.eta_loss_fn.weight: ([\d.]+)"
    eta_loss_fn_match = re.search(eta_loss_fn_pattern, run_content)
    if eta_loss_fn_match:
        eta_loss_fn_weight = float(eta_loss_fn_match.group(1))
        print(f"Eta loss function weight: {eta_loss_fn_weight}")
    else:
        print(f"No Eta loss function weight found for Run {run_id}")

    # Find threshold value
    threshold_pattern = rf"- model.train_sys.steps.0.model.threshold: ([\d.]+)"
    threshold_match = re.search(threshold_pattern, run_content)
    if threshold_match:
        threshold_value = float(threshold_match.group(1))
        print(f"Threshold: {threshold_value}")
    else:
        print(f"No threshold found for Run {run_id}")

    # Find Eta Sum
    eta_sum_pattern = rf"- {dataset}/Eta Sum: ([\d.]+) \(± ([\d.]+)\)"
    eta_sum_match = re.search(eta_sum_pattern, run_content)
    if eta_sum_match:
        eta_sum_value = float(eta_sum_match.group(1))
        eta_sum_stderr = float(eta_sum_match.group(2))
        print(f"{dataset} Eta Sum: {eta_sum_value} ± {eta_sum_stderr}")
    else:
        print(f"No {dataset} Eta Sum found for Run {run_id}")

    # Find Transition Accuracy
    val_pattern = rf"- {dataset}/Transition Accuracy: ([\d.]+) \(± ([\d.]+)\)"
    val_match = re.search(val_pattern, run_content)
    if val_match:
        val_accuracy = float(val_match.group(1))
        val_stderr = float(val_match.group(2))
        print(f"{dataset} Transition Accuracy: {val_accuracy} ± {val_stderr}")
    else:
        print(f"No {dataset} Transition Accuracy found for Run {run_id}")
    
    eta_data = {}
    for match in eta_matches:
        eta_num = match[0]
        value = float(match[1])
        stderr = float(match[2])
        eta_data[eta_num] = (value, stderr)
    
    return eta_data

def plot_eta_bars(eta_data, run_id, dataset):
    """
    Create a bar plot of Eta statistics.
    
    Args:
        eta_data: Dictionary with eta numbers and (value, stderr) tuples
        run_id: Run number for the title
        dataset: Dataset name for the title
    """
    # Sort eta numbers numerically (handle '123+' specially)
    eta_nums = []
    for key in eta_data.keys():
        if key.endswith('+'):
            eta_nums.append((999, key))  # Put '123+' at the end
        else:
            eta_nums.append((int(key), key))
    eta_nums.sort()
    
    # Extract sorted data
    labels = [num[1] for num in eta_nums]
    values = [eta_data[label][0] for label in labels]
    errors = [eta_data[label][1] for label in labels]
    
    # Clip error bars to prevent going below 0
    lower_errors = []
    upper_errors = []
    for v, e in zip(values, errors):
        lower_errors.append(min(e, v))  # Don't let error bar go below 0
        upper_errors.append(e)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 8))
    
    x = np.arange(len(labels))
    bars = ax.bar(x, values, yerr=[lower_errors, upper_errors], capsize=3, alpha=0.7, edgecolor='black')
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Customize the plot
    ax.set_xlabel('Eta Sum', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(f'Eta Sum Spread - 4 Train Envs, 20% Data', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for i, (v, e) in enumerate(zip(values, upper_errors)):
        if v > 0:  # Only label non-zero values
            ax.text(i, v + e + max(values) * 0.01, f'{v:.1f}', 
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(DATA_PATH / f"eta_bars_run_{run_id}_{dataset}.png")
    

def main():
    # Configuration variables
    file_path = MODEL_DATA_PATH / "results_comb_sparse-sparsity_study.md"
    dataset = "Test"  # Options: "Test", "Train", "Validation"
    
    # Extract and plot data
    for run_id in range(1, 17):
        print(f"Processing Run {run_id}...")
        try:
            eta_data = extract_eta_data(file_path, run_id, dataset)
            if eta_data:
                plot_eta_bars(eta_data, run_id, dataset)
                print(f"Found {len(eta_data)} Eta values")
            else:
                print(f"No Eta data found for Run {run_id} - {dataset}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 40)

if __name__ == "__main__":
    main()