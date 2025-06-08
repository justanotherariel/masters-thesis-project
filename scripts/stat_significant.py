import numpy as np
from scipy import stats
import math

def test_statistical_significance(mean1, std1, mean2, std2, n1=10, n2=10, alpha=0.05):
    """
    Test statistical significance between two models using a two-sample t-test.
    
    Parameters:
    -----------
    mean1 : float
        Mean performance of model 1
    std1 : float
        Standard deviation of model 1
    mean2 : float
        Mean performance of model 2
    std2 : float
        Standard deviation of model 2
    n1 : int
        Number of runs for model 1 (default: 10)
    n2 : int
        Number of runs for model 2 (default: 10)
    alpha : float
        Significance level (default: 0.05 for 95% confidence)
    
    Returns:
    --------
    dict: Dictionary containing test results
    """
    
    # Perform two-sample t-test using summary statistics
    t_stat, p_value = stats.ttest_ind_from_stats(
        mean1, std1, n1,
        mean2, std2, n2,
        equal_var=False  # Welch's t-test (doesn't assume equal variances)
    )
    
    # Calculate effect size (Cohen's d)
    pooled_std = math.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    cohens_d = abs(mean1 - mean2) / pooled_std
    
    # Determine significance
    is_significant = p_value < alpha
    
    # Calculate confidence interval for the difference in means
    diff_mean = mean1 - mean2
    se_diff = math.sqrt(std1**2/n1 + std2**2/n2)
    df = (std1**2/n1 + std2**2/n2)**2 / ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
    t_critical = stats.t.ppf(1 - alpha/2, df)
    ci_lower = diff_mean - t_critical * se_diff
    ci_upper = diff_mean + t_critical * se_diff
    
    results = {
        'model1_mean': mean1,
        'model1_std': std1,
        'model2_mean': mean2,
        'model2_std': std2,
        'difference': diff_mean,
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'alpha': alpha,
        'cohens_d': cohens_d,
        'confidence_interval': (ci_lower, ci_upper),
        'degrees_of_freedom': df
    }
    
    return results

def print_results(results):
    """Print formatted results of the statistical test."""
    print("=" * 60)
    print("STATISTICAL SIGNIFICANCE TEST RESULTS")
    print("=" * 60)
    print(f"Model 1: Mean = {results['model1_mean']:.4f}, Std = {results['model1_std']:.4f}")
    print(f"Model 2: Mean = {results['model2_mean']:.4f}, Std = {results['model2_std']:.4f}")
    print(f"Difference (Model 1 - Model 2): {results['difference']:.4f}")
    print("-" * 40)
    print(f"t-statistic: {results['t_statistic']:.4f}")
    print(f"p-value: {results['p_value']:.6f}")
    print(f"Degrees of freedom: {results['degrees_of_freedom']:.2f}")
    print(f"Significance level (α): {results['alpha']}")
    print("-" * 40)
    
    if results['is_significant']:
        print("✅ RESULT: Statistically significant difference detected!")
        print(f"   The p-value ({results['p_value']:.6f}) is less than α ({results['alpha']})")
    else:
        print("❌ RESULT: No statistically significant difference detected.")
        print(f"   The p-value ({results['p_value']:.6f}) is greater than α ({results['alpha']})")
    
    print(f"\nEffect size (Cohen's d): {results['cohens_d']:.4f}")
    if results['cohens_d'] < 0.2:
        effect_size = "negligible"
    elif results['cohens_d'] < 0.5:
        effect_size = "small"
    elif results['cohens_d'] < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    print(f"Effect size interpretation: {effect_size}")
    
    print(f"\n95% Confidence Interval for difference: [{results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f}]")
    print("=" * 60)

# Example usage
if __name__ == "__main__":
    # Example data - replace with your actual values
    model1_mean = 0.6876  # Replace with your model 1 average
    model1_std = 0.1042   # Replace with your model 1 standard deviation
    
    model2_mean = 0.7998  # Replace with your model 2 average
    model2_std = 0.0286   # Replace with your model 2 standard deviation
    
    # Run the test
    results = test_statistical_significance(
        model1_mean, model1_std, 
        model2_mean, model2_std, 
        n1=10, n2=10, 
        alpha=0.05
    )
    
    # Print results
    print_results(results)
    
    # You can also access individual results:
    print(f"\nQuick answer: {'Significant' if results['is_significant'] else 'Not significant'}")
    print(f"P-value: {results['p_value']:.6f}")