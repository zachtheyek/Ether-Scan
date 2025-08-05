"""
Test script to verify drift rate bias fix
Compares the original biased method vs the fixed uniform method
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def biased_drift_selection(n_samples: int = 10000) -> List[float]:
    """
    Original biased method from the author's code
    Uses (-1)**(int(random()*3+1)) which gives 2:1 negative bias
    """
    drift_rates = []
    
    for _ in range(n_samples):
        # Original biased code
        random_val = np.random.random()
        sign = (-1)**(int(random_val*3+1))
        
        # Generate drift magnitude
        drift_magnitude = np.random.uniform(0, 10)
        drift_rate = sign * drift_magnitude
        
        drift_rates.append(drift_rate)
    
    return drift_rates

def fixed_drift_selection(n_samples: int = 10000) -> List[float]:
    """
    Fixed method using uniform distribution
    Equal probability of positive and negative drift rates
    """
    drift_rates = []
    
    for _ in range(n_samples):
        # Fixed uniform method
        drift_rate = np.random.uniform(-10, 10)
        drift_rates.append(drift_rate)
    
    return drift_rates

def analyze_drift_distribution(drift_rates: List[float], method_name: str) -> dict:
    """
    Analyze the distribution of drift rates
    """
    drift_array = np.array(drift_rates)
    
    n_positive = np.sum(drift_array > 0)
    n_negative = np.sum(drift_array < 0)
    n_zero = np.sum(drift_array == 0)
    
    stats = {
        'method': method_name,
        'total': len(drift_rates),
        'positive': n_positive,
        'negative': n_negative,
        'zero': n_zero,
        'positive_ratio': n_positive / len(drift_rates),
        'negative_ratio': n_negative / len(drift_rates),
        'mean': np.mean(drift_array),
        'std': np.std(drift_array),
        'min': np.min(drift_array),
        'max': np.max(drift_array)
    }
    
    return stats

def plot_comparison(biased_drifts: List[float], fixed_drifts: List[float]):
    """
    Create comparison plots
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Histogram comparison
    bins = np.linspace(-10, 10, 50)
    
    ax1.hist(biased_drifts, bins=bins, alpha=0.5, label='Biased (Original)', 
             color='red', density=True)
    ax1.hist(fixed_drifts, bins=bins, alpha=0.5, label='Fixed (Uniform)', 
             color='blue', density=True)
    ax1.set_xlabel('Drift Rate (Hz/s)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Drift Rate Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sign distribution bar chart
    biased_stats = analyze_drift_distribution(biased_drifts, 'Biased')
    fixed_stats = analyze_drift_distribution(fixed_drifts, 'Fixed')
    
    categories = ['Negative', 'Positive']
    biased_counts = [biased_stats['negative'], biased_stats['positive']]
    fixed_counts = [fixed_stats['negative'], fixed_stats['positive']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax2.bar(x - width/2, biased_counts, width, label='Biased', color='red', alpha=0.7)
    ax2.bar(x + width/2, fixed_counts, width, label='Fixed', color='blue', alpha=0.7)
    ax2.set_xlabel('Drift Direction')
    ax2.set_ylabel('Count')
    ax2.set_title('Drift Direction Counts')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Ratio comparison
    ax3.bar(['Biased\n(Original)', 'Fixed\n(Uniform)'], 
            [biased_stats['negative_ratio'], fixed_stats['negative_ratio']],
            color=['red', 'blue'], alpha=0.7)
    ax3.axhline(y=0.5, color='green', linestyle='--', label='Expected (0.5)')
    ax3.set_ylabel('Negative Drift Ratio')
    ax3.set_title('Negative Drift Rate Ratio')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('drift_rate_bias_comparison.png', dpi=150)
    plt.show()

def main():
    """
    Run the drift rate bias test
    """
    print("Testing Drift Rate Bias Fix")
    print("=" * 50)
    
    # Generate samples
    n_samples = 10000
    print(f"Generating {n_samples} samples for each method...")
    
    biased_drifts = biased_drift_selection(n_samples)
    fixed_drifts = fixed_drift_selection(n_samples)
    
    # Analyze distributions
    biased_stats = analyze_drift_distribution(biased_drifts, 'Biased (Original)')
    fixed_stats = analyze_drift_distribution(fixed_drifts, 'Fixed (Uniform)')
    
    # Print results
    print("\n" + "=" * 50)
    print("BIASED METHOD (Original):")
    print(f"  Positive drift rates: {biased_stats['positive']} ({biased_stats['positive_ratio']:.1%})")
    print(f"  Negative drift rates: {biased_stats['negative']} ({biased_stats['negative_ratio']:.1%})")
    print(f"  Mean drift rate: {biased_stats['mean']:.3f} Hz/s")
    print(f"  Negative:Positive ratio: {biased_stats['negative']/biased_stats['positive']:.2f}:1")
    
    print("\n" + "=" * 50)
    print("FIXED METHOD (Uniform):")
    print(f"  Positive drift rates: {fixed_stats['positive']} ({fixed_stats['positive_ratio']:.1%})")
    print(f"  Negative drift rates: {fixed_stats['negative']} ({fixed_stats['negative_ratio']:.1%})")
    print(f"  Mean drift rate: {fixed_stats['mean']:.3f} Hz/s")
    print(f"  Negative:Positive ratio: {fixed_stats['negative']/fixed_stats['positive']:.2f}:1")
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  Biased method has {biased_stats['negative']/biased_stats['positive']:.2f}x more negative drifts")
    print(f"  Fixed method has {fixed_stats['negative']/fixed_stats['positive']:.2f}x ratio (expected: 1.0)")
    
    # Create plots
    print("\nGenerating comparison plots...")
    plot_comparison(biased_drifts, fixed_drifts)
    
    # Statistical test
    from scipy import stats
    chi2, p_value = stats.chisquare([fixed_stats['negative'], fixed_stats['positive']])
    print(f"\nChi-square test for uniform distribution (fixed method):")
    print(f"  p-value: {p_value:.4f} (>0.05 indicates uniform distribution)")

if __name__ == "__main__":
    main()
