"""
Loads inference results & summarizes high-confidence detections
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = pd.read_csv('/output/seti/detections.csv')

print(f"Total detections: {len(results)}")
print(f"\nDetection summary:")
print(results['classification'].value_counts())

# High confidence detections
high_conf = results[results['confidence'] > 0.9]
print(f"\nHigh confidence detections: {len(high_conf)}")

if len(high_conf) > 0:
    print("\nTop 5 detections:")
    print(high_conf.head()[['frequency_MHz', 'confidence', 'classification']])

# Plot confidence distribution
plt.figure(figsize=(10, 6))
plt.hist(results['confidence'], bins=50, alpha=0.7)
plt.xlabel('Confidence')
plt.ylabel('Count')
plt.title('Detection Confidence Distribution')
plt.savefig('/output/seti/confidence_dist.png')
