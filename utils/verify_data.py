# verify_data.py
import numpy as np
import os

def verify_data_format(data_path):
    files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    
    for file in files[:3]:  # Check first 3 files
        filepath = os.path.join(data_path, file)
        data = np.load(filepath)
        print(f"\nFile: {file}")
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Min value: {data.min()}")
        print(f"Max value: {data.max()}")
        print(f"File size: {os.path.getsize(filepath) / 1e9:.2f} GB")

print("=== Training Data ===")
verify_data_format('/data/seti/training/')

print("\n=== Test Data ===")
verify_data_format('/data/seti/testing/')
