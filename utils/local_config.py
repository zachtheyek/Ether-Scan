# Update config.py or create local_config.py
import os

# Update paths
DATA_PATH = '/data/seti'
MODEL_PATH = '/models/seti'
OUTPUT_PATH = '/output/seti'

# Training data files
TRAINING_FILES = [
    'real_filtered_LARGE_HIP110750.npy',
    'real_filtered_LARGE_HIP13402.npy', 
    'real_filtered_LARGE_HIP8497.npy'
]

# Test data files
TEST_FILES = [
    'real_filtered_LARGE_testHIP15638.npy',
    'real_filtered_LARGE_testHIP83043.npy'
]

# Update the main.py load_background_data function
def load_background_data(data_path: str) -> np.ndarray:
    backgrounds = []
    for filename in TRAINING_FILES:
        filepath = os.path.join(data_path, 'training', filename)
        if os.path.exists(filepath):
            data = np.load(filepath)
            # Take a subset if files are very large
            # Original code uses slices like [8000:] or [:4000]
            if filename == 'real_filtered_LARGE_HIP110750.npy':
                data = data[8000:]  # Skip first 8000 snippets
            else:
                data = data[:4000]  # Use first 4000 snippets
            backgrounds.append(data)
            print(f"Loaded {filename}: shape {data.shape}")
    
    return np.vstack(backgrounds)
