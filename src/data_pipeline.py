# Data Pipeline Script
import pandas as pd
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

# Example augmentation function
def augment_data(data):
    # Apply some transformation or augmentation
    augmented_data = data.copy()
    return augmented_data

# Saving augmented data
def save_data(data, file_path):
    data.to_csv(file_path, index=False)

# Usage
# data = load_data('data/iris.csv')
# augmented_data = augment_data(data)
# save_data(augmented_data, 'data/augmented_iris.csv')