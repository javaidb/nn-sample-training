import pandas as pd
import torch

def load_data(file_name):
    """Load battery data from a CSV file."""
    file_path = f"../data/{file_name}.csv"
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess the data (e.g., normalize, handle missing values)."""
    # Fill missing values with forward fill
    data.fillna(method='ffill', inplace=True)
    
    # Normalize features (mean normalization)
    normalized_data = (data - data.mean()) / data.std()
    
    return normalized_data

def split_data(data, target_column, test_size=0.2):
    """Split the data into training and testing sets."""
    # Shuffle the dataset
    shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate the split index
    split_index = int(len(shuffled_data) * (1 - test_size))
    
    # Split into training and test sets
    train_data = shuffled_data[:split_index]
    test_data = shuffled_data[split_index:]
    
    # Separate features and target
    X_train = train_data.drop(columns=[target_column]).values
    y_train = train_data[target_column].values
    X_test = test_data.drop(columns=[target_column]).values
    y_test = test_data[target_column].values
    
    # Convert to PyTorch tensors
    return (torch.tensor(X_train).float(), torch.tensor(y_train).float(),
            torch.tensor(X_test).float(), torch.tensor(y_test).float())
