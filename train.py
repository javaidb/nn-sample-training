import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import os

# Step 1: Define Custom Dataset for Synthetic Data
class SyntheticDataset(Dataset):
    def __init__(self, csv_file):
        # Load synthetic data from CSV
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Extract feature and label from the dataset
        x = torch.tensor(self.data.iloc[idx, 0], dtype=torch.float32)
        y = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)
        return x, y

# Step 2: Define a Simple Neural Network Model for Regression
class SimpleNN(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size)  # 1 input feature
        self.fc2 = nn.Linear(hidden_size, 1)  # 1 output feature
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 3: Define the objective function for Optuna hyperparameter optimization
def objective(trial):
    # Hyperparameters to optimize
    hidden_size = trial.suggest_int('hidden_size', 16, 128)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

    # Step 4: Initialize Model, Loss Function, and Optimizer
    model = SimpleNN(hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Step 5: Load dataset and create DataLoader
    train_dataset = SyntheticDataset(csv_file='./data/synthetic_data.csv')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Step 6: Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.view(-1, 1)  # Reshape input to match model's expected shape
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Optuna optimizes for the best model, here we return the average loss per epoch
        avg_loss = total_loss / len(train_loader)
        
        # Step 7: Report the loss to Optuna
        trial.report(avg_loss, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_loss

# Step 8: Main function to optimize hyperparameters and visualize results
def main():
    # Step 9: Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction='minimize')  # We want to minimize the loss
    study.optimize(objective, n_trials=50)  # Run optimization for 50 trials

    # Step 10: Print the best hyperparameters found by Optuna
    print(f"Best hyperparameters: {study.best_params}")

    # Step 11: Visualize the optimization history
    optuna.visualization.plot_optimization_history(study)
    plt.show()

    # Step 12: Visualize the hyperparameter importance
    optuna.visualization.plot_param_importances(study)
    plt.show()

    # Step 13: Retrain the model with the best hyperparameters
    best_hidden_size = study.best_params['hidden_size']
    best_learning_rate = study.best_params['learning_rate']

    # Initialize model with best hyperparameters
    model = SimpleNN(best_hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_learning_rate)

    # Retrain with the full dataset
    train_dataset = SyntheticDataset(csv_file='./data/synthetic_data.csv')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Print loss for every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")

    # Step 14: Save the trained model
    torch.save(model.state_dict(), 'best_model.pth')

    print("Training complete. Model saved as 'best_model.pth'.")

if __name__ == '__main__':
    main()
