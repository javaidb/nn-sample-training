import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch

import optuna
from pathlib import Path
from datetime import datetime

now = datetime.now()
formatted_datetime = now.strftime('%Y%m%d%H%M%S')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("MNIST_Optimization")

def build_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []
    in_features = 784
    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features
    layers.append(nn.Linear(in_features, 10))
    return nn.Sequential(*layers)

def objective(trial):
    with mlflow.start_run(nested=True):
        model = build_model(trial).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_loguniform("lr", 1e-5, 1e-1))
        criterion = nn.CrossEntropyLoss()

        # Split data into train and validation
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        full_dataset = torchvision.datasets.MNIST(root="./training/data", train=True, download=True, transform=transform)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True)

        mlflow.log_params(trial.params)
        best_val_loss = float('inf')

        try:
            for epoch in range(10):
                # Training phase
                model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data.view(data.size(0), -1))
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                    if batch_idx % 100 == 0:  # Log every 100 batches
                        trial.report(loss.item(), epoch * len(train_loader) + batch_idx)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()

                avg_loss = total_loss / len(train_loader)
                accuracy = 100. * correct / total
                
                # Validation phase
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data.view(data.size(0), -1))
                        val_loss += criterion(output, target).item()
                        _, predicted = output.max(1)
                        val_total += target.size(0)
                        val_correct += predicted.eq(target).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100. * val_correct / val_total

                # Log metrics
                mlflow.log_metrics({
                    "train_loss": avg_loss,
                    "train_accuracy": accuracy,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy
                }, step=epoch)

                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    mlflow.pytorch.log_model(model, "best_model")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
            raise e

    return best_val_loss  # Return best validation loss instead of last training loss

if __name__ == "__main__":
    # Create and run Optuna study with SQLite storage
    study = optuna.create_study(direction="minimize", storage="sqlite:///optuna.db", study_name=F"mnist_optimization_{formatted_datetime}")
    study.optimize(objective, n_trials=10)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Create output directory
    Path("./training/output").mkdir(parents=True, exist_ok=True)
    
    # Save best model
    best_model = build_model(trial)
    torch.save(best_model.state_dict(), "./training/output/best_model.pth")
    
    # Save study for later analysis
    with mlflow.start_run():
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_value", study.best_value)
        
        # Log the best model
        best_model = build_model(study.best_trial)
        mlflow.pytorch.log_model(best_model, "best_model")
        
        # Log the study results as an artifact
        study.trials_dataframe().to_csv("./training/output/optuna_results.csv")
        mlflow.log_artifact("./training/output/optuna_results.csv")
        print(f"Study saved to ./training/output/optuna_results.csv")

        print("To view the Optuna Dashboard, run the following command in your terminal:")
        print("optuna-dashboard sqlite:///optuna.db")
