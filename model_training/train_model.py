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

import yaml
from src.training_config import process_user_config
from src.hyperparameter_tuning import run_hyperparameter_tuning
from src.data_module import DataModule
from src.cell_model import LSTM, CellModel

now = datetime.now()
formatted_datetime = now.strftime('%Y%m%d%H%M%S')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlflow.set_tracking_uri("http://mlflow:5000")
# mlflow.set_experiment("SOX_Testing")

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

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.MNIST(root="./training/data", train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        mlflow.log_params(trial.params)

        for epoch in range(10):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data.view(data.size(0), -1))
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                trial.report(loss.item(), epoch * len(train_loader) + batch_idx)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            mlflow.log_metric("loss", loss.item(), step=epoch)
        mlflow.pytorch.log_model(model, "model")

    return loss.item()

if __name__ == "__main__":

    
    with open("config.yaml", "r") as confg_file:
        user_config = yaml.safe_load(confg_file)

    user_config["data_module"].update({"data_module": DataModule})
    user_config["model"].update({"model": LSTM})

    training_config = process_user_config(user_config)
    study = run_hyperparameter_tuning(training_config=training_config)
    
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
    
    # # Save study for later analysis
    # with mlflow.start_run():
    #     mlflow.log_params(study.best_trial.params)
    #     mlflow.log_metric("best_value", study.best_value)
        
    #     # Log the best model
    #     best_model = build_model(study.best_trial)
    #     mlflow.pytorch.log_model(best_model, "best_model")
        
    #     # Log the study results as an artifact
    #     study.trials_dataframe().to_csv("./training/output/optuna_results.csv")
    #     mlflow.log_artifact("./training/output/optuna_results.csv")
    #     print(f"Study saved to ./training/output/optuna_results.csv")

    #     print("To view the Optuna Dashboard, run the following command in your terminal:")
    #     print("optuna-dashboard sqlite:///optuna.db")
