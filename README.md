# MNIST Neural Network Optimization

A PyTorch implementation of MNIST digit classification using Optuna for hyperparameter optimization and MLflow for experiment tracking.

## Project Structure

```
.
├── model_training/
│   └── train_model_sample.py    # Training script with Optuna optimization
├── model_verification/
│   └── verify_model.py          # Model evaluation and visualization
└── training/
    ├── data/                    # MNIST dataset storage
    └── output/                  # Trained models and results
```

## Features

- Automated hyperparameter optimization using Optuna
- Experiment tracking with MLflow
- Model architecture search (1-3 layers)
- Training/validation split (80/20)
- Comprehensive model verification with visualizations

## Usage with Docker

1. Start MLflow and run training:
```bash
docker compose up mlflow cuda_model_trainer
```

2. Run model verification:
```bash
docker compose up model_verification
```

MLflow UI will be available at `http://localhost:5000`

## Outputs

- Best model weights: `training/output/best_model.pth`
- Confusion matrix: `model_verification/output/confusion_matrix.png`
- Sample predictions: `model_verification/output/sample_predictions.png`
- Training results: `training/output/optuna_results.csv`

## Experiment Tracking

View training progress and results in the MLflow UI at `http://localhost:5000`. The UI will update in real-time as the training progresses.