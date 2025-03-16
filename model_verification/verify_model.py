import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json

def get_model_architecture(state_dict):
    """Extract model architecture from state dict keys"""
    # Get all unique layer indices from the state dict keys
    layer_indices = sorted(list(set(int(k.split('.')[0]) for k in state_dict.keys())))
    
    # Get weight matrices only
    weight_matrices = {i: state_dict[f"{i}.weight"] for i in layer_indices}
    
    # Extract layer sizes from weight matrices
    layer_sizes = []
    first_layer = min(layer_indices)
    first_matrix = weight_matrices[first_layer]
    input_size = first_matrix.shape[1]  # Input size from first layer
    
    # For each layer, get its output size
    hidden_sizes = []
    for idx in layer_indices[:-1]:  # Exclude the last layer
        hidden_sizes.append(weight_matrices[idx].shape[0])
    
    return len(hidden_sizes), hidden_sizes

def load_model(model_path):
    # Load the state dict first
    checkpoint = torch.load(model_path)
    print("\nCheckpoint contents:")
    print(f"Trial parameters: {checkpoint['trial_params']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"Validation accuracy: {checkpoint['val_accuracy']:.2f}%")
    
    # Print shapes of model weights
    print("\nModel state dict shapes:")
    for key, value in checkpoint['state_dict'].items():
        print(f"  {key}: {value.shape}")
    
    # Build model with same architecture
    layers = []
    in_features = 784
    params = checkpoint['trial_params']
    n_layers = params['n_layers']
    
    for i in range(n_layers):
        out_features = params[f'n_units_l{i}']
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features
    layers.append(nn.Linear(in_features, 10))
    
    model = nn.Sequential(*layers)
    
    # Print model structure before loading weights
    print("\nModel structure:")
    print(model)
    
    try:
        model.load_state_dict(checkpoint['state_dict'])
        print("\nSuccessfully loaded state dict")
    except Exception as e:
        print("\nError loading state dict:", str(e))
        raise
    
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    # Add debugging for first batch
    first_batch = True
    
    with torch.no_grad():
        for data, target in test_loader:
            # Print original data shape and stats
            if first_batch:
                print("\nOriginal input stats:")
                print(f"Original shape: {data.shape}")
                print(f"Original range: [{data.min():.3f}, {data.max():.3f}]")
                print(f"Original mean: {data.mean():.3f}")
                print(f"Original std: {data.std():.3f}")
            
            # Flatten the input images
            data = data.view(data.size(0), -1)  # Flatten from (B, 1, 28, 28) to (B, 784)
            if first_batch:
                print("\nFlattened input stats:")
                print(f"Flattened shape: {data.shape}")
                print(f"Input range: [{data.min():.3f}, {data.max():.3f}]")
                print(f"Input mean: {data.mean():.3f}")
                print(f"Input std: {data.std():.3f}")
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if first_batch:
                print(f"\nOutput stats:")
                print(f"Output shape: {output.shape}")
                print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
                print(f"Output mean: {output.mean().item():.3f}")
                print(f"Output std: {output.std().item():.3f}")
                print(f"\nFirst output sample (logits):")
                print(output[0])
                print(f"Softmax of first output:")
                print(torch.nn.functional.softmax(output[0], dim=0))
                first_batch = False
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Print predictions for first few items in first batch
            if len(all_preds) <= 10:
                print(f"\nFirst few predictions vs targets:")
                for p, t in zip(predicted[:10].cpu().numpy(), target[:10].cpu().numpy()):
                    print(f"Predicted: {p}, Target: {t}")
    
    accuracy = 100. * correct / total
    return accuracy, all_preds, all_targets

def plot_confusion_matrix(y_true, y_pred, save_path):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def visualize_predictions(model, test_loader, device, num_samples=10):
    model.eval()
    images, labels = next(iter(test_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    with torch.no_grad():
        # Flatten the input images
        flattened_images = images.view(images.size(0), -1)  # Flatten from (B, 1, 28, 28) to (B, 784)
        outputs = model(flattened_images.to(device))
        _, predicted = torch.max(outputs, 1)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for idx in range(num_samples):
        axes[idx].imshow(images[idx].squeeze(), cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f'True: {labels[idx]}\nPred: {predicted[idx].cpu()}')
    
    plt.tight_layout()
    plt.savefig('./model_verification/output/sample_predictions.png')
    plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model configuration and weights
    model_path = "./training/output/best_model.pth"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"\nLoading model from: {model_path}")
    # Load the model with architecture detection
    model = load_model(model_path)
    model = model.to(device)
    
    # Prepare the test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root="./training/data",
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate the model
    accuracy, all_preds, all_targets = evaluate_model(model, test_loader, device)
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    # Create output directory if it doesn't exist
    Path("./model_verification/output").mkdir(parents=True, exist_ok=True)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        all_targets,
        all_preds,
        './model_verification/output/confusion_matrix.png'
    )
    print("Confusion matrix saved to ./model_verification/output/confusion_matrix.png")
    
    # Visualize some predictions
    visualize_predictions(model, test_loader, device)
    print("Sample predictions saved to ./model_verification/output/sample_predictions.png")

if __name__ == "__main__":
    main()