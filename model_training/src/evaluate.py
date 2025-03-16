import torch
import torch.optim as optim
import torch.nn as nn

def evaluate_model(model, test_loader):
    """Evaluate the model on the test set."""
    model.eval()
    total_loss = 0.0
    
    criterion = nn.MSELoss()  # Assuming regression task
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)  # Compute loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f'Average Test Loss: {avg_loss:.4f}')
