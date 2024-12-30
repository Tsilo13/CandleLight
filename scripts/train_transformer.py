import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.transformer_model import TransformerModel
from src.data.data_loader import load_data

# Dataset class
class CandlestickDataset(Dataset):
    def __init__(self, data, seq_len):
        """
        Dataset for candlestick sequences.
        Args:
            data (torch.Tensor): Input data of shape (num_samples, input_dim).
            seq_len (int): Length of each sequence.
        """
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]  # Sequence input
        y = self.data[idx + 1 : idx + self.seq_len + 1]  # Sequence target
        return x, y

# Training loop
def train_model(model, dataloader, criterion, optimizer, device, epochs=10):
    """
    Training loop for the transformer model.
    Args:
        model (nn.Module): The transformer model.
        dataloader (DataLoader): DataLoader for the dataset.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device (CPU or GPU).
        epochs (int): Number of training epochs.
    """
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")

# Main function
def main():
    # Load data
    data = load_data("data/SPY_5min_60d.csv")  # Replace with your dataset path
    
    # Drop non-numeric columns like Datetime
    if 'Datetime' in data.columns:
        data = data.drop(columns=['Datetime'])

    # Ensure all data is numeric
    data = data.select_dtypes(include=[float, int])
    
    # Handle missing values (if any)
    data = data.fillna(0)  # Replace NaN with 0
    data_tensor = torch.tensor(data.values, dtype=torch.float32)

    # Parameters
    seq_len = 48
    input_dim = 3
    embed_dim = 64
    num_heads = 4
    dff = 256
    num_layers = 4
    output_dim = 3
    batch_size = 32
    epochs = 10
    learning_rate = 1e-4

    # Prepare dataset and dataloader
    dataset = CandlestickDataset(data_tensor, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = TransformerModel(input_dim, embed_dim, num_heads, dff, num_layers, output_dim)
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, dataloader, criterion, optimizer, device, epochs)

if __name__ == "__main__":
    main()
