import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os


# Generate synthetic helix data
class HelixDataset(Dataset):
    def __init__(self, num_samples=1000, num_points=100, noise_level=0.1):
        self.num_samples = num_samples
        self.num_points = num_points
        self.noise_level = noise_level
        self.data = []
        self.labels = []
        self._generate_data()

    def _generate_data(self):
        for _ in range(self.num_samples):
            amplitude = np.random.uniform(2.0, 10.0)
            frequency = np.random.uniform(400.0, 800.0)
            shift = np.random.uniform(0, 2 * np.pi)
            t = np.linspace(0, 2 * np.pi, self.num_points)
            x = amplitude * np.cos(frequency * t + shift)
            y = amplitude * np.sin(frequency * t + shift)
            z = t

            # Add Gaussian noise to the helix points
            x += np.random.normal(0, self.noise_level, size=self.num_points)
            y += np.random.normal(0, self.noise_level, size=self.num_points)
            z += np.random.normal(0, self.noise_level, size=self.num_points)

            helix_points = np.stack([x, y, z], axis=1)
            self.data.append(helix_points)
            self.labels.append([amplitude, frequency, shift])

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Define ResNet-inspired CNN model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, input_channels=3, num_blocks=3, num_classes=3):
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        # Residual layers
        self.layers = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(num_blocks)]
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (batch_size, channels, num_points)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


# Training loop
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=50):
    model.to(device)  # Move model to GPU if available
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_data, batch_labels in dataloader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)  # Move data to GPU

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# Function to seed everything for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model_to_torchscript(model, device, filepath="resnet1d_scripted.pt"):
    model.eval()  # Switch to evaluation mode
    example_input = torch.randn(1, 100, 3).to(device)  # Example input matching your model's input dimensions
    scripted_model = torch.jit.trace(model, example_input)  # Trace the model
    scripted_model.save(filepath)  # Save the scripted model
    print(f"Model saved to {filepath}")

# Main script
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # check if model exists
    if os.path.exists("model.pth"):
        print("Model already exists")

        # Load the model
        model = ResNet1D(input_channels=3, num_blocks=3, num_classes=3)
        model.load_state_dict(torch.load("model.pth"))
        model.to(device)
        model.eval()

        # Generate a new helix sample
        dataset = HelixDataset(num_samples=1, num_points=1000, noise_level=2.0)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # create plot to visualize the helix
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Helix')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_zlim(0, 2 * np.pi)


        for batch_data, batch_labels in dataloader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            print("Predicted:", outputs)
            print("Ground truth:", batch_labels)
            # plot predicted
            amplitude, frequency, shift = outputs[0].cpu().detach().numpy()
            t = np.linspace(0, 2 * np.pi, 1000)
            x = amplitude * np.cos(frequency * t + shift)
            y = amplitude * np.sin(frequency * t + shift)
            z = t
            ax.plot(x, y, z, color='b')
            # plot gt
            amplitude, frequency, shift = batch_labels[0].cpu().numpy()
            x = amplitude * np.cos(frequency * t + shift)
            y = amplitude * np.sin(frequency * t + shift)
            z = t
            ax.plot(x, y, z, color='r')
            break

        plt.show()

        save_model_to_torchscript(model, device, filepath="resnet1d_scripted.pt")

        exit(0)

    # Dataset and DataLoader
    dataset = HelixDataset(num_samples=100000, num_points=1000, noise_level=2.0)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Model, criterion, and optimizer
    model = ResNet1D(input_channels=3, num_blocks=3, num_classes=3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=50)

    # Save the model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved")
