import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class SignalDataset(Dataset):
    """Dataset for non-uniformly sampled signals"""

    def __init__(self, num_samples=1000, seq_length=50, prediction_length=20):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.prediction_length = prediction_length
        self.data = []
        self.targets = []
        self.generate_data()

    def generate_data(self):
        """Generate synthetic signals with varying frequencies and amplitudes"""

        for _ in range(self.num_samples):
            # Random signal parameters - original high frequency ranges
            frequency = np.random.uniform(10, 100)  # 10-100 Hz
            amplitude = np.random.uniform(3, 15)  # 3-15 pixels
            phase = np.random.uniform(0, 2 * np.pi)
            noise_level = np.random.uniform(0.1, 0.5)
            signal_shift = np.random.uniform(20, 1900)  # Random shift in signal

            # Generate uniform time sampling with sufficient resolution
            total_points = self.seq_length + self.prediction_length

            # Calculate sampling rate: at least 10x the highest frequency (Nyquist + margin)
            sampling_rate = frequency * 20  # 20x for good resolution
            dt = 1.0 / sampling_rate

            # Create uniform time grid
            time_points = np.arange(total_points) * dt

            # Generate clean signal
            signal = amplitude * np.sin(2 * np.pi * frequency * time_points + phase) + signal_shift

            # Add noise
            signal += np.random.normal(0, noise_level, len(signal))

            # Split into input sequence and target
            input_seq = signal[:self.seq_length]
            target_seq = signal[self.seq_length:self.seq_length + self.prediction_length] - signal_shift

            # For uniform sampling, time differences are constant
            # But we'll include the sampling rate info as a feature
            dt_normalized = dt * frequency  # Normalize by frequency for scale invariance
            time_features = np.full(self.seq_length, dt_normalized)

            # Combine signal values with time step information
            input_features = np.column_stack([
                input_seq,
                time_features
            ])

            self.data.append(input_features)
            self.targets.append(target_seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.targets[idx])


class SineActivation(nn.Module):
    """Sine activation function for periodic signals"""

    def __init__(self, w0=30.0):  # Higher w0 for high frequency signals
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(w0))  # Make w0 learnable

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SignalMLP(nn.Module):
    """Improved MLP with sine activations for signal prediction"""

    def __init__(self, input_size=100, hidden_size=256, num_layers=4, output_size=20, w0=30.0):
        super(SignalMLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Build layers with sine activations
        layers = []

        # First layer with higher w0 for high frequency signals
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(SineActivation(w0=w0))

        # Hidden layers with lower w0
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(SineActivation(w0=10.0))  # Lower w0 for hidden layers

        # Output layer (no activation for regression)
        layers.append(nn.Linear(hidden_size, output_size))

        self.network = nn.Sequential(*layers)

        # Proper initialization for high frequency signals
        self.init_weights()

    def init_weights(self):
        """Proper initialization for high frequency sine activation networks"""
        with torch.no_grad():
            # First layer - uniform distribution based on input size
            first_layer = self.network[0]
            bound = 1.0 / self.input_size
            first_layer.weight.uniform_(-bound, bound)

            # Hidden layers - smaller bounds for high frequency stability
            for i in range(2, len(self.network), 2):  # Every other layer (skip activations)
                if i < len(self.network) - 1:  # Not the output layer
                    layer = self.network[i]
                    bound = np.sqrt(6.0 / layer.in_features) / 10.0  # Smaller bound for stability
                    layer.weight.uniform_(-bound, bound)
                else:  # Output layer - standard initialization
                    layer = self.network[i]
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        layer.bias.fill_(0.0)

    def forward(self, x):
        # Flatten the sequence (batch_size, seq_len, features) -> (batch_size, seq_len * features)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.network(x)


class HybridSignalModel(nn.Module):
    """Improved hybrid model"""

    def __init__(self, seq_length=50, input_features=2, hidden_size=128, output_size=20):
        super(HybridSignalModel, self).__init__()

        # Temporal processing with multiple scales
        self.conv1 = nn.Conv1d(input_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Sine MLP for final prediction - higher w0 for high freq signals
        combined_size = 32 + seq_length * input_features  # Features from conv + original
        self.sine_mlp = SignalMLP(
            input_size=combined_size,
            hidden_size=hidden_size,
            num_layers=4,
            output_size=output_size,
            w0=30.0  # Higher w0 for high frequency signals
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.size(0)

        # Extract temporal features
        x_conv = x.transpose(1, 2)  # (batch, features, seq_len)

        # Multi-scale convolutions
        conv1_out = torch.relu(self.conv1(x_conv))
        conv2_out = torch.relu(self.conv2(conv1_out))
        conv3_out = torch.relu(self.conv3(conv2_out))

        # Global pooling to get summary features
        pooled = self.global_pool(conv3_out).squeeze(-1)  # (batch, 32)
        pooled = self.dropout(pooled)

        # Flatten original input
        x_flat = x.view(batch_size, -1)

        # Combine features
        combined = torch.cat([pooled, x_flat], dim=1)

        # Apply sine MLP
        output = self.sine_mlp(combined)

        return output


def train_model(model, train_loader, val_loader, num_epochs=200, learning_rate=0.001, model_name="model"):
    """Improved training function with better error handling"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Training on device: {device}")

    criterion = nn.MSELoss()

    # Better optimizer settings
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # More aggressive learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.7, min_lr=1e-5, verbose=True
    )

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20

    # Create unique filename for this model
    best_model_path = f'best_{model_name}_model.pth'

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        try:
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch + 1}, batch {batch_idx}")
                    break

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model with error handling
                try:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'train_loss': train_loss
                    }, best_model_path)
                except Exception as e:
                    print(f"Warning: Could not save model checkpoint: {e}")
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 5 == 0:  # More frequent logging
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}')

        except KeyboardInterrupt:
            print("Training interrupted by user")
            break
        except Exception as e:
            print(f"Error during training: {e}")
            break

    # Load best model if it exists
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']} with val_loss: {checkpoint['val_loss']:.6f}")
        except Exception as e:
            print(f"Warning: Could not load best model: {e}")

    return train_losses, val_losses


def visualize_prediction(model, dataset, num_examples=3):
    """Visualize model predictions with error handling"""
    try:
        device = next(model.parameters()).device
        model.eval()
        fig, axes = plt.subplots(num_examples, 1, figsize=(15, 4 * num_examples))
        if num_examples == 1:
            axes = [axes]

        with torch.no_grad():
            for i in range(num_examples):
                # Get a random sample
                idx = np.random.randint(0, len(dataset))
                input_seq, target_seq = dataset[idx]

                # Make prediction
                input_batch = input_seq.unsqueeze(0).to(device)
                prediction = model(input_batch).squeeze().cpu().numpy()

                # Plot
                input_signal = input_seq[:, 0].numpy()  # First column is signal values
                target_signal = target_seq.numpy()

                x_input = np.arange(len(input_signal))
                x_target = np.arange(len(input_signal), len(input_signal) + len(target_signal))
                x_pred = x_target

                axes[i].plot(x_input, input_signal, 'b-', label='Input Sequence', linewidth=2)
                axes[i].plot(x_target, target_signal, 'g-', label='True Future', linewidth=2)
                axes[i].plot(x_pred, prediction, 'r--', label='Predicted Future', linewidth=2)

                axes[i].axvline(x=len(input_signal) - 0.5, color='black', linestyle=':', alpha=0.7)
                axes[i].set_xlabel('Time Steps')
                axes[i].set_ylabel('Signal Amplitude')
                axes[i].set_title(f'Signal Prediction Example {i + 1}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

                # Calculate and display error
                mse = np.mean((target_signal - prediction) ** 2)
                axes[i].text(0.02, 0.98, f'MSE: {mse:.4f}', transform=axes[i].transAxes,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in visualization: {e}")


def plot_training_history(train_losses, val_losses):
    """Plot training and validation losses with error handling"""
    try:
        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
        plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # Smoothed version
        plt.subplot(1, 2, 2)
        window = max(1, len(train_losses) // 20)
        if len(train_losses) > window:
            train_smooth = np.convolve(train_losses, np.ones(window) / window, mode='valid')
            val_smooth = np.convolve(val_losses, np.ones(window) / window, mode='valid')
            epochs_smooth = np.arange(window // 2, len(train_losses) - window // 2 + 1)
            plt.plot(epochs_smooth, train_smooth, label='Training Loss (Smoothed)', color='blue')
            plt.plot(epochs_smooth, val_smooth, label='Validation Loss (Smoothed)', color='red')
        else:
            plt.plot(train_losses, label='Training Loss', color='blue')
            plt.plot(val_losses, label='Validation Loss', color='red')

        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Smoothed Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plotting training history: {e}")


def evaluate_model(model, test_loader):
    """Evaluate model performance with error handling"""
    try:
        device = next(model.parameters()).device
        model.eval()

        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)

                loss = nn.MSELoss()(outputs, batch_y)
                total_loss += loss.item()

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate additional metrics
        mae = np.mean(np.abs(all_predictions - all_targets))
        rmse = np.sqrt(avg_loss)

        print(f"Test Results:")
        print(f"  MSE: {avg_loss:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")

        return avg_loss, all_predictions, all_targets
    except Exception as e:
        print(f"Error in model evaluation: {e}")
        return None, None, None


def main():
    """Main training script with improved error handling"""
    try:
        print("Generating dataset...")
        output = 10
        # Create datasets with more reasonable sizes
        train_dataset = SignalDataset(num_samples=50000, seq_length=50, prediction_length=output)
        val_dataset = SignalDataset(num_samples=1000, seq_length=50, prediction_length=output)
        test_dataset = SignalDataset(num_samples=1000, seq_length=50, prediction_length=output)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

        print("Creating models...")

        # Try both models with settings optimized for high frequency signals
        models_to_try = {
            "Sine_MLP": SignalMLP(
                input_size=50 * 2,  # seq_length * input_features
                hidden_size=256,  # Larger for high freq signals
                num_layers=4,  # Deeper for complex patterns
                output_size=output,
                w0=30.0  # High w0 for high freq
            ),
            "Hybrid_Model": HybridSignalModel(
                seq_length=50,
                input_features=2,
                hidden_size=256,  # Larger hidden size
                output_size=output
            )
        }

        # Choose which model to train
        model_name = "Hybrid_Model"  # Change this to "Sine_MLP" to try the pure sine MLP
        model = models_to_try[model_name]

        print(f"{model_name} parameters: {sum(p.numel() for p in model.parameters()):,}")

        print(f"Starting training with {model_name}...")
        # Train model with settings suitable for high frequency signals
        train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            num_epochs=150, learning_rate=0.0005,  # Lower LR for stability with high freq
            model_name=model_name
        )

        if train_losses:  # Check if training was successful
            print("Training completed!")
            print(f"Final training loss: {train_losses[-1]:.6f}")
            print(f"Final validation loss: {val_losses[-1]:.6f}")

            # Plot training history
            plot_training_history(train_losses, val_losses)

            # Evaluate on test set
            print("\nEvaluating on test set...")
            evaluate_model(model, test_loader)

            # Visualize predictions
            print("Generating prediction examples...")
            visualize_prediction(model, test_dataset, num_examples=4)

            # Save final model
            final_model_path = f'signal_{model_name}_final.pth'
            try:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_name': model_name,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }, final_model_path)
                print(f"Model saved as '{final_model_path}'")
            except Exception as e:
                print(f"Warning: Could not save final model: {e}")
        else:
            print("Training failed or was interrupted.")

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
