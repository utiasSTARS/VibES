import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class RealDataset(Dataset):
    """Dataset for real signal data from txt file"""

    def __init__(self, data_file, seq_length=50, prediction_length=20, overlap_ratio=0.5):
        self.seq_length = seq_length
        self.prediction_length = prediction_length
        self.overlap_ratio = overlap_ratio

        # Load and process real data
        self.time_series, self.sampling_info = self.load_real_data(data_file)

        # Create sequences
        self.data = []
        self.targets = []
        self.create_sequences()

        print(f"Created {len(self.data)} sequences from real data")
        print(f"Original time series length: {len(self.time_series)}")

    def load_real_data(self, data_file):
        """Load real data from txt file (time, value format)"""
        try:
            # Try different delimiters
            for delimiter in ['\t', ' ', ',', ';']:
                try:
                    data = pd.read_csv(data_file, delimiter=delimiter, header=None)
                    if data.shape[1] >= 2:
                        break
                except:
                    continue

            if data.shape[1] < 2:
                # Try space-separated with multiple spaces
                data = pd.read_csv(data_file, delim_whitespace=True, header=None)

            # Extract time and values
            time_points = data.iloc[:, 0].values
            signal_values = data.iloc[:, 1].values

            # Handle potential string values
            if signal_values.dtype == 'object':
                signal_values = pd.to_numeric(signal_values, errors='coerce')
                # Remove NaN values
                valid_mask = ~np.isnan(signal_values)
                time_points = time_points[valid_mask]
                signal_values = signal_values[valid_mask]

            # remove DC component
            # signal_values -= np.mean(signal_values)

            # time to seconds from us
            time_points = time_points / 1e6  # Convert microseconds to seconds

            print(f"Loaded {len(signal_values)} data points")
            print(f"Time range: {time_points[0]:.4f} to {time_points[-1]:.4f}")
            print(f"Signal range: {signal_values.min():.4f} to {signal_values.max():.4f}")

            # Calculate sampling information
            time_diffs = np.diff(time_points)
            avg_dt = np.mean(time_diffs)
            dt_std = np.std(time_diffs)

            print(f"Average sampling interval: {avg_dt:.6f}")
            print(f"Sampling interval std: {dt_std:.6f}")

            # Check if sampling is approximately uniform
            is_uniform = dt_std < 0.1 * avg_dt
            print(f"Sampling is {'uniform' if is_uniform else 'non-uniform'}")

            sampling_info = {
                'avg_dt': avg_dt,
                'dt_std': dt_std,
                'is_uniform': is_uniform,
                'time_points': time_points
            }

            return signal_values, sampling_info

        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please ensure your file format is: time_value1 signal_value1")
            print("                                    time_value2 signal_value2")
            print("                                    ...")
            raise

    def create_sequences(self):
        """Create overlapping sequences from the time series"""
        total_length = self.seq_length + self.prediction_length

        if len(self.time_series) < total_length:
            raise ValueError(f"Time series too short. Need at least {total_length} points, got {len(self.time_series)}")

        # Calculate step size based on overlap ratio
        step_size = max(1, int(self.seq_length * (1 - self.overlap_ratio)))

        # Create sequences
        for i in range(0, len(self.time_series) - total_length + 1, step_size):
            # Extract signal sequence
            full_sequence = self.time_series[i:i + total_length]

            input_seq = full_sequence[:self.seq_length]
            target_seq = full_sequence[self.seq_length:self.seq_length + self.prediction_length]

            # Create time features
            if self.sampling_info['is_uniform']:
                # For uniform sampling, use constant dt
                dt_normalized = self.sampling_info['avg_dt']
                time_features = np.full(self.seq_length, dt_normalized)
            else:
                # For non-uniform sampling, use actual time differences
                start_idx = i
                time_slice = self.sampling_info['time_points'][start_idx:start_idx + self.seq_length + 1]
                time_diffs = np.diff(time_slice)
                # Pad to match sequence length
                if len(time_diffs) < self.seq_length:
                    time_diffs = np.pad(time_diffs, (0, self.seq_length - len(time_diffs)),
                                        mode='constant', constant_values=self.sampling_info['avg_dt'])
                time_features = time_diffs[:self.seq_length]

            # Normalize signal values to match the range the model was trained on
            # Assuming original model was trained on signals with range roughly [-15, 15]
            signal_std = np.std(input_seq)
            if signal_std > 0:
                normalization_factor = 10.0 / signal_std  # Target std of ~10
                input_seq_norm = input_seq * normalization_factor
                target_seq_norm = target_seq * normalization_factor
            else:
                input_seq_norm = input_seq
                target_seq_norm = target_seq
                normalization_factor = 1.0

            # Combine features
            input_features = np.column_stack([
                input_seq_norm,
                time_features
            ])

            self.data.append(input_features)
            self.targets.append(target_seq_norm)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.targets[idx])


# Import your original model classes (SineActivation, SignalMLP, HybridSignalModel)
class SineActivation(nn.Module):
    """Sine activation function for periodic signals"""

    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(w0))

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SignalMLP(nn.Module):
    """Improved MLP with sine activations for signal prediction"""

    def __init__(self, input_size=100, hidden_size=256, num_layers=4, output_size=20, w0=30.0):
        super(SignalMLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(SineActivation(w0=w0))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(SineActivation(w0=10.0))

        layers.append(nn.Linear(hidden_size, output_size))

        self.network = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            first_layer = self.network[0]
            bound = 1.0 / self.input_size
            first_layer.weight.uniform_(-bound, bound)

            for i in range(2, len(self.network), 2):
                if i < len(self.network) - 1:
                    layer = self.network[i]
                    bound = np.sqrt(6.0 / layer.in_features) / 10.0
                    layer.weight.uniform_(-bound, bound)
                else:
                    layer = self.network[i]
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        layer.bias.fill_(0.0)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.network(x)


class HybridSignalModel(nn.Module):
    """Improved hybrid model"""

    def __init__(self, seq_length=50, input_features=2, hidden_size=128, output_size=20):
        super(HybridSignalModel, self).__init__()

        self.conv1 = nn.Conv1d(input_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        combined_size = 32 + seq_length * input_features
        self.sine_mlp = SignalMLP(
            input_size=combined_size,
            hidden_size=hidden_size,
            num_layers=4,
            output_size=output_size,
            w0=30.0
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.size(0)

        x_conv = x.transpose(1, 2)

        conv1_out = torch.relu(self.conv1(x_conv))
        conv2_out = torch.relu(self.conv2(conv1_out))
        conv3_out = torch.relu(self.conv3(conv2_out))

        pooled = self.global_pool(conv3_out).squeeze(-1)
        pooled = self.dropout(pooled)

        x_flat = x.view(batch_size, -1)
        combined = torch.cat([pooled, x_flat], dim=1)

        output = self.sine_mlp(combined)
        return output


def load_pretrained_model(model_path, model_type="Hybrid_Model", seq_length=50, output_size=20):
    """Load a pretrained model"""
    try:
        # Create model instance
        if model_type == "Hybrid_Model":
            model = HybridSignalModel(
                seq_length=seq_length,
                input_features=2,
                hidden_size=256,
                output_size=output_size
            )
        elif model_type == "Sine_MLP":
            model = SignalMLP(
                input_size=seq_length * 2,
                hidden_size=256,
                num_layers=4,
                output_size=output_size,
                w0=30.0
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load pretrained weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Successfully loaded pretrained {model_type} model")
        return model

    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        print("Creating new model instead...")

        if model_type == "Hybrid_Model":
            return HybridSignalModel(seq_length=seq_length, input_features=2,
                                     hidden_size=256, output_size=output_size)
        else:
            return SignalMLP(input_size=seq_length * 2, hidden_size=256,
                             num_layers=4, output_size=output_size, w0=30.0)


def fine_tune_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.0001):
    """Fine-tune the model on real data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Fine-tuning on device: {device}")

    criterion = nn.MSELoss()

    # Use lower learning rate for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # More conservative learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.8, min_lr=1e-6, verbose=True
    )

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch + 1}")
                break

            loss.backward()
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
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, 'best_finetuned_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}')

    # Load best model
    if os.path.exists('best_finetuned_model.pth'):
        checkpoint = torch.load('best_finetuned_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best fine-tuned model with val_loss: {checkpoint['val_loss']:.6f}")

    return train_losses, val_losses


def visualize_real_data_prediction(model, dataset, num_examples=3):
    """Visualize model predictions on real data"""
    device = next(model.parameters()).device
    model.eval()

    fig, axes = plt.subplots(num_examples, 1, figsize=(15, 4 * num_examples))
    if num_examples == 1:
        axes = [axes]

    with torch.no_grad():
        for i in range(num_examples):
            idx = np.random.randint(0, len(dataset))
            input_seq, target_seq = dataset[idx]

            input_batch = input_seq.unsqueeze(0).to(device)
            prediction = model(input_batch).squeeze().cpu().numpy()

            input_signal = input_seq[:, 0].numpy()
            target_signal = target_seq.numpy()

            x_input = np.arange(len(input_signal))
            x_target = np.arange(len(input_signal), len(input_signal) + len(target_signal))

            axes[i].plot(x_input, input_signal, 'b-', label='Input Sequence', linewidth=2)
            axes[i].plot(x_target, target_signal, 'g-', label='True Future', linewidth=2)
            axes[i].plot(x_target, prediction, 'r--', label='Predicted Future', linewidth=2)

            axes[i].axvline(x=len(input_signal) - 0.5, color='black', linestyle=':', alpha=0.7)
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Signal Amplitude')
            axes[i].set_title(f'Real Data Prediction Example {i + 1}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

            mse = np.mean((target_signal - prediction) ** 2)
            axes[i].text(0.02, 0.98, f'MSE: {mse:.4f}', transform=axes[i].transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


def main():
    """Main fine-tuning script"""

    # Configuration
    DATA_FILE = "centroid_data_x.txt"  # Replace with your actual file path
    PRETRAINED_MODEL_PATH = "best_Hybrid_Model_model.pth"  # Path to your pretrained model
    MODEL_TYPE = "Hybrid_Model"  # or "Sine_MLP"
    SEQ_LENGTH = 50
    PREDICTION_LENGTH = 10

    print("Loading real data...")
    try:
        # Create dataset from real data
        real_dataset = RealDataset(
            data_file=DATA_FILE,
            seq_length=SEQ_LENGTH,
            prediction_length=PREDICTION_LENGTH,
            overlap_ratio=0.7  # High overlap for more training data
        )

        # Split into train/validation (80/20)
        dataset_size = len(real_dataset)
        train_size = int(0.9 * dataset_size)
        val_size = dataset_size - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            real_dataset, [train_size, val_size]
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Load pretrained model
        print("Loading pretrained model...")
        model = load_pretrained_model(
            PRETRAINED_MODEL_PATH,
            model_type=MODEL_TYPE,
            seq_length=SEQ_LENGTH,
            output_size=PREDICTION_LENGTH
        )

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Fine-tune the model
        print("Starting fine-tuning...")
        train_losses, val_losses = fine_tune_model(
            model, train_loader, val_loader,
            num_epochs=10000,
            learning_rate=0.00001  # Lower learning rate for fine-tuning
        )

        print("Fine-tuning completed!")
        print(f"Final validation loss: {val_losses[-1]:.6f}")

        # Plot training history
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Fine-tuning History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.show()

        # Visualize predictions on real data
        print("Generating prediction examples...")
        visualize_real_data_prediction(model, real_dataset, num_examples=4)

        # Save fine-tuned model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': MODEL_TYPE,
            'seq_length': SEQ_LENGTH,
            'prediction_length': PREDICTION_LENGTH,
        }, 'final_finetuned_model.pth')

        print("Fine-tuned model saved as 'final_finetuned_model.pth'")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
