import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import copy
from collections import defaultdict
import random

# Optimize PyTorch for CPU multithreading
torch.set_num_threads(8)
print(f"PyTorch using {torch.get_num_threads()} threads")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ----- Data Preparation -----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load full MNIST datasets
train_ds_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds_full = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

print(f"Train dataset size: {len(train_ds_full)}")
print(f"Test dataset size: {len(test_ds_full)}")


# ========================================
# 🧠 MODEL DEFINITION (same as sync FedAvg)
# ========================================


class MLPModel(nn.Module):
    """Simple MLP model for MNIST classification"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ========================================
# 👥 CLIENT SIMULATION (same split logic)
# ========================================


class AsyncFederatedClient:
    """
    Represents a client in asynchronous federated learning.
    Each client:
      - Has its own local data
      - Maintains a local copy of the global model parameters (possibly stale)
    """

    def __init__(self, client_id, local_data, local_labels):
        self.client_id = client_id
        self.local_data = local_data
        self.local_labels = local_labels
        self.data_size = len(local_data)

        # Local model snapshot (state dict) and version
        self.local_params = None
        self.model_version = 0  # version of global model this snapshot came from

        print(f"  Client {client_id}: {self.data_size} samples")

    def pull_from_server(self, global_params, version):
        """Update local parameters from server (simulate pull)."""
        self.local_params = {k: v.clone() for k, v in global_params.items()}
        self.model_version = version

    def train_async(self, base_model, epochs=1, lr=0.01, batch_size=32):
        """
        Train locally starting from the client's (possibly stale) local_params.

        Args:
            base_model: A model instance with correct architecture (weights will be overwritten)
            epochs: Number of local epochs
            lr: Learning rate
            batch_size: Batch size

        Returns:
            updated_params: state_dict after local training
            used_version: global model version this client started from (for staleness)
        """
        if self.local_params is None:
            raise ValueError("Client has no local parameters. Call pull_from_server first.")

        # Copy global structure and load local (possibly stale) params
        local_model = copy.deepcopy(base_model)
        with torch.no_grad():
            for name, param in local_model.named_parameters():
                param.data.copy_(self.local_params[name])

        local_model.to(DEVICE)
        local_model.train()

        # Create DataLoader for local data
        local_dataset = torch.utils.data.TensorDataset(
            torch.stack(self.local_data),
            torch.tensor(self.local_labels)
        )
        local_loader = DataLoader(local_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

        for _ in range(epochs):
            for images, labels in local_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                # Forward pass
                outputs = local_model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Return updated parameters and the version they started from
        updated_params = {name: param.data.clone().cpu()
                          for name, param in local_model.named_parameters()}
        return updated_params, self.model_version


def split_data_among_clients(dataset, num_clients, distribution='iid'):
    """Same client split logic as sync FedAvg example."""
    clients = []

    if distribution == 'iid':
        print(f"\n📊 Creating {num_clients} clients with IID data distribution...")
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        samples_per_client = len(dataset) // num_clients

        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(dataset)
            client_indices = indices[start_idx:end_idx]

            client_data = []
            client_labels = []
            for idx in client_indices:
                image, label = dataset[idx]
                client_data.append(image)
                client_labels.append(label)

            clients.append(AsyncFederatedClient(i, client_data, client_labels))

    elif distribution == 'non-iid':
        print(f"\n📊 Creating {num_clients} clients with Non-IID data distribution...")
        print(f"   Each client will have data primarily from 1-2 classes")

        data_by_class = defaultdict(list)
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            data_by_class[label].append((image, label))

        classes = list(range(10))
        for client_id in range(num_clients):
            num_classes_for_client = random.randint(1, 2)
            client_classes = random.sample(classes, num_classes_for_client)

            client_data = []
            client_labels = []

            # Main classes
            for class_label in client_classes:
                class_data = data_by_class[class_label]
                num_samples = int(0.8 * len(class_data))
                samples = random.sample(class_data, min(num_samples, len(class_data)))
                for image, label in samples:
                    client_data.append(image)
                    client_labels.append(label)

            # Some other data
            all_other_data = []
            for class_label in range(10):
                if class_label not in client_classes:
                    all_other_data.extend(data_by_class[class_label])

            num_random = len(client_data) // 10
            if num_random > 0 and len(all_other_data) > 0:
                random_samples = random.sample(all_other_data, min(num_random, len(all_other_data)))
                for image, label in random_samples:
                    client_data.append(image)
                    client_labels.append(label)

            clients.append(AsyncFederatedClient(client_id, client_data, client_labels))

    return clients


# ========================================
# 📏 EVALUATION
# ========================================


def evaluate_global_model(model, test_loader):
    """Evaluate global model on test set"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    return accuracy, avg_loss


def print_model_stats(model):
    """Print model parameter statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")


# ========================================
# 🚀 ASYNCHRONOUS FEDERATED LEARNING EXPERIMENT
# ========================================

print(f"\n🎯 Setting up Asynchronous Federated Learning Experiment")
print(f"=" * 80)

# Experiment parameters
num_clients = 10
num_events = 200           # number of async update events
local_epochs = 2
learning_rate = 0.01
distribution_type = 'non-iid'  # 'iid' or 'non-iid'

print(f"\n🔧 Experiment Configuration (Async):")
print(f"   Number of clients: {num_clients}")
print(f"   Number of async events: {num_events}")
print(f"   Local epochs per event: {local_epochs}")
print(f"   Learning rate: {learning_rate}")
print(f"   Data distribution: {distribution_type.upper()}")

# Split data among clients
clients = split_data_among_clients(train_ds_full, num_clients, distribution=distribution_type)

# Create test DataLoader
test_loader = DataLoader(test_ds_full, batch_size=512, shuffle=False)

# Initialize global model and params
global_model = MLPModel(num_classes=10).to(DEVICE)
print(f"\n📊 Global Model (Async):")
print_model_stats(global_model)

global_params = {name: param.data.clone().cpu() for name, param in global_model.named_parameters()}
global_version = 0  # increments every time server applies an update

# Initialize all clients with initial global params
for client in clients:
    client.pull_from_server(global_params, global_version)

# Track metrics over async events
event_indices = []
event_accuracies = []
event_losses = []

print(f"\n🚀 Starting Asynchronous Federated Learning (FedAsync-style)...")
print(f"=" * 80)

# Async update loop (one client event at a time)
for event in range(1, num_events + 1):
    # 1) Pick a random client to update asynchronously
    client = random.choice(clients)

    # 2) Client trains starting from its (possibly stale) local snapshot
    updated_params, used_version = client.train_async(
        base_model=global_model,
        epochs=local_epochs,
        lr=learning_rate,
        batch_size=32,
    )

    # 3) Compute staleness and server mixing factor (simple staleness-aware update)
    staleness = global_version - used_version
    # Higher staleness → smaller alpha
    alpha_base = 0.6
    alpha = alpha_base / (1.0 + staleness)

    # 4) Apply async update to global params:
    #    w_{t+1} = (1 - alpha) * w_t + alpha * w_client
    for name in global_params.keys():
        global_params[name] = (1 - alpha) * global_params[name] + alpha * updated_params[name]

    global_version += 1

    # 5) Update global_model weights from global_params
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            param.data.copy_(global_params[name].to(DEVICE))

    # 6) Client pulls fresh global params (synchronization for this client only)
    client.pull_from_server(global_params, global_version)

    # 7) Periodically evaluate global model to observe oscillations
    if event % 10 == 0 or event == 1:
        acc, loss = evaluate_global_model(global_model, test_loader)
        event_indices.append(event)
        event_accuracies.append(acc)
        event_losses.append(loss)
        print(f"\n📊 After async event {event}/{num_events}:")
        print(f"   Used client: {client.client_id}, staleness: {staleness}")
        print(f"   Alpha: {alpha:.3f}")
        print(f"   Global Accuracy: {acc:.4f} ({acc*100:.2f}%), Loss: {loss:.4f}")


# ========================================
# 📊 RESULTS ANALYSIS & VISUALIZATION
# ========================================

print(f"\n" + "=" * 80)
print(f"ASYNC FEDERATED LEARNING RESULTS")
print(f"=" * 80)

print(f"\n📈 Final Performance (Async):")
print(f"   Final Accuracy: {event_accuracies[-1]:.4f} ({event_accuracies[-1]*100:.2f}%)")
print(f"   Best Accuracy: {max(event_accuracies):.4f} ({max(event_accuracies)*100:.2f}%)")

print(f"\n🎓 Observations (Async vs Sync FedAvg):")
print(f"   - Async updates can be based on STALE models (high staleness).")
print(f"   - This can cause the global model to oscillate or converge more slowly.")
print(f"   - Compare this curve with the synchronous FedAvg results to see the difference.")


def plot_async_federated_results():
    """Plot async global performance over events."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy over async events
    ax1.plot(event_indices, event_accuracies, 'b-o', linewidth=2, markersize=6, label='Async Global Accuracy')
    ax1.set_xlabel('Async Event', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Async Federated Learning: Accuracy vs Events', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1])

    # Plot loss over async events
    ax2.plot(event_indices, event_losses, 'r-o', linewidth=2, markersize=6, label='Async Global Loss')
    ax2.set_xlabel('Async Event', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title('Async Federated Learning: Loss vs Events', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('federated_learning_async_results.png', dpi=150, bbox_inches='tight')
    plt.show()


print(f"\n📊 Creating Asynchronous Federated Learning Visualization...")
plot_async_federated_results()

print(f"\n" + "=" * 80)
print(f"ASYNC FEDERATED LEARNING EXPERIMENT COMPLETE!")
print(f"=" * 80)


