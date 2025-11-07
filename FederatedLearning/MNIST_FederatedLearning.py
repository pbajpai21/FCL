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
# 🧠 FEDERATED LEARNING IMPLEMENTATION
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


class FederatedClient:
    """
    Represents a client in federated learning.
    Each client has local data and can train the model locally.
    """
    def __init__(self, client_id, local_data, local_labels):
        self.client_id = client_id
        self.local_data = local_data
        self.local_labels = local_labels
        self.data_size = len(local_data)
        
        print(f"  Client {client_id}: {self.data_size} samples")
    
    def train_local(self, global_model, epochs=1, lr=0.01, batch_size=32):
        """
        Train model locally on client's data.
        
        Returns:
            Updated model parameters
            Number of training steps
        """
        # Copy global model
        local_model = copy.deepcopy(global_model)
        local_model.train()
        
        # Create DataLoader for local data
        local_dataset = torch.utils.data.TensorDataset(
            torch.stack(self.local_data),
            torch.tensor(self.local_labels)
        )
        local_loader = DataLoader(local_dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
        
        # Local training
        num_steps = 0
        for epoch in range(epochs):
            for images, labels in local_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                # Forward pass
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                num_steps += 1
        
        # Return updated parameters (not the model object)
        return {name: param.data.clone() for name, param in local_model.named_parameters()}, num_steps


def split_data_among_clients(dataset, num_clients, distribution='iid'):
    """
    Split dataset among clients.
    
    Args:
        dataset: Full dataset
        num_clients: Number of clients
        distribution: 'iid' or 'non-iid'
    
    Returns:
        List of FederatedClient objects
    """
    clients = []
    
    if distribution == 'iid':
        # IID: Randomly shuffle and split
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
            
            clients.append(FederatedClient(i, client_data, client_labels))
    
    elif distribution == 'non-iid':
        # Non-IID: Each client gets mostly one or two classes
        print(f"\n📊 Creating {num_clients} clients with Non-IID data distribution...")
        print(f"   Each client will have data primarily from 1-2 classes")
        
        # Organize data by class
        data_by_class = defaultdict(list)
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            data_by_class[label].append((image, label))
        
        # Assign classes to clients
        classes = list(range(10))
        for client_id in range(num_clients):
            # Each client gets 1-2 classes
            num_classes_for_client = random.randint(1, 2)
            client_classes = random.sample(classes, num_classes_for_client)
            
            client_data = []
            client_labels = []
            
            # Assign data from selected classes
            for class_label in client_classes:
                class_data = data_by_class[class_label]
                # Take 80% of data from assigned classes
                num_samples = int(0.8 * len(class_data))
                samples = random.sample(class_data, min(num_samples, len(class_data)))
                
                for image, label in samples:
                    client_data.append(image)
                    client_labels.append(label)
            
            # Add some random data from other classes (10% of client data)
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
            
            clients.append(FederatedClient(client_id, client_data, client_labels))
    
    return clients


def federated_averaging(global_model, client_updates, client_weights):
    """
    Federated Averaging (FedAvg) algorithm.
    
    Aggregates client model updates using weighted average.
    
    Args:
        global_model: Current global model parameters
        client_updates: List of client model parameter dictionaries
        client_weights: List of weights (typically number of samples)
    
    Returns:
        Updated global model parameters
    """
    # Calculate total weight
    total_weight = sum(client_weights)
    
    # Initialize aggregated parameters
    aggregated_params = {}
    for key in global_model.keys():
        aggregated_params[key] = torch.zeros_like(global_model[key])
    
    # Weighted average
    for update, weight in zip(client_updates, client_weights):
        for key in aggregated_params.keys():
            aggregated_params[key] += (weight / total_weight) * update[key]
    
    return aggregated_params


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
# 🎯 FEDERATED LEARNING EXPERIMENT
# ========================================

print(f"\n🎯 Setting up Federated Learning Experiment")
print(f"=" * 80)

# Experiment parameters
num_clients = 10
num_rounds = 20
local_epochs = 3
learning_rate = 0.01
fraction_of_clients = 0.5  # Select 50% of clients per round
distribution_type = 'non-iid'  # 'iid' or 'non-iid'

print(f"\n🔧 Experiment Configuration:")
print(f"   Number of clients: {num_clients}")
print(f"   Number of communication rounds: {num_rounds}")
print(f"   Local epochs per round: {local_epochs}")
print(f"   Learning rate: {learning_rate}")
print(f"   Clients per round: {int(fraction_of_clients * num_clients)}")
print(f"   Data distribution: {distribution_type.upper()}")

# Split data among clients
clients = split_data_among_clients(train_ds_full, num_clients, distribution=distribution_type)

# Create test DataLoader
test_loader = DataLoader(test_ds_full, batch_size=512, shuffle=False)

# Initialize global model
global_model = MLPModel(num_classes=10).to(DEVICE)
print(f"\n📊 Global Model:")
print_model_stats(global_model)

# Store results
round_accuracies = []
round_losses = []

print(f"\n🚀 Starting Federated Learning Training...")
print(f"=" * 80)

# Federated Learning Training Loop
for round_num in range(1, num_rounds + 1):
    print(f"\n🔄 Round {round_num}/{num_rounds}")
    print(f"-" * 60)
    
    # 1. Select subset of clients (simulating client availability)
    num_selected = max(1, int(fraction_of_clients * num_clients))
    selected_clients = random.sample(clients, num_selected)
    print(f"📡 Selected {num_selected} clients: {[c.client_id for c in selected_clients]}")
    
    # 2. Get current global model parameters
    global_params = {name: param.data.clone() for name, param in global_model.named_parameters()}
    
    # 3. Local training on selected clients
    client_updates = []
    client_weights = []
    client_losses = []
    
    for client in selected_clients:
        # Update global model with current parameters
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                param.data.copy_(global_params[name])
        
        # Local training
        updated_params, num_steps = client.train_local(
            global_model,
            epochs=local_epochs,
            lr=learning_rate
        )
        
        client_updates.append(updated_params)
        client_weights.append(client.data_size)
    
    # 4. Aggregate updates (Federated Averaging)
    print(f"🔀 Aggregating updates from {len(client_updates)} clients...")
    new_global_params = federated_averaging(
        global_params,
        client_updates,
        client_weights
    )
    
    # 5. Update global model
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            param.data.copy_(new_global_params[name])
    
    # 6. Evaluate global model
    accuracy, loss = evaluate_global_model(global_model, test_loader)
    round_accuracies.append(accuracy)
    round_losses.append(loss)
    
    print(f"📊 Global Model Performance:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Loss: {loss:.4f}")

# ========================================
# 📊 RESULTS ANALYSIS
# ========================================

print(f"\n" + "=" * 80)
print(f"FEDERATED LEARNING RESULTS")
print(f"=" * 80)

print(f"\n📈 Final Performance:")
print(f"   Final Accuracy: {round_accuracies[-1]:.4f} ({round_accuracies[-1]*100:.2f}%)")
print(f"   Best Accuracy: {max(round_accuracies):.4f} ({max(round_accuracies)*100:.2f}%)")
print(f"   Improvement: {round_accuracies[-1] - round_accuracies[0]:.4f} ({round_accuracies[-1] - round_accuracies[0]*100:.2f}%)")

print(f"\n🎓 Key Insights:")
print(f"   - {num_clients} clients participated in federated learning")
print(f"   - Data distribution: {distribution_type.upper()}")
print(f"   - Total communication rounds: {num_rounds}")
print(f"   - Each client trained locally without sharing raw data")
print(f"   - Global model improved through weighted aggregation")

# ========================================
# 📈 VISUALIZATION
# ========================================

def plot_federated_learning_results():
    """Create visualization of federated learning progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy over rounds
    rounds = range(1, num_rounds + 1)
    ax1.plot(rounds, round_accuracies, 'b-o', linewidth=2, markersize=6, label='Global Model Accuracy')
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Federated Learning: Accuracy vs Rounds', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1])
    
    # Plot loss over rounds
    ax2.plot(rounds, round_losses, 'r-o', linewidth=2, markersize=6, label='Global Model Loss')
    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title('Federated Learning: Loss vs Rounds', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('federated_learning_results.png', dpi=150, bbox_inches='tight')
    plt.show()

print(f"\n📊 Creating Federated Learning Visualization...")
plot_federated_learning_results()

# Print comparison with centralized training
print(f"\n" + "=" * 80)
print(f"COMPARISON: Federated vs Centralized Learning")
print(f"=" * 80)

print(f"\n📊 To compare with centralized training:")
print(f"   1. Federated Learning: {round_accuracies[-1]*100:.2f}% (this experiment)")
print(f"   2. Centralized: Train model on all data at once")
print(f"   3. Typically: Centralized > Federated (due to non-IID, limited clients)")
print(f"   4. Trade-off: Privacy vs Performance")

print(f"\n💡 Federated Learning Advantages:")
print(f"   ✅ Privacy: Data never leaves clients")
print(f"   ✅ Scalability: Works with many devices")
print(f"   ✅ Compliance: Meets privacy regulations")
print(f"   ⚠️  Communication: Requires multiple rounds")
print(f"   ⚠️  Non-IID: Can hurt performance")

print(f"\n🎉 Federated Learning Experiment Complete!")

