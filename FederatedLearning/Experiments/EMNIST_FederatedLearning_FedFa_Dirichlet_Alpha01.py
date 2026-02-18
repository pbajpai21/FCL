import copy
import math
import random
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Optimize PyTorch for CPU multithreading
torch.set_num_threads(8)
print(f"PyTorch using {torch.get_num_threads()} threads")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Global tracking lists (initialized to avoid NameError in plots)
buff_event_indices: list = []
buff_accuracies: list = []
buff_losses: list = []
event_indices_la: list = []
event_accuracies_la: list = []
event_losses_la: list = []
event_indices_pf: list = []
event_accuracies_pf: list = []
event_losses_pf: list = []


# ========================================
# 🧹 DATA PREPARATION (EMNIST, DIRICHLET NON-IID)
# ========================================

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_ds_full = datasets.EMNIST(
    root="./data", split="balanced", train=True, download=True, transform=transform
)
test_ds_full = datasets.EMNIST(
    root="./data", split="balanced", train=False, download=True, transform=transform
)

full_num_classes = len(train_ds_full.classes)
print(f"EMNIST Balanced - total classes: {full_num_classes}")
print(f"Train dataset size: {len(train_ds_full)}")
print(f"Test dataset size: {len(test_ds_full)}")

# Restrict experiment to at most 10 labels (0-9)
MAX_LABELS = 20
allowed_labels = list(range(MAX_LABELS))
num_classes = MAX_LABELS
print(f"Using only first {MAX_LABELS} labels for experiment: {allowed_labels}")


# ========================================
# 🧠 MODEL DEFINITION
# ========================================


class MLPModel(nn.Module):
    """Simple MLP model for EMNIST classification"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ========================================
# 👥 CLIENT SIMULATION
# ========================================


class AsyncFederatedClient:
    """
    Represents a client in fully asynchronous FedFa-style training on EMNIST.
    """

    def __init__(self, client_id, local_data, local_labels):
        self.client_id = client_id
        self.local_data = local_data
        self.local_labels = local_labels
        self.data_size = len(local_data)

        # Local model snapshot (state dict) and version
        self.local_params = None
        self.model_version = 0

        print(f"  Client {client_id}: {self.data_size} samples")

    def pull_from_server(self, global_params, version: int):
        self.local_params = {k: v.clone() for k, v in global_params.items()}
        self.model_version = version

    def train_async(
        self, base_model, epochs: int = 1, lr: float = 0.01, batch_size: int = 32
    ):
        if self.local_params is None:
            raise ValueError(
                "Client has no local parameters. Call pull_from_server first."
            )

        local_model = copy.deepcopy(base_model)
        with torch.no_grad():
            for name, param in local_model.named_parameters():
                param.data.copy_(self.local_params[name])

        local_model.to(DEVICE)
        local_model.train()

        local_dataset = torch.utils.data.TensorDataset(
            torch.stack(self.local_data), torch.tensor(self.local_labels)
        )
        local_loader = DataLoader(local_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

        for _ in range(epochs):
            for images, labels in local_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        updated_params = {
            name: param.data.clone().cpu() for name, param in local_model.named_parameters()
        }
        return updated_params, self.model_version


def split_emnist_dirichlet_non_iid(dataset, num_clients: int, alpha: float = 0.1):
    """
    Dirichlet non-IID split (restricted to first MAX_LABELS labels):
      - For each class k in {0, ..., MAX_LABELS-1}, partition its samples
        across clients according to a Dirichlet(alpha) draw.
      - Smaller alpha => stronger non-IID (clients dominated by a few classes).
    """
    num_classes_local = MAX_LABELS

    # Group indices by label
    data_by_class = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        data_by_class[int(label)].append(idx)

    # Initialize list of indices for each client
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes_local):
        idxs = data_by_class[c]
        random.shuffle(idxs)
        n = len(idxs)
        if n == 0:
            continue

        # Dirichlet over clients for this class
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        # Convert proportions to split sizes (integers that sum to n)
        split_sizes = np.floor(proportions * n).astype(int)
        diff = n - int(split_sizes.sum())
        # Distribute any rounding remainder
        for i in range(diff):
            split_sizes[i % num_clients] += 1

        assert split_sizes.sum() == n

        start = 0
        for cid in range(num_clients):
            sz = split_sizes[cid]
            if sz > 0:
                client_indices[cid].extend(idxs[start : start + sz])
            start += sz

    clients = []
    client_label_sets = []
    for client_id in range(num_clients):
        idxs = client_indices[client_id]
        random.shuffle(idxs)
        client_data = []
        client_labels = []
        for idx in idxs:
            image, label = dataset[idx]
            client_data.append(image)
            client_labels.append(int(label))
        label_set = sorted(list(set(client_labels)))
        client_label_sets.append(label_set)
        print(
            f"  Client {client_id}: labels {label_set}, "
            f"samples {len(client_labels)}"
        )
        clients.append(AsyncFederatedClient(client_id, client_data, client_labels))

    return clients, client_label_sets


# ========================================
# 📏 EVALUATION HELPERS
# ========================================


def evaluate_global_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
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
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")


def evaluate_model_on_client_data(model, client, batch_size: int = 256):
    """
    Evaluate a global model on a single client's local data
    (used as that client's "test" set).
    """
    model.eval()
    dataset = torch.utils.data.TensorDataset(
        torch.stack(client.local_data), torch.tensor(client.local_labels)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total if total > 0 else 0.0


# ========================================
# 🧮 FEDFA SERVERS + FEDBUFF
# ========================================


class FedFaServer:
    """
    Exact FedFa-style server (Algorithm 1 in the paper):
      - Maintains a sliding window buffer S of size K = buffer_size
      - Each time a new local model w_l arrives, enqueue it
      - Once the buffer is full, set the global model to the
        uniform average of the K models in the buffer.
    """

    def __init__(self, global_model, buffer_size: int = 5):
        self.global_model = global_model
        self.global_params = {
            name: p.data.clone().cpu() for name, p in global_model.named_parameters()
        }
        self.global_version = 0
        self.buffer = deque(maxlen=buffer_size)  # this is S in the pseudocode
        self.buffer_size = buffer_size

    def apply_fedfa_update(self, client_params, client_id: int):
        """
        Enqueue the client's local model parameters.
        When the buffer is full (size K), set global model to the
        uniform average of all models currently in the buffer.
        """
        entry = {
            "params": {name: p.clone() for name, p in client_params.items()},
            "client_id": client_id,
        }
        self.buffer.append(entry)

        buffer_client_ids = [e["client_id"] for e in self.buffer]
        print(f"   Buffer client IDs (oldest→newest): {buffer_client_ids}")

        if len(self.buffer) < self.buffer_size:
            return

        K = len(self.buffer)
        new_global = {}
        for name in self.global_params.keys():
            acc = None
            for e in self.buffer:
                p = e["params"][name]
                acc = p.clone() if acc is None else acc + p
            new_global[name] = acc / float(K)

        self.global_params = {name: p.clone() for name, p in new_global.items()}
        self.global_version += 1

        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.data.copy_(self.global_params[name].to(DEVICE))


class FedFaServerParticipationFair:
    """
    FedFa-style server with participation-aware weighting:
      - Maintains a sliding window buffer S of size K = buffer_size
      - At most one entry per client in the buffer (latest model)
      - When the buffer is full, sets the global model to a
        participation-weighted average of the models in the buffer,
        where each client i has weight ~ 1 / sqrt(participation_count_i).
    """

    def __init__(self, global_model, buffer_size: int = 5):
        self.global_model = global_model
        self.global_params = {
            name: p.data.clone().cpu() for name, p in global_model.named_parameters()
        }
        self.global_version = 0
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

        self.participation_counts = defaultdict(int)
        self.eps = 1e-8

    def apply_fedfa_update(self, client_params, client_id: int):
        self.participation_counts[client_id] += 1

        entry = {
            "params": {name: p.clone() for name, p in client_params.items()},
            "client_id": client_id,
        }

        existing_index = None
        for i, e in enumerate(self.buffer):
            if e["client_id"] == client_id:
                existing_index = i
                break

        if existing_index is not None:
            self.buffer[existing_index] = entry
            print(f"   Replaced existing entry for client {client_id} (PF-FedFa)")
        else:
            self.buffer.append(entry)

        buffer_client_ids = [e["client_id"] for e in self.buffer]
        print(f"   [PF-FedFa] Buffer client IDs (oldest→newest): {buffer_client_ids}")

        if len(self.buffer) < self.buffer_size:
            return

        raw_weights = []
        for e in self.buffer:
            cid = e["client_id"]
            count = self.participation_counts[cid]
            w_tilde = 1.0 / math.sqrt(float(count) + self.eps)
            raw_weights.append(w_tilde)

        Z = sum(raw_weights)
        norm_weights = [w / Z for w in raw_weights]

        new_global = {}
        for name in self.global_params.keys():
            acc = None
            for e, w in zip(self.buffer, norm_weights):
                p = e["params"][name]
                weighted = w * p
                acc = weighted.clone() if acc is None else acc + weighted
            new_global[name] = acc

        self.global_params = {name: p.clone() for name, p in new_global.items()}
        self.global_version += 1

        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.data.copy_(self.global_params[name].to(DEVICE))


class FedBuffServer:
    """
    Simple FedBuff-style server:
      - Buffer of K pending updates (deltas),
      - When K updates collected, average deltas and apply to global.
    """

    def __init__(self, global_model, buffer_size: int = 5, eta_g: float = 1.0):
        self.global_model = global_model
        self.global_params = {
            name: p.data.clone().cpu() for name, p in global_model.named_parameters()
        }
        self.global_version = 0
        self.buffer = []
        self.buffer_size = buffer_size
        self.eta_g = eta_g

    def apply_update(self, client_params):
        delta = {}
        for name in self.global_params.keys():
            delta[name] = client_params[name] - self.global_params[name]
        self.buffer.append(delta)

        if len(self.buffer) < self.buffer_size:
            return

        new_global = {}
        for name in self.global_params.keys():
            acc = None
            for d in self.buffer:
                acc = d[name].clone() if acc is None else acc + d[name]
            avg_delta = acc / float(len(self.buffer))
            new_global[name] = self.global_params[name] + self.eta_g * avg_delta

        self.global_params = {name: p.clone() for name, p in new_global.items()}
        self.global_version += 1
        self.buffer = []

        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.data.copy_(self.global_params[name].to(DEVICE))


class FedFaServerLabelAware:
    """
    FedFa-style server with label-distribution-aware weighting:
      - Maintains a sliding window buffer S of size K = buffer_size
      - At most one entry per client in the buffer (latest model)
      - When the buffer is full, sets the global model to a
        label-aware weighted average of the models in the buffer,
        where each client i has a precomputed score reflecting how
        "rare" its labels are globally.
    """

    def __init__(self, global_model, buffer_size: int, client_label_scores: dict):
        self.global_model = global_model
        self.global_params = {
            name: p.data.clone().cpu() for name, p in global_model.named_parameters()
        }
        self.global_version = 0
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.client_label_scores = client_label_scores

    def apply_fedfa_update(self, client_params, client_id: int):
        entry = {
            "params": {name: p.clone() for name, p in client_params.items()},
            "client_id": client_id,
        }

        existing_index = None
        for i, e in enumerate(self.buffer):
            if e["client_id"] == client_id:
                existing_index = i
                break

        if existing_index is not None:
            self.buffer[existing_index] = entry
            print(f"   Replaced existing entry for client {client_id} (LabelAware-FedFa)")
        else:
            self.buffer.append(entry)

        buffer_client_ids = [e["client_id"] for e in self.buffer]
        print(f"   [LabelAware-FedFa] Buffer client IDs (oldest→newest): {buffer_client_ids}")

        if len(self.buffer) < self.buffer_size:
            return

        raw_weights = []
        for e in self.buffer:
            cid = e["client_id"]
            w_tilde = self.client_label_scores.get(cid, 1.0)
            raw_weights.append(max(w_tilde, 1e-8))

        Z = sum(raw_weights)
        norm_weights = [w / Z for w in raw_weights]

        new_global = {}
        for name in self.global_params.keys():
            acc = None
            for e, w in zip(self.buffer, norm_weights):
                p = e["params"][name]
                weighted = w * p
                acc = weighted.clone() if acc is None else acc + weighted
            new_global[name] = acc

        self.global_params = {name: p.clone() for name, p in new_global.items()}
        self.global_version += 1

        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.data.copy_(self.global_params[name].to(DEVICE))


# ========================================
# 🚀 EMNIST FEDFA EXPERIMENT (DIRICHLET α=0.1)
# ========================================

print(
    f"\n🎯 Setting up EMNIST Federated Learning Experiments "
    f"(FedAvg vs FedFa variants, Dirichlet non-IID, alpha=0.1)"
)
print("=" * 80)

num_clients = 15
num_events = 1000
local_epochs = 2
local_lr = 0.01
buffer_size = 10
dirichlet_alpha = 0.1

print("\n🔧 Base Configuration:")
print(f"   Number of clients: {num_clients}")
print(f"   Number of async events (FedFa): {num_events}")
print(f"   Local epochs per round/event: {local_epochs}")
print(f"   Client learning rate (η_ell): {local_lr}")
print(f"   FedFa buffer size K: {buffer_size}")
print(f"   Non-IID pattern: Dirichlet with alpha={dirichlet_alpha}")

# Build Dirichlet non-IID clients (strong heterogeneity, alpha=0.1)
clients, client_label_sets = split_emnist_dirichlet_non_iid(
    train_ds_full, num_clients, alpha=dirichlet_alpha
)

# ----- Label-aware metadata (for Label-Aware FedFa) -----
global_label_counts = defaultdict(int)
for client in clients:
    for y in client.local_labels:
        global_label_counts[int(y)] += 1
total_train_samples = sum(global_label_counts.values())
global_label_freq = {
    c: cnt / float(total_train_samples) for c, cnt in global_label_counts.items()
}

label_rarity = {c: 1.0 / (freq + 1e-12) for c, freq in global_label_freq.items()}

client_label_scores = {}
for client in clients:
    per_client_counts = defaultdict(int)
    for y in client.local_labels:
        per_client_counts[int(y)] += 1
    total_c = float(len(client.local_labels))
    score = 0.0
    for lab, cnt in per_client_counts.items():
        p_c = cnt / total_c
        score += p_c * label_rarity[lab]
    client_label_scores[client.client_id] = score

print("\n🔍 Label-aware FedFa: example client label scores (first few):")
for cid in range(min(10, num_clients)):
    print(f"   Client {cid}: label-aware score={client_label_scores[cid]:.4f}")

# Global test loader: restrict to the same first MAX_LABELS labels
filtered_test_indices = []
for idx in range(len(test_ds_full)):
    _, label = test_ds_full[idx]
    if int(label) in allowed_labels:
        filtered_test_indices.append(idx)

print(
    f"Original test size: {len(test_ds_full)}, "
    f"filtered test size (labels {allowed_labels}): {len(filtered_test_indices)}"
)
test_subset = torch.utils.data.Subset(test_ds_full, filtered_test_indices)
test_loader = DataLoader(test_subset, batch_size=512, shuffle=False)


# ========================================
# 🔁 FedAvg (synchronous) baseline
# ========================================

print(f"\n================ FedAvg (synchronous) ================\n")

global_model_fedavg = MLPModel(num_classes=num_classes).to(DEVICE)
print(f"\n📊 Global Model (EMNIST FedAvg):")
print_model_stats(global_model_fedavg)

rounds_fedavg = num_events // num_clients
fedavg_round_indices = []
fedavg_accuracies = []
fedavg_losses = []

for rnd in range(1, rounds_fedavg + 1):
    client_states = []
    client_sizes = []

    for client in clients:
        local_model = copy.deepcopy(global_model_fedavg).to(DEVICE)
        local_model.train()

        local_dataset = torch.utils.data.TensorDataset(
            torch.stack(client.local_data), torch.tensor(client.local_labels)
        )
        local_loader = DataLoader(local_dataset, batch_size=32, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=local_lr)

        for _ in range(local_epochs):
            for images, labels in local_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        client_states.append(
            {name: p.data.clone().cpu() for name, p in local_model.named_parameters()}
        )
        client_sizes.append(client.data_size)

    total_samples = float(sum(client_sizes))
    new_global = {}
    for name in client_states[0].keys():
        agg = None
        for state, n_k in zip(client_states, client_sizes):
            w = n_k / total_samples
            v = state[name]
            agg = w * v if agg is None else agg + w * v
        new_global[name] = agg

    with torch.no_grad():
        for name, param in global_model_fedavg.named_parameters():
            param.data.copy_(new_global[name].to(DEVICE))

    acc, loss = evaluate_global_model(global_model_fedavg, test_loader)
    fedavg_round_indices.append(rnd)
    fedavg_accuracies.append(acc)
    fedavg_losses.append(loss)
    print(f"\n📊 After FedAvg round {rnd}/{rounds_fedavg}:")
    print(f"   Global Accuracy: {acc:.4f} ({acc*100:.2f}%), Loss: {loss:.4f}")

print("\n" + "=" * 80)
print("EMNIST FEDAVG SYNCHRONOUS FEDERATED LEARNING RESULTS")
print("=" * 80)
if fedavg_accuracies:
    print("\n📈 Final FedAvg Performance:")
    print(
        f"   Final Accuracy: {fedavg_accuracies[-1]:.4f} "
        f"({fedavg_accuracies[-1]*100:.2f}%)"
    )
    print(
        f"   Best Accuracy: {max(fedavg_accuracies):.4f} "
        f"({max(fedavg_accuracies)*100:.2f}%)"
    )
else:
    print("\nNo FedAvg evaluation points collected.")


# ========================================
# 🚀 Baseline FedFa (asynchronous)
# ========================================

print(f"\n================ FedFa (asynchronous) ================\n")

global_model_fedfa = MLPModel(num_classes=num_classes).to(DEVICE)
print(f"\n📊 Global Model (EMNIST FedFa-style):")
print_model_stats(global_model_fedfa)

server = FedFaServer(global_model_fedfa, buffer_size=buffer_size)

for client in clients:
    client.pull_from_server(server.global_params, server.global_version)

event_indices = []
event_accuracies = []
event_losses = []

print(
    "\n🚀 Starting EMNIST FedFa-Style Fully Asynchronous Federated "
    "Learning (Dirichlet non-IID)..."
)
print("=" * 80)

for event in range(1, num_events + 1):
    client = random.choice(clients)

    updated_params, used_version = client.train_async(
        base_model=global_model_fedfa,
        epochs=local_epochs,
        lr=local_lr,
        batch_size=32,
    )

    staleness = server.global_version - used_version

    server.apply_fedfa_update(
        updated_params,
        client_id=client.client_id,
    )

    client.pull_from_server(server.global_params, server.global_version)

    if event % 20 == 0 or event == 1:
        acc, loss = evaluate_global_model(global_model_fedfa, test_loader)
        event_indices.append(event)
        event_accuracies.append(acc)
        event_losses.append(loss)
        print(f"\n📊 After EMNIST FedFa async event {event}/{num_events}:")
        print(f"   Used client: {client.client_id}, staleness: {staleness}")
        print(f"   Current buffer size: {len(server.buffer)}")
        print(
            f"   Global Accuracy: {acc:.4f} ({acc*100:.2f}%), Loss: {loss:.4f}"
        )

print("\n" + "=" * 80)
print("EMNIST FEDFA-STYLE FULLY ASYNC FEDERATED LEARNING RESULTS (BASELINE)")
print("=" * 80)

if event_accuracies:
    print("\n📈 Final Performance (Baseline EMNIST FedFa-style):")
    print(
        f"   Final Accuracy: {event_accuracies[-1]:.4f} "
        f"({event_accuracies[-1]*100:.2f}%)"
    )
    print(
        f"   Best Accuracy: {max(event_accuracies):.4f} "
        f"({max(event_accuracies)*100:.2f}%)"
    )
else:
    print("\nNo baseline FedFa evaluation points collected.")


# ========================================
# 🚀 FedBuff (asynchronous)
# ========================================

buff_event_indices = []
buff_accuracies = []
buff_losses = []

print(f"\n================ FedBuff (asynchronous) ================\n")

global_model_fedbuff = MLPModel(num_classes=num_classes).to(DEVICE)
print(f"\n📊 Global Model (EMNIST FedBuff-style):")
print_model_stats(global_model_fedbuff)

server_buff = FedBuffServer(global_model_fedbuff, buffer_size=buffer_size, eta_g=1.0)

for client in clients:
    client.pull_from_server(server_buff.global_params, server_buff.global_version)

print("\n🚀 Starting EMNIST FedBuff Async Federated Learning (Dirichlet non-IID)...")
print("=" * 80)

for event in range(1, num_events + 1):
    client = random.choice(clients)

    updated_params, _ = client.train_async(
        base_model=global_model_fedbuff,
        epochs=local_epochs,
        lr=local_lr,
        batch_size=32,
    )

    server_buff.apply_update(updated_params)
    client.pull_from_server(server_buff.global_params, server_buff.global_version)

    if event % 20 == 0 or event == 1:
        acc, loss = evaluate_global_model(global_model_fedbuff, test_loader)
        buff_event_indices.append(event)
        buff_accuracies.append(acc)
        buff_losses.append(loss)
        print(f"\n📊 After FedBuff async event {event}/{num_events}:")
        print(f"   Used client: {client.client_id}")
        print(f"   Current FedBuff buffer size: {len(server_buff.buffer)}")
        print(
            f"   Global Accuracy (FedBuff): {acc:.4f} "
            f"({acc*100:.2f}%), Loss: {loss:.4f}"
        )

print("\n" + "=" * 80)
print("EMNIST FEDBUFF FULLY ASYNC FEDERATED LEARNING RESULTS")
print("=" * 80)


# ========================================
# 🚀 Participation-Fair FedFa (asynchronous)
# ========================================

print(
    f"\n================ Participation-Fair FedFa (asynchronous) "
    f"===============\n"
)

global_model_fedfa_pf = MLPModel(num_classes=num_classes).to(DEVICE)
print(f"\n📊 Global Model (EMNIST Participation-Fair FedFa):")
print_model_stats(global_model_fedfa_pf)

server_pf = FedFaServerParticipationFair(global_model_fedfa_pf, buffer_size=buffer_size)

for client in clients:
    client.pull_from_server(server_pf.global_params, server_pf.global_version)

event_indices_pf = []
event_accuracies_pf = []
event_losses_pf = []

print(
    "\n🚀 Starting EMNIST Participation-Fair FedFa Async Federated "
    "Learning (Dirichlet non-IID)..."
)
print("=" * 80)

for event in range(1, num_events + 1):
    client = random.choice(clients)

    updated_params, used_version = client.train_async(
        base_model=global_model_fedfa_pf,
        epochs=local_epochs,
        lr=local_lr,
        batch_size=32,
    )

    staleness = server_pf.global_version - used_version

    server_pf.apply_fedfa_update(
        updated_params,
        client_id=client.client_id,
    )

    client.pull_from_server(server_pf.global_params, server_pf.global_version)

    if event % 20 == 0 or event == 1:
        acc, loss = evaluate_global_model(global_model_fedfa_pf, test_loader)
        event_indices_pf.append(event)
        event_accuracies_pf.append(acc)
        event_losses_pf.append(loss)
        print(
            f"\n📊 After Participation-Fair FedFa async event "
            f"{event}/{num_events}:"
        )
        print(f"   Used client: {client.client_id}, staleness: {staleness}")
        print(
            f"   Current Participation-Fair FedFa buffer size: "
            f"{len(server_pf.buffer)}"
        )
        print(
            f"   Global Accuracy (Participation-Fair FedFa): {acc:.4f} "
            f"({acc*100:.2f}%), Loss: {loss:.4f}"
        )

print("\n" + "=" * 80)
print("EMNIST PARTICIPATION-FAIR FEDFA FULLY ASYNC FEDERATED LEARNING RESULTS")
print("=" * 80)


# ========================================
# 🚀 Label-Aware FedFa (asynchronous)
# ========================================

print(f"\n================ Label-Aware FedFa (asynchronous) ================\n")

global_model_fedfa_la = MLPModel(num_classes=num_classes).to(DEVICE)
print(f"\n📊 Global Model (EMNIST Label-Aware FedFa):")
print_model_stats(global_model_fedfa_la)

server_la = FedFaServerLabelAware(
    global_model_fedfa_la, buffer_size=buffer_size, client_label_scores=client_label_scores
)

for client in clients:
    client.pull_from_server(server_la.global_params, server_la.global_version)

print(
    "\n🚀 Starting EMNIST Label-Aware FedFa Async Federated "
    "Learning (Dirichlet non-IID)..."
)
print("=" * 80)

event_indices_la = []
event_accuracies_la = []
event_losses_la = []

for event in range(1, num_events + 1):
    client = random.choice(clients)

    updated_params, used_version = client.train_async(
        base_model=global_model_fedfa_la,
        epochs=local_epochs,
        lr=local_lr,
        batch_size=32,
    )

    staleness = server_la.global_version - used_version

    server_la.apply_fedfa_update(
        updated_params,
        client_id=client.client_id,
    )

    client.pull_from_server(server_la.global_params, server_la.global_version)

    if event % 20 == 0 or event == 1:
        acc, loss = evaluate_global_model(global_model_fedfa_la, test_loader)
        event_indices_la.append(event)
        event_accuracies_la.append(acc)
        event_losses_la.append(loss)
        print(f"\n📊 After Label-Aware FedFa async event {event}/{num_events}:")
        print(f"   Used client: {client.client_id}, staleness: {staleness}")
        print(f"   Current Label-Aware FedFa buffer size: {len(server_la.buffer)}")
        print(
            f"   Global Accuracy (Label-Aware FedFa): {acc:.4f} "
            f"({acc*100:.2f}%), Loss: {loss:.4f}"
        )

print("\n" + "=" * 80)
print("EMNIST LABEL-AWARE FEDFA FULLY ASYNC FEDERATED LEARNING RESULTS")
print("=" * 80)

if event_accuracies_la:
    print("\n📈 Final Performance (Label-Aware EMNIST FedFa):")
    print(
        f"   Final Accuracy: {event_accuracies_la[-1]:.4f} "
        f"({event_accuracies_la[-1]*100:.2f}%)"
    )
    print(
        f"   Best Accuracy: {max(event_accuracies_la):.4f} "
        f"({max(event_accuracies_la)*100:.2f}%)"
    )
else:
    print("\nNo Label-Aware FedFa evaluation points collected.")


print(
    "\n📊 Per-client accuracy on final global models "
    "(FedAvg vs FedFa / FedBuff / PF-FedFa / Label-Aware):"
)

per_client_acc_fedavg = []
per_client_acc_baseline = []
per_client_acc_buff = []
per_client_acc_pf = []
per_client_acc_la = []
for client in clients:
    acc_fedavg = evaluate_model_on_client_data(global_model_fedavg, client)
    acc_base = evaluate_model_on_client_data(global_model_fedfa, client)
    acc_buff = evaluate_model_on_client_data(global_model_fedbuff, client)
    acc_pf = evaluate_model_on_client_data(global_model_fedfa_pf, client)
    acc_la = evaluate_model_on_client_data(global_model_fedfa_la, client)
    per_client_acc_fedavg.append(acc_fedavg)
    per_client_acc_baseline.append(acc_base)
    per_client_acc_buff.append(acc_buff)
    per_client_acc_pf.append(acc_pf)
    per_client_acc_la.append(acc_la)
    print(
        f"   Client {client.client_id}: "
        f"FedAvg={acc_fedavg:.4f} ({acc_fedavg*100:.2f}%), "
        f"FedFa={acc_base:.4f} ({acc_base*100:.2f}%), "
        f"FedBuff={acc_buff:.4f} ({acc_buff*100:.2f}%), "
        f"PF-FedFa={acc_pf:.4f} ({acc_pf*100:.2f}%), "
        f"LabelAware-FedFa={acc_la:.4f} ({acc_la*100:.2f}%)"
    )

if per_client_acc_fedavg:
    mean_fedavg = sum(per_client_acc_fedavg) / len(per_client_acc_fedavg)
    print(
        f"\n   FedAvg mean per-client accuracy: "
        f"{mean_fedavg:.4f} ({mean_fedavg*100:.2f}%)"
    )
if per_client_acc_buff:
    mean_buff = sum(per_client_acc_buff) / len(per_client_acc_buff)
    print(
        f"   FedBuff mean per-client accuracy: "
        f"{mean_buff:.4f} ({mean_buff*100:.2f}%)"
    )
if per_client_acc_pf:
    mean_pf = sum(per_client_acc_pf) / len(per_client_acc_pf)
    print(
        f"   Participation-Fair FedFa mean per-client accuracy: "
        f"{mean_pf:.4f} ({mean_pf*100:.2f}%)"
    )
if per_client_acc_la:
    mean_la = sum(per_client_acc_la) / len(per_client_acc_la)
    print(
        f"   Label-Aware FedFa mean per-client accuracy: "
        f"{mean_la:.4f} ({mean_la*100:.2f}%)"
    )


def plot_emnist_fedfa_results():
    if (
        not event_indices
        and not buff_event_indices
        and not event_indices_pf
        and not event_indices_la
    ):
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if buff_event_indices:
        ax1.plot(
            buff_event_indices,
            buff_accuracies,
            "r-o",
            linewidth=2,
            markersize=6,
            label="FedBuff Accuracy",
        )
    if event_indices:
        ax1.plot(
            event_indices,
            event_accuracies,
            "b-x",
            linewidth=2,
            markersize=6,
            label="FedFa (baseline) Accuracy",
        )
    if event_indices_pf:
        ax1.plot(
            event_indices_pf,
            event_accuracies_pf,
            "y-*",
            linewidth=2,
            markersize=6,
            label="Participation-Fair FedFa Accuracy",
        )
    if event_indices_la:
        ax1.plot(
            event_indices_la,
            event_accuracies_la,
            "g-^",
            linewidth=2,
            markersize=6,
            label="Label-Aware FedFa Accuracy",
        )
    ax1.set_xlabel("Async Event", fontsize=12)
    ax1.set_ylabel("Test Accuracy", fontsize=12)
    ax1.set_title(
        "EMNIST FedFa Async FL (Dirichlet α=0.1): Accuracy vs Events",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1])

    if buff_event_indices:
        ax2.plot(
            buff_event_indices,
            buff_losses,
            "r-o",
            linewidth=2,
            markersize=6,
            label="FedBuff Loss",
        )
    if event_indices:
        ax2.plot(
            event_indices,
            event_losses,
            "b-x",
            linewidth=2,
            markersize=6,
            label="FedFa (baseline) Loss",
        )
    if event_indices_pf:
        ax2.plot(
            event_indices_pf,
            event_losses_pf,
            "y-*",
            linewidth=2,
            markersize=6,
            label="Participation-Fair FedFa Loss",
        )
    if event_indices_la:
        ax2.plot(
            event_indices_la,
            event_losses_la,
            "g-^",
            linewidth=2,
            markersize=6,
            label="Label-Aware FedFa Loss",
        )
    ax2.set_xlabel("Async Event", fontsize=12)
    ax2.set_ylabel("Test Loss", fontsize=12)
    ax2.set_title(
        "EMNIST FedFa Async FL (Dirichlet α=0.1): Loss vs Events",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        "emnist_federated_learning_fedfa_dirichlet_alpha01.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


print(
    "\n📊 Creating EMNIST FedFa-Style Asynchronous Federated Learning Visualization "
    "(FedBuff vs FedFa vs PF-FedFa vs Label-Aware, Dirichlet α=0.1)..."
)
plot_emnist_fedfa_results()

print("\n" + "=" * 80)
print("EMNIST FEDFA-STYLE FULLY ASYNC (DIRICHLET α=0.1) EXPERIMENTS COMPLETE!")
print("=" * 80)


def plot_emnist_fedavg_vs_fedfa():
    if not fedavg_round_indices or not event_indices:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(fedavg_round_indices, fedavg_accuracies, "g-o", label="FedAvg (sync)")

    plt.xlabel("FedAvg Rounds", fontsize=12)
    plt.ylabel("Test Accuracy (same global test set)", fontsize=12)
    plt.title(
        "EMNIST FedAvg (Dirichlet non-IID, same clients)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "emnist_federated_learning_fedavg_vs_fedfa_dirichlet_alpha01.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


print("\n📊 Creating EMNIST FedAvg vs FedFa Comparison Visualization (Dirichlet α=0.1)...")
plot_emnist_fedavg_vs_fedfa()

print("\n" + "=" * 80)
print("EMNIST FEDAVG VS FEDFA (DIRICHLET α=0.1) COMPARISON COMPLETE!")
print("=" * 80)


print("\n" + "=" * 80)
print("EMNIST FINAL GLOBAL ACCURACY SUMMARY (DIRICHLET α=0.1)")
print("=" * 80)


def _print_final_best(name, acc_list):
    if acc_list:
        final_acc = acc_list[-1]
        best_acc = max(acc_list)
        print(
            f"{name:30s} | Final: {final_acc:.4f} ({final_acc*100:.2f}%)"
            f" | Best: {best_acc:.4f} ({best_acc*100:.2f}%)"
        )
    else:
        print(f"{name:30s} | no evaluation points collected")


_print_final_best("FedAvg (sync)", fedavg_accuracies)
_print_final_best("FedBuff (async)", buff_accuracies)
_print_final_best("FedFa (baseline async)", event_accuracies)
_print_final_best("PF-FedFa (async)", event_accuracies_pf)
_print_final_best("Label-Aware FedFa (async)", event_accuracies_la)


######### Checking actual accuracy on rare, medium, and common labels #########


def plot_rarity_grouped_accuracy(model, test_loader, global_label_freq):
    """
    Groups EMNIST classes by their frequency and calculates accuracy for each group.
    """
    freqs = list(global_label_freq.values())
    low_thresh = np.percentile(freqs, 20)
    high_thresh = np.percentile(freqs, 80)

    group_map = {}  # label -> group_name
    for label, freq in global_label_freq.items():
        if freq <= low_thresh:
            group_map[label] = "Rare Labels"
        elif freq >= high_thresh:
            group_map[label] = "Common Labels"
        else:
            group_map[label] = "Medium Labels"

    model.eval()
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for l, p in zip(labels, predicted):
                class_total[l.item()] += 1
                if l.item() == p.item():
                    class_correct[l.item()] += 1

    group_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for label, correct in class_correct.items():
        g_name = group_map.get(label, "Medium Labels")
        group_stats[g_name]["correct"] += correct
        group_stats[g_name]["total"] += class_total[label]

    names = ["Common Labels", "Medium Labels", "Rare Labels"]
    accuracies = [
        group_stats[n]["correct"] / max(1, group_stats[n]["total"]) for n in names
    ]

    return names, accuracies


names, base_accs = plot_rarity_grouped_accuracy(
    global_model_fedfa, test_loader, global_label_freq
)
_, la_accs = plot_rarity_grouped_accuracy(
    global_model_fedfa_la, test_loader, global_label_freq
)

plt.figure(figsize=(10, 6))
x = np.arange(len(names))
width = 0.35

plt.bar(x - width / 2, base_accs, width, label="Baseline FedFa", color="skyblue")
plt.bar(x + width / 2, la_accs, width, label="Label-Aware FedFa", color="orange")

plt.ylabel("Accuracy")
plt.title("Accuracy Improvement on Rare vs. Common Labels (Dirichlet α=0.1)")
plt.xticks(x, names)
plt.legend()
plt.ylim(0, 1.0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

