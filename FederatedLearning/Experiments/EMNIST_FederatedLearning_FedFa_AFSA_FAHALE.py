import copy
import math
import random
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.set_num_threads(8)
print(f"PyTorch using {torch.get_num_threads()} threads")

random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ----- Data Preparation (EMNIST) -----
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

MAX_LABELS = 20
allowed_labels = list(range(MAX_LABELS))
num_classes = MAX_LABELS
print(f"Using only first {MAX_LABELS} labels for experiment: {allowed_labels}")


class MLPModel(nn.Module):
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


class AsyncFederatedClient:
    def __init__(self, client_id, local_data, local_labels, verbose: bool = True):
        self.client_id = client_id
        self.local_data = local_data
        self.local_labels = local_labels
        self.data_size = len(local_data)
        self.local_params = None
        self.model_version = 0
        if verbose:
            print(f"  Client {client_id}: {self.data_size} samples")

    def pull_from_server(self, global_params, version: int):
        self.local_params = {k: v.clone() for k, v in global_params.items()}
        self.model_version = version

    def train_async(self, base_model, epochs: int = 1, lr: float = 0.01, batch_size: int = 32):
        if self.local_params is None:
            raise ValueError("Client has no local parameters. Call pull_from_server first.")

        local_model = copy.deepcopy(base_model)
        with torch.no_grad():
            for name, param in local_model.named_parameters():
                param.data.copy_(self.local_params[name])

        local_model.to(DEVICE)
        local_model.train()

        local_dataset = torch.utils.data.TensorDataset(
            torch.stack(self.local_data),
            torch.tensor(self.local_labels),
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

        updated_params = {name: p.data.clone().cpu() for name, p in local_model.named_parameters()}
        return updated_params, self.model_version

    def _build_local_loader(self, batch_size: int = 32):
        local_dataset = torch.utils.data.TensorDataset(
            torch.stack(self.local_data),
            torch.tensor(self.local_labels),
        )
        return DataLoader(local_dataset, batch_size=batch_size, shuffle=True)

    def _estimate_classwise_stats(self, model, max_batches: int = 4, batch_size: int = 64):
        """
        Returns:
          class_loss_mean: dict[label] -> mean CE loss on that class
          class_logit_true_mean: dict[label] -> mean true-class logit
        """
        model.eval()
        loader = self._build_local_loader(batch_size=batch_size)
        class_loss_sum = defaultdict(float)
        class_logit_sum = defaultdict(float)
        class_count = defaultdict(int)
        ce = nn.CrossEntropyLoss(reduction="none")

        with torch.no_grad():
            for b_idx, (images, labels) in enumerate(loader):
                if b_idx >= max_batches:
                    break
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                losses = ce(logits, labels)
                for i in range(labels.size(0)):
                    y = int(labels[i].item())
                    class_loss_sum[y] += float(losses[i].item())
                    class_logit_sum[y] += float(logits[i, y].item())
                    class_count[y] += 1

        class_loss_mean = {}
        class_logit_true_mean = {}
        for y, cnt in class_count.items():
            class_loss_mean[y] = class_loss_sum[y] / float(max(1, cnt))
            class_logit_true_mean[y] = class_logit_sum[y] / float(max(1, cnt))
        return class_loss_mean, class_logit_true_mean

def split_emnist_labels_per_client(dataset, num_clients: int, labels_per_client: int, verbose: bool = True):
    data_by_class = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        lab = int(label)
        if lab in allowed_labels:
            data_by_class[lab].append(idx)

    all_labels = list(range(MAX_LABELS))
    random.shuffle(all_labels)

    client_label_sets = []
    for cid in range(num_clients):
        start = labels_per_client * cid
        labs = [all_labels[(start + off) % MAX_LABELS] for off in range(labels_per_client)]
        client_label_sets.append(labs)

    clients = []
    for client_id, labs in enumerate(client_label_sets):
        indices = []
        for lab in labs:
            indices.extend(data_by_class[lab])
        random.shuffle(indices)
        client_data, client_labels = [], []
        for idx in indices:
            image, label = dataset[idx]
            client_data.append(image)
            client_labels.append(int(label))
        clients.append(AsyncFederatedClient(client_id, client_data, client_labels, verbose=verbose))
    return clients, client_label_sets


def evaluate_global_model(model, test_loader):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
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
    return correct / total, total_loss / len(test_loader)


def evaluate_model_on_client_data(model, client, batch_size: int = 256):
    model.eval()
    dataset = torch.utils.data.TensorDataset(
        torch.stack(client.local_data),
        torch.tensor(client.local_labels),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0.0


class FedFaServer:
    def __init__(self, global_model, buffer_size: int):
        self.global_model = global_model
        self.global_params = {n: p.data.clone().cpu() for n, p in global_model.named_parameters()}
        self.global_version = 0
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def apply_update(self, client_params, client_id: int):
        self.buffer.append({"params": {n: p.clone() for n, p in client_params.items()}, "client_id": client_id})
        if len(self.buffer) < self.buffer_size:
            return False
        new_global = {}
        K = len(self.buffer)
        for name in self.global_params.keys():
            acc = None
            for e in self.buffer:
                acc = e["params"][name].clone() if acc is None else acc + e["params"][name]
            new_global[name] = acc / float(K)
        self.global_params = {n: p.clone() for n, p in new_global.items()}
        self.global_version += 1
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.data.copy_(self.global_params[name].to(DEVICE))
        return True


class AFSAFLSemiAsyncServer:
    """
    AFSA-FL from provided paper:
      - Dynamic client-selection ratio C in [C_min, C_max]
      - Async rounds r = ceil(C * AFR), each waits for ceil(C*M) updates
      - Then one synchronous correction round
      - Activity-based aggregation weights: (n_i / n) * A_i
    """

    def __init__(
        self,
        global_model,
        num_clients: int,
        client_sizes: dict,
        c_min: float = 0.4,
        c_max: float = 0.8,
        afr: int = 10,
        activity_lambda: float = 0.9,
    ):
        self.global_model = global_model
        self.global_params = {n: p.data.clone().cpu() for n, p in global_model.named_parameters()}
        self.global_version = 0
        self.num_clients = num_clients
        self.client_sizes = client_sizes
        self.total_size = float(sum(client_sizes.values()))
        self.c_min = c_min
        self.c_max = c_max
        self.afr = afr
        self.activity_lambda = activity_lambda

        self.pending = []
        self.activity = {cid: 0.0 for cid in range(num_clients)}
        self.last_participation = {cid: 0 for cid in range(num_clients)}
        self.round_idx = 0
        self.async_rounds_done = 0

        self._reset_cycle()

    def _reset_cycle(self):
        self.C = random.uniform(self.c_min, self.c_max)
        self.r_async = max(1, int(math.ceil(self.C * self.afr)))
        self.quorum = max(1, int(math.ceil(self.C * self.num_clients)))
        self.async_rounds_done = 0

    def _update_activity(self, participated_clients):
        self.round_idx += 1
        for cid in range(self.num_clients):
            x = 1.0 if cid in participated_clients else 0.0
            self.activity[cid] = self.activity_lambda * self.activity[cid] + (1.0 - self.activity_lambda) * x
            if x > 0:
                self.last_participation[cid] = self.round_idx

    def _weighted_aggregate(self, entries):
        if not entries:
            return False
        raw_weights = []
        for e in entries:
            cid = e["client_id"]
            data_w = self.client_sizes[cid] / self.total_size
            act_w = max(self.activity.get(cid, 0.0), 1e-6)
            raw_weights.append(max(data_w * act_w, 1e-12))
        Z = sum(raw_weights)
        weights = [w / Z for w in raw_weights]

        new_global = {}
        for name in self.global_params.keys():
            acc = None
            for e, w in zip(entries, weights):
                weighted = w * e["params"][name]
                acc = weighted.clone() if acc is None else acc + weighted
            new_global[name] = acc

        self.global_params = {n: p.clone() for n, p in new_global.items()}
        self.global_version += 1
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.data.copy_(self.global_params[name].to(DEVICE))
        return True

    def submit_update(self, client_params, client_id: int):
        self.pending.append({"params": {n: p.clone() for n, p in client_params.items()}, "client_id": client_id})

    def maybe_async_step(self):
        if len(self.pending) < self.quorum:
            return False
        entries = self.pending[: self.quorum]
        self.pending = self.pending[self.quorum :]
        participated = [e["client_id"] for e in entries]
        self._update_activity(participated)
        did = self._weighted_aggregate(entries)
        self.async_rounds_done += 1
        return did

    def maybe_sync_correction(self):
        if self.async_rounds_done < self.r_async:
            return False
        # One synchronous correction: aggregate all pending available updates.
        entries = list(self.pending)
        self.pending = []
        if not entries:
            self._reset_cycle()
            return False
        participated = [e["client_id"] for e in entries]
        self._update_activity(participated)
        did = self._weighted_aggregate(entries)
        self._reset_cycle()
        return did


class FedFaServerParticipationFair:
    def __init__(self, global_model, buffer_size: int):
        self.global_model = global_model
        self.global_params = {n: p.data.clone().cpu() for n, p in global_model.named_parameters()}
        self.global_version = 0
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.participation_counts = defaultdict(int)
        self.eps = 1e-8

    def apply_update(self, client_params, client_id: int):
        self.participation_counts[client_id] += 1
        entry = {"params": {n: p.clone() for n, p in client_params.items()}, "client_id": client_id}
        existing_idx = None
        for i, e in enumerate(self.buffer):
            if e["client_id"] == client_id:
                existing_idx = i
                break
        if existing_idx is not None:
            self.buffer[existing_idx] = entry
        else:
            self.buffer.append(entry)

        if len(self.buffer) < self.buffer_size:
            return False

        raw_weights = []
        for e in self.buffer:
            cid = e["client_id"]
            raw_weights.append(1.0 / math.sqrt(float(self.participation_counts[cid]) + self.eps))
        z = sum(raw_weights)
        weights = [w / z for w in raw_weights]

        new_global = {}
        for name in self.global_params.keys():
            acc = None
            for e, w in zip(self.buffer, weights):
                weighted = w * e["params"][name]
                acc = weighted.clone() if acc is None else acc + weighted
            new_global[name] = acc

        self.global_params = {n: p.clone() for n, p in new_global.items()}
        self.global_version += 1
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.data.copy_(self.global_params[name].to(DEVICE))
        return True


class FedFaServerLabelAware:
    def __init__(self, global_model, buffer_size: int, client_label_scores: dict):
        self.global_model = global_model
        self.global_params = {n: p.data.clone().cpu() for n, p in global_model.named_parameters()}
        self.global_version = 0
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.client_label_scores = client_label_scores

    def apply_update(self, client_params, client_id: int):
        entry = {"params": {n: p.clone() for n, p in client_params.items()}, "client_id": client_id}
        existing_idx = None
        for i, e in enumerate(self.buffer):
            if e["client_id"] == client_id:
                existing_idx = i
                break
        if existing_idx is not None:
            self.buffer[existing_idx] = entry
        else:
            self.buffer.append(entry)

        if len(self.buffer) < self.buffer_size:
            return False

        raw_weights = []
        for e in self.buffer:
            cid = e["client_id"]
            raw_weights.append(max(self.client_label_scores.get(cid, 1.0), 1e-8))
        z = sum(raw_weights)
        weights = [w / z for w in raw_weights]

        new_global = {}
        for name in self.global_params.keys():
            acc = None
            for e, w in zip(self.buffer, weights):
                weighted = w * e["params"][name]
                acc = weighted.clone() if acc is None else acc + weighted
            new_global[name] = acc

        self.global_params = {n: p.clone() for n, p in new_global.items()}
        self.global_version += 1
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.data.copy_(self.global_params[name].to(DEVICE))
        return True


class FedFaServerRIRO:
    """
    RIRO-FedFa: Rare-In, Rare-Out Buffer Management.
    Keeps a semantic-aware buffer via selective eviction:
      E = (Age + beta) / Rarity
    and evicts the highest-E entry when full.
    """

    def __init__(self, global_model, buffer_size: int, client_label_scores: dict, beta: float = 1.0):
        self.global_model = global_model
        self.global_params = {n: p.data.clone().cpu() for n, p in global_model.named_parameters()}
        self.global_version = 0
        self.buffer = []  # list for selective eviction
        self.buffer_size = buffer_size
        self.client_label_scores = client_label_scores
        self.beta = beta

    def apply_update(self, client_params, client_id: int):
        # 1) Search-and-replace to keep only latest update per client in buffer.
        for i, entry in enumerate(self.buffer):
            if entry["client_id"] == client_id:
                self.buffer.pop(i)
                break

        # 2) Selective eviction if full.
        if len(self.buffer) >= self.buffer_size:
            eviction_scores = []
            for entry in self.buffer:
                age = self.global_version - entry["arrival_version"]
                rarity = entry["rarity_score"]
                e_score = (age + self.beta) / rarity
                eviction_scores.append(e_score)

            # Victim = highest expiration score.
            victim_idx = max(range(len(eviction_scores)), key=lambda i: eviction_scores[i])
            self.buffer.pop(victim_idx)

        # 3) Insert new update.
        new_entry = {
            "params": {n: p.clone() for n, p in client_params.items()},
            "client_id": client_id,
            "arrival_version": self.global_version,
            "rarity_score": max(self.client_label_scores.get(client_id, 1.0), 1e-8),
        }
        self.buffer.append(new_entry)

        # 4) Aggregate only when buffer full.
        if len(self.buffer) < self.buffer_size:
            return False

        new_global = {}
        K = len(self.buffer)
        for name in self.global_params.keys():
            acc = None
            for e in self.buffer:
                acc = e["params"][name].clone() if acc is None else acc + e["params"][name]
            new_global[name] = acc / float(K)

        self.global_params = {n: p.clone() for n, p in new_global.items()}
        self.global_version += 1
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.data.copy_(self.global_params[name].to(DEVICE))
        return True


print("\n🎯 Setting up EMNIST comparison: FedFa vs AFSA-FL vs PF-FedFa vs Label-Aware FedFa vs RIRO-FedFa")
print("=" * 80)

num_clients = 50
num_events = 2000
fedfa_buffer_size = 6
eval_every = 20

shared_local_lr = 1e-3
shared_local_epochs = 3
shared_batch_size = 64
labels_per_client = 2

# Participation imbalance for RIRO-stress setup.
enable_biased_sampling = True
heavy_client_fraction = 0.2  # top 20% clients
heavy_total_prob = 0.5       # account for ~50% sampling probability

# AFSA-FL specific (from paper text)
afsa_c_min = 0.4
afsa_c_max = 0.8
afsa_afr = 10
afsa_crash_prob = 0.3
afsa_activity_lambda = 0.9

clients, client_label_sets = split_emnist_labels_per_client(
    train_ds_full, num_clients, labels_per_client=labels_per_client
)
client_sizes = {c.client_id: c.data_size for c in clients}

if enable_biased_sampling:
    heavy_count = max(1, int(round(heavy_client_fraction * num_clients)))
    light_count = max(1, num_clients - heavy_count)
    w_heavy = heavy_total_prob / float(heavy_count)
    w_light = (1.0 - heavy_total_prob) / float(light_count)
    client_sampling_weights = [
        w_heavy if c.client_id < heavy_count else w_light for c in clients
    ]
else:
    client_sampling_weights = None


def sample_client():
    if client_sampling_weights is None:
        return random.choice(clients)
    return random.choices(clients, weights=client_sampling_weights, k=1)[0]

# Label-aware metadata
global_label_counts = defaultdict(int)
for c in clients:
    for y in c.local_labels:
        global_label_counts[int(y)] += 1
total_train_samples = float(sum(global_label_counts.values()))
global_label_freq = {lab: cnt / total_train_samples for lab, cnt in global_label_counts.items()}
label_rarity = {lab: 1.0 / (freq + 1e-12) for lab, freq in global_label_freq.items()}

client_label_scores = {}
for c in clients:
    counts = defaultdict(int)
    for y in c.local_labels:
        counts[int(y)] += 1
    total_c = float(len(c.local_labels))
    score = 0.0
    for lab, cnt in counts.items():
        score += (cnt / total_c) * label_rarity[lab]
    client_label_scores[c.client_id] = score

all_used_labels = sorted({lab for labs in client_label_sets for lab in labs})
filtered_test_indices = []
for idx in range(len(test_ds_full)):
    _, label = test_ds_full[idx]
    if int(label) in all_used_labels:
        filtered_test_indices.append(idx)
test_subset = torch.utils.data.Subset(test_ds_full, filtered_test_indices)
test_loader = DataLoader(test_subset, batch_size=512, shuffle=False)
print(f"Test size filtered to used labels: {len(filtered_test_indices)}")
print(f"Config: clients={num_clients}, labels_per_client={labels_per_client}, events={num_events}, buffer={fedfa_buffer_size}")
if enable_biased_sampling:
    print(
        f"Biased sampling enabled: top {int(heavy_client_fraction*100)}% clients "
        f"carry ~{int(heavy_total_prob*100)}% selection probability."
    )


# ---------- FedFa ----------
global_model_fedfa = MLPModel(num_classes=num_classes).to(DEVICE)
server_fedfa = FedFaServer(global_model_fedfa, buffer_size=fedfa_buffer_size)
for c in clients:
    c.pull_from_server(server_fedfa.global_params, server_fedfa.global_version)

fedfa_events, fedfa_accs, fedfa_losses = [], [], []

print("\n🚀 Running FedFa...")
for event in range(1, num_events + 1):
    c = sample_client()
    upd, used_v = c.train_async(
        global_model_fedfa,
        epochs=shared_local_epochs,
        lr=shared_local_lr,
        batch_size=shared_batch_size,
    )
    stale = server_fedfa.global_version - used_v
    server_fedfa.apply_update(upd, c.client_id)
    c.pull_from_server(server_fedfa.global_params, server_fedfa.global_version)
    if event % eval_every == 0 or event == 1:
        acc, loss = evaluate_global_model(global_model_fedfa, test_loader)
        fedfa_events.append(event)
        fedfa_accs.append(acc)
        fedfa_losses.append(loss)
        print(f"[FedFa] event={event}, client={c.client_id}, stale={stale}, acc={acc:.4f}, loss={loss:.4f}")


# ---------- AFSA-FL ----------
global_model_afsa = MLPModel(num_classes=num_classes).to(DEVICE)
server_afsa = AFSAFLSemiAsyncServer(
    global_model_afsa,
    num_clients=num_clients,
    client_sizes=client_sizes,
    c_min=afsa_c_min,
    c_max=afsa_c_max,
    afr=afsa_afr,
    activity_lambda=afsa_activity_lambda,
)
for c in clients:
    c.pull_from_server(server_afsa.global_params, server_afsa.global_version)

afsa_events, afsa_accs, afsa_losses = [], [], []

print("\n🚀 Running AFSA-FL...")
for event in range(1, num_events + 1):
    c = sample_client()
    upd, used_v = c.train_async(
        global_model_afsa,
        epochs=shared_local_epochs,
        lr=shared_local_lr,
        batch_size=shared_batch_size,
    )

    # AFSA paper models unreliable clients; we simulate crashes.
    crashed = random.random() < afsa_crash_prob
    if not crashed:
        server_afsa.submit_update(upd, c.client_id)
        server_afsa.maybe_async_step()
        server_afsa.maybe_sync_correction()

    # Semi-async distribution: active/crashed clients refresh to latest.
    c.pull_from_server(server_afsa.global_params, server_afsa.global_version)

    if event % eval_every == 0 or event == 1:
        acc, loss = evaluate_global_model(global_model_afsa, test_loader)
        afsa_events.append(event)
        afsa_accs.append(acc)
        afsa_losses.append(loss)
        print(
            f"[AFSA] event={event}, client={c.client_id}, crashed={crashed}, "
            f"pending={len(server_afsa.pending)}, C={server_afsa.C:.3f}, r={server_afsa.r_async}, "
            f"acc={acc:.4f}, loss={loss:.4f}"
        )


# ---------- Participation-Fair FedFa ----------
global_model_pf = MLPModel(num_classes=num_classes).to(DEVICE)
server_pf = FedFaServerParticipationFair(global_model_pf, buffer_size=fedfa_buffer_size)
for c in clients:
    c.pull_from_server(server_pf.global_params, server_pf.global_version)

pf_events, pf_accs, pf_losses = [], [], []

print("\n🚀 Running Participation-Fair FedFa...")
for event in range(1, num_events + 1):
    c = sample_client()
    upd, used_v = c.train_async(
        global_model_pf,
        epochs=shared_local_epochs,
        lr=shared_local_lr,
        batch_size=shared_batch_size,
    )
    stale = server_pf.global_version - used_v
    server_pf.apply_update(upd, c.client_id)
    c.pull_from_server(server_pf.global_params, server_pf.global_version)
    if event % eval_every == 0 or event == 1:
        acc, loss = evaluate_global_model(global_model_pf, test_loader)
        pf_events.append(event)
        pf_accs.append(acc)
        pf_losses.append(loss)
        print(f"[PF-FedFa] event={event}, client={c.client_id}, stale={stale}, acc={acc:.4f}, loss={loss:.4f}")


# ---------- Label-Aware FedFa ----------
global_model_la = MLPModel(num_classes=num_classes).to(DEVICE)
server_la = FedFaServerLabelAware(
    global_model_la,
    buffer_size=fedfa_buffer_size,
    client_label_scores=client_label_scores,
)
for c in clients:
    c.pull_from_server(server_la.global_params, server_la.global_version)

la_events, la_accs, la_losses = [], [], []

print("\n🚀 Running Label-Aware FedFa...")
for event in range(1, num_events + 1):
    c = sample_client()
    upd, used_v = c.train_async(
        global_model_la,
        epochs=shared_local_epochs,
        lr=shared_local_lr,
        batch_size=shared_batch_size,
    )
    stale = server_la.global_version - used_v
    server_la.apply_update(upd, c.client_id)
    c.pull_from_server(server_la.global_params, server_la.global_version)
    if event % eval_every == 0 or event == 1:
        acc, loss = evaluate_global_model(global_model_la, test_loader)
        la_events.append(event)
        la_accs.append(acc)
        la_losses.append(loss)
        print(f"[LA-FedFa] event={event}, client={c.client_id}, stale={stale}, acc={acc:.4f}, loss={loss:.4f}")


# ---------- RIRO-FedFa ----------
global_model_riro = MLPModel(num_classes=num_classes).to(DEVICE)
server_riro = FedFaServerRIRO(
    global_model_riro,
    buffer_size=fedfa_buffer_size,
    client_label_scores=client_label_scores,
    beta=1.0,
)
for c in clients:
    c.pull_from_server(server_riro.global_params, server_riro.global_version)

riro_events, riro_accs, riro_losses = [], [], []

print("\n🚀 Running RIRO-FedFa...")
for event in range(1, num_events + 1):
    c = sample_client()
    upd, used_v = c.train_async(
        global_model_riro,
        epochs=shared_local_epochs,
        lr=shared_local_lr,
        batch_size=shared_batch_size,
    )
    stale = server_riro.global_version - used_v
    server_riro.apply_update(upd, c.client_id)
    c.pull_from_server(server_riro.global_params, server_riro.global_version)
    if event % eval_every == 0 or event == 1:
        acc, loss = evaluate_global_model(global_model_riro, test_loader)
        riro_events.append(event)
        riro_accs.append(acc)
        riro_losses.append(loss)
        print(f"[RIRO-FedFa] event={event}, client={c.client_id}, stale={stale}, acc={acc:.4f}, loss={loss:.4f}")


print("\n📊 Plotting comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(fedfa_events, fedfa_accs, "b-x", label="FedFa")
ax1.plot(afsa_events, afsa_accs, "m-s", label="AFSA-FL")
ax1.plot(pf_events, pf_accs, "y-*", label="PF-FedFa")
ax1.plot(la_events, la_accs, "g-^", label="LA-FedFa")
ax1.plot(riro_events, riro_accs, "c-o", label="RIRO-FedFa")
ax1.set_xlabel("Event")
ax1.set_ylabel("Test Accuracy")
ax1.set_title("EMNIST: Accuracy vs Events", fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim([0, 1])

ax2.plot(fedfa_events, fedfa_losses, "b-x", label="FedFa")
ax2.plot(afsa_events, afsa_losses, "m-s", label="AFSA-FL")
ax2.plot(pf_events, pf_losses, "y-*", label="PF-FedFa")
ax2.plot(la_events, la_losses, "g-^", label="LA-FedFa")
ax2.plot(riro_events, riro_losses, "c-o", label="RIRO-FedFa")
ax2.set_xlabel("Event")
ax2.set_ylabel("Test Loss")
ax2.set_title("EMNIST: Loss vs Events", fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("emnist_fedfa_afsa_pf_la_riro_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


def print_final_best(name, accs):
    if not accs:
        print(f"{name:20s} | no evaluation points")
        return
    final_acc = accs[-1]
    best_acc = max(accs)
    avg_acc = sum(accs) / float(len(accs))
    print(
        f"{name:20s} | Final: {final_acc:.4f} ({final_acc*100:.2f}%)"
        f" | Best: {best_acc:.4f} ({best_acc*100:.2f}%)"
        f" | Avg: {avg_acc:.4f} ({avg_acc*100:.2f}%)"
    )


print("\n" + "=" * 80)
print("FINAL SUMMARY: FedFa vs AFSA-FL vs PF-FedFa vs LA-FedFa vs RIRO-FedFa")
print("=" * 80)
print_final_best("FedFa", fedfa_accs)
print_final_best("AFSA-FL", afsa_accs)
print_final_best("PF-FedFa", pf_accs)
print_final_best("LA-FedFa", la_accs)
print_final_best("RIRO-FedFa", riro_accs)

print("\n📊 Per-client accuracy on final global models:")
for c in clients:
    a1 = evaluate_model_on_client_data(global_model_fedfa, c)
    a2 = evaluate_model_on_client_data(global_model_afsa, c)
    a3 = evaluate_model_on_client_data(global_model_pf, c)
    a4 = evaluate_model_on_client_data(global_model_la, c)
    a5 = evaluate_model_on_client_data(global_model_riro, c)
    print(
        f"   Client {c.client_id}: "
        f"FedFa={a1:.4f} ({a1*100:.2f}%), "
        f"AFSA-FL={a2:.4f} ({a2*100:.2f}%), "
        f"PF-FedFa={a3:.4f} ({a3*100:.2f}%), "
        f"LA-FedFa={a4:.4f} ({a4*100:.2f}%), "
        f"RIRO-FedFa={a5:.4f} ({a5*100:.2f}%)"
    )



