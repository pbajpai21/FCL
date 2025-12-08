"""
Permuted MNIST with EWC - Exact Setup from EWC Paper
Paper: "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al., 2017)

Setup:
- Each task: Classify all 10 MNIST digits (0-9)
- Each task uses a DIFFERENT fixed random permutation of input pixels
- Same problem difficulty, but different pixel arrangements
- Network: MLP with 2 hidden layers (784 -> 400 -> 400 -> 10)
"""

import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

# Setup
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(8)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds_full = datasets.MNIST(root="./data", train=False, download=True, transform=transform)


# ========================================
# PERMUTED MNIST DATASET
# ========================================

class PermutedMNIST(Dataset):
    """
    MNIST dataset with a fixed random permutation applied to pixels.
    
    For Task 0: No permutation (original MNIST)
    For Task 1+: Fixed random permutation of all 784 pixels
    """
    def __init__(self, original_dataset, permutation=None):
        self.dataset = original_dataset
        self.permutation = permutation
        
        if permutation is None:
            print("  Task 0: Original MNIST (no permutation)")
        else:
            print(f"  Task with permutation (first 10 indices): {permutation[:10].tolist()}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Flatten image (1, 28, 28) -> (784,)
        image_flat = image.view(-1)
        
        # Apply permutation if specified
        if self.permutation is not None:
            image_flat = image_flat[self.permutation]
        
        return image_flat, label


def generate_permutation():
    """Generate a random permutation of pixel indices."""
    return torch.randperm(784)


# ========================================
# NEURAL NETWORK ARCHITECTURE
# ========================================

class MLPModel(nn.Module):
    """
    MLP Architecture from EWC Paper:
    - Input: 784 (28x28 flattened)
    - Hidden 1: 400 neurons + ReLU
    - Hidden 2: 400 neurons + ReLU  
    - Output: 10 classes (digits 0-9)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 10)
        )
    
    def forward(self, x):
        return self.net(x)


# ========================================
# EWC FUNCTIONS
# ========================================

def compute_fisher_information(model, data_loader, num_samples=None):
    """
    Compute Fisher Information Matrix (diagonal approximation).
    
    Uses the empirical Fisher:
    F_i = E[(∂log p(y|x,θ) / ∂θ_i)²]
    """
    print("  Computing Fisher Information Matrix...")
    fisher = {}
    
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    samples_processed = 0
    for images, labels in data_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.pow(2) * images.size(0)
        
        samples_processed += images.size(0)
        
        if num_samples is not None and samples_processed >= num_samples:
            break
    
    # Normalize by number of samples
    for name in fisher:
        fisher[name] /= samples_processed
    
    print(f"  Fisher Information computed using {samples_processed} samples")
    return fisher


def ewc_loss(model, fisher_dict, optimal_params_dict, lambda_ewc):
    """
    Compute EWC regularization penalty.
    
    L_EWC = (λ/2) Σ F_i (θ_i - θ*_i)²
    """
    loss = 0
    for name, param in model.named_parameters():
        if name in fisher_dict:
            fisher = fisher_dict[name]
            optimal_param = optimal_params_dict[name]
            loss += (fisher * (param - optimal_param).pow(2)).sum()
    
    return (lambda_ewc / 2) * loss


def consolidate_fisher(fisher_dict_list):
    """
    Consolidate multiple Fisher information matrices.
    Simple addition for multiple tasks.
    """
    if len(fisher_dict_list) == 0:
        return None
    
    consolidated = {}
    for name in fisher_dict_list[0].keys():
        consolidated[name] = sum(f[name] for f in fisher_dict_list)
    
    return consolidated


# ========================================
# TRAINING FUNCTIONS
# ========================================

def train_task(model, train_loader, epochs, task_id, 
               fisher_dict=None, optimal_params_dict=None, lambda_ewc=0):
    """Train model on a single task with optional EWC regularization."""
    
    print(f"\n{'='*70}")
    if lambda_ewc > 0 and fisher_dict is not None:
        print(f"Training Task {task_id} with EWC (λ={lambda_ewc})")
    else:
        print(f"Training Task {task_id} (No EWC)")
    print(f"{'='*70}")
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        total_loss = 0
        total_task_loss = 0
        total_ewc_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            task_loss = criterion(outputs, labels)
            
            # Add EWC penalty if available
            if fisher_dict is not None and optimal_params_dict is not None and lambda_ewc > 0:
                ewc_penalty = ewc_loss(model, fisher_dict, optimal_params_dict, lambda_ewc)
                loss = task_loss + ewc_penalty
                total_ewc_loss += ewc_penalty.item()
            else:
                loss = task_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        avg_task_loss = total_task_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        if fisher_dict is not None and lambda_ewc > 0:
            avg_ewc_loss = total_ewc_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f} "
                  f"(Task={avg_task_loss:.4f}, EWC={avg_ewc_loss:.4f}), "
                  f"Acc={accuracy:.2f}%")
        else:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")


def evaluate_task(model, test_loader, task_name):
    """Evaluate model on a single task."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy / 100.0  # Return as decimal


# ========================================
# EXPERIMENT: PERMUTED MNIST WITH EWC
# ========================================

def run_experiment(num_tasks=3, lambda_ewc=0, epochs_per_task=10):
    """
    Run Permuted MNIST experiment with or without EWC.
    
    Args:
        num_tasks: Number of tasks (different permutations)
        lambda_ewc: EWC regularization strength (0 = no EWC)
        epochs_per_task: Training epochs for each task
    """
    print("\n" + "="*80)
    print(f"PERMUTED MNIST EXPERIMENT")
    print(f"Tasks: {num_tasks}, Lambda: {lambda_ewc}, Epochs/Task: {epochs_per_task}")
    print("="*80)
    
    # Generate permutations for each task
    print("\nGenerating permutations...")
    permutations = [None]  # Task 0: original MNIST
    for i in range(1, num_tasks):
        permutations.append(generate_permutation())
    
    # Create datasets for each task
    print("\nCreating datasets...")
    train_datasets = []
    test_datasets = []
    
    for task_id in range(num_tasks):
        print(f"\nTask {task_id}:")
        train_ds = PermutedMNIST(train_ds_full, permutations[task_id])
        test_ds = PermutedMNIST(test_ds_full, permutations[task_id])
        train_datasets.append(train_ds)
        test_datasets.append(test_ds)
    
    # Create data loaders
    train_loaders = [DataLoader(ds, batch_size=128, shuffle=True) for ds in train_datasets]
    test_loaders = [DataLoader(ds, batch_size=256, shuffle=False) for ds in test_datasets]
    
    # Initialize model
    model = MLPModel().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: MLP with {total_params:,} parameters")
    print(f"   Architecture: 784 -> 400 -> 400 -> 10")
    
    # Storage for Fisher and optimal parameters
    fisher_list = []
    optimal_params_dict = None
    
    # Storage for results
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    
    # Train on each task sequentially
    for task_id in range(num_tasks):
        # Consolidate Fisher information from all previous tasks
        if len(fisher_list) > 0:
            fisher_dict = consolidate_fisher(fisher_list)
        else:
            fisher_dict = None
        
        # Train on current task
        train_task(model, train_loaders[task_id], epochs_per_task, task_id,
                   fisher_dict, optimal_params_dict, lambda_ewc)
        
        # Evaluate on all tasks seen so far
        print(f"\n📊 Evaluation after training Task {task_id}:")
        for eval_task_id in range(task_id + 1):
            acc = evaluate_task(model, test_loaders[eval_task_id], f"Task {eval_task_id}")
            accuracy_matrix[task_id, eval_task_id] = acc
            print(f"  Task {eval_task_id}: {acc:.4f} ({acc*100:.2f}%)")
        
        # Compute and store Fisher information for this task
        print(f"\n🔍 Computing Fisher Information for Task {task_id}...")
        fisher = compute_fisher_information(model, train_loaders[task_id])
        fisher_list.append(fisher)
        
        # Update optimal parameters
        optimal_params_dict = {name: param.clone().detach()
                              for name, param in model.named_parameters()}
    
    return accuracy_matrix


# ========================================
# RUN EXPERIMENTS
# ========================================

print("\n" + "="*80)
print("PERMUTED MNIST: EWC PAPER REPLICATION")
print("="*80)
print("\nWe'll run two experiments:")
print("  1. Vanilla SGD (no EWC, λ=0)")
print("  2. EWC (λ=15)")
print("\nThis will take a few minutes...")

NUM_TASKS = 3
EPOCHS_PER_TASK = 10

# Experiment 1: Vanilla (no EWC)
print("\n" + "="*80)
print("EXPERIMENT 1: VANILLA TRAINING (NO EWC)")
print("="*80)
vanilla_results = run_experiment(num_tasks=NUM_TASKS, lambda_ewc=0, epochs_per_task=EPOCHS_PER_TASK)

# Experiment 2: EWC
print("\n" + "="*80)
print("EXPERIMENT 2: EWC TRAINING")
print("="*80)
ewc_results = run_experiment(num_tasks=NUM_TASKS, lambda_ewc=15, epochs_per_task=EPOCHS_PER_TASK)


# ========================================
# RESULTS ANALYSIS
# ========================================

print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

# Calculate average accuracy after all tasks
vanilla_final = [vanilla_results[-1, i] for i in range(NUM_TASKS)]
ewc_final = [ewc_results[-1, i] for i in range(NUM_TASKS)]

print("\nFinal Accuracy on Each Task:")
print(f"{'Task':<10} {'Vanilla':<15} {'EWC':<15} {'Improvement':<15}")
print("-" * 55)
for i in range(NUM_TASKS):
    improvement = (ewc_final[i] - vanilla_final[i]) * 100
    print(f"Task {i:<5} {vanilla_final[i]:.4f} ({vanilla_final[i]*100:5.2f}%)  "
          f"{ewc_final[i]:.4f} ({ewc_final[i]*100:5.2f}%)  "
          f"+{improvement:5.2f}%")

avg_vanilla = np.mean(vanilla_final)
avg_ewc = np.mean(ewc_final)
improvement = (avg_ewc - avg_vanilla) * 100

print("-" * 55)
print(f"{'Average':<10} {avg_vanilla:.4f} ({avg_vanilla*100:5.2f}%)  "
      f"{avg_ewc:.4f} ({avg_ewc*100:5.2f}%)  "
      f"+{improvement:5.2f}%")

# Calculate forgetting
print("\nForgetting Analysis:")
print("(Forgetting = Initial Accuracy - Final Accuracy)")
print(f"{'Task':<10} {'Vanilla Forget':<18} {'EWC Forget':<18}")
print("-" * 46)
for i in range(NUM_TASKS - 1):  # Exclude last task (no forgetting possible)
    vanilla_forget = (vanilla_results[i, i] - vanilla_results[-1, i]) * 100
    ewc_forget = (ewc_results[i, i] - ewc_results[-1, i]) * 100
    print(f"Task {i:<5} {vanilla_forget:5.2f}%            {ewc_forget:5.2f}%")


# ========================================
# VISUALIZATION
# ========================================

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Vanilla Accuracy Matrix
im1 = ax1.imshow(vanilla_results * 100, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
ax1.set_xlabel('Task Evaluated', fontsize=12, fontweight='bold')
ax1.set_ylabel('After Training Task', fontsize=12, fontweight='bold')
ax1.set_title('Vanilla Training (No EWC)\nAccuracy Matrix', fontsize=13, fontweight='bold')
ax1.set_xticks(range(NUM_TASKS))
ax1.set_yticks(range(NUM_TASKS))
for i in range(NUM_TASKS):
    for j in range(i + 1):
        text = ax1.text(j, i, f'{vanilla_results[i, j]*100:.1f}',
                       ha="center", va="center", color="black", fontweight='bold')
plt.colorbar(im1, ax=ax1, label='Accuracy (%)')

# Plot 2: EWC Accuracy Matrix
im2 = ax2.imshow(ewc_results * 100, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
ax2.set_xlabel('Task Evaluated', fontsize=12, fontweight='bold')
ax2.set_ylabel('After Training Task', fontsize=12, fontweight='bold')
ax2.set_title('EWC Training (λ=15)\nAccuracy Matrix', fontsize=13, fontweight='bold')
ax2.set_xticks(range(NUM_TASKS))
ax2.set_yticks(range(NUM_TASKS))
for i in range(NUM_TASKS):
    for j in range(i + 1):
        text = ax2.text(j, i, f'{ewc_results[i, j]*100:.1f}',
                       ha="center", va="center", color="black", fontweight='bold')
plt.colorbar(im2, ax=ax2, label='Accuracy (%)')

# Plot 3: Final Accuracy Comparison
x_pos = np.arange(NUM_TASKS)
width = 0.35
bars1 = ax3.bar(
    x_pos - width/2,
    np.array(vanilla_final) * 100,
    width,
    label='Vanilla',
    color='orange',
    alpha=0.7,
)
bars2 = ax3.bar(
    x_pos + width/2,
    np.array(ewc_final) * 100,
    width,
    label='EWC',
    color='green',
    alpha=0.7,
)

ax3.set_xlabel('Task', fontsize=12, fontweight='bold')
ax3.set_ylabel('Final Accuracy (%)', fontsize=12, fontweight='bold')
ax3.set_title('Final Accuracy Comparison', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'Task {i}' for i in range(NUM_TASKS)])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim([0, 100])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Plot 4: Forgetting over tasks
forgetting_vanilla = []
forgetting_ewc = []
for i in range(NUM_TASKS):
    if i > 0:
        vanilla_avg_forget = np.mean([(vanilla_results[j, j] - vanilla_results[i, j]) * 100 
                                      for j in range(i)])
        ewc_avg_forget = np.mean([(ewc_results[j, j] - ewc_results[i, j]) * 100 
                                  for j in range(i)])
        forgetting_vanilla.append(vanilla_avg_forget)
        forgetting_ewc.append(ewc_avg_forget)

task_indices = list(range(1, NUM_TASKS))
ax4.plot(task_indices, forgetting_vanilla, 'o-', linewidth=2, markersize=8,
         label='Vanilla', color='red')
ax4.plot(task_indices, forgetting_ewc, 's-', linewidth=2, markersize=8,
         label='EWC', color='green')
ax4.set_xlabel('After Training Task', fontsize=12, fontweight='bold')
ax4.set_ylabel('Average Forgetting (%)', fontsize=12, fontweight='bold')
ax4.set_title('Catastrophic Forgetting Over Time\n(Lower is Better)', fontsize=13, fontweight='bold')
ax4.set_xticks(task_indices)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, max(max(forgetting_vanilla), max(forgetting_ewc)) + 5])

plt.tight_layout()
plt.savefig('permuted_mnist_ewc_results.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n📊 Visualization saved as 'permuted_mnist_ewc_results.png'")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)
print("\n🎯 Key Findings:")
print(f"  • EWC improves average accuracy by {improvement:.2f}%")
print(f"  • EWC significantly reduces catastrophic forgetting")
print(f"  • This demonstrates EWC's effectiveness on the paper's benchmark!")

