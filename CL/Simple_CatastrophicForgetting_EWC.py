"""
EWC Demonstration - Reducing Catastrophic Forgetting
Train on 3 tasks sequentially with Elastic Weight Consolidation (EWC):
Task 1: digits 1,2,3
Task 2: digits 4,5,6
Task 3: digits 7,8,9

EWC protects important weights from previous tasks while learning new tasks.
"""

import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy

# Setup
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

# MNIST Labels
MNIST_LABELS = {
    0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four',
    5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'
}

# Filter dataset by labels
class FilteredMNIST:
    def __init__(self, original_dataset, target_labels):
        self.data = []
        self.targets = []
        
        print(f"Creating dataset with labels {target_labels}...")
        for i in range(len(original_dataset)):
            image, label = original_dataset[i]
            if label in target_labels:
                self.data.append(image)
                # Remap labels: e.g., [1,2,3] → [0,1,2]
                new_label = target_labels.index(label)
                self.targets.append(new_label)
        
        print(f"  Found {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# Simple MLP Model
class MLPModel(nn.Module):
    def __init__(self, num_classes=3):
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
# EWC FUNCTIONS
# ========================================

def compute_fisher_information(model, data_loader):
    """
    Compute Fisher Information Matrix (diagonal approximation).
    
    Fisher Information tells us how important each parameter is for the current task.
    Higher Fisher value = more important parameter = needs more protection.
    """
    print("  Computing Fisher Information Matrix...")
    fisher = {}
    
    # Initialize Fisher dictionary with zeros
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Accumulate gradients squared (Fisher approximation)
    for images, labels in data_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.pow(2)
    
    # Average over dataset
    num_samples = len(data_loader)
    for name in fisher:
        fisher[name] /= num_samples
    
    print("  Fisher Information computed!")
    return fisher


def ewc_loss(model, fisher_dict, optimal_params_dict, lambda_ewc=1000):
    """
    Compute EWC penalty: penalizes deviation from old optimal parameters.
    
    EWC Loss = λ/2 * Σ F_i * (θ_i - θ*_i)²
    
    Where:
    - F_i: Fisher Information (importance) of parameter i
    - θ_i: Current parameter value
    - θ*_i: Old optimal parameter value (from previous task)
    - λ: Strength of the penalty
    """
    loss = 0
    for name, param in model.named_parameters():
        if name in fisher_dict:
            # Penalize changes to important parameters
            fisher = fisher_dict[name]
            optimal_param = optimal_params_dict[name]
            loss += (fisher * (param - optimal_param).pow(2)).sum()
    
    return (lambda_ewc / 2) * loss


# Training function with EWC
def train_with_ewc(model, train_loader, epochs, task_name, 
                   fisher_dict=None, optimal_params_dict=None, lambda_ewc=1000):
    """
    Train with EWC regularization to prevent catastrophic forgetting.
    
    Total Loss = Task Loss + EWC Loss
    """
    print(f"\n{'='*60}")
    print(f"Training on {task_name} with EWC (λ={lambda_ewc})")
    print(f"{'='*60}")
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        total_task_loss = 0
        total_ewc_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            task_loss = criterion(outputs, labels)
            
            # Add EWC penalty if we have previous task info
            if fisher_dict is not None and optimal_params_dict is not None:
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
        
        avg_loss = total_loss / len(train_loader)
        avg_task_loss = total_task_loss / len(train_loader)
        
        if fisher_dict is not None:
            avg_ewc_loss = total_ewc_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs}: Total Loss = {avg_loss:.4f} "
                  f"(Task: {avg_task_loss:.4f}, EWC: {avg_ewc_loss:.4f})")
        else:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")


# Evaluation function
def evaluate(model, test_loader, task_name):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"  {task_name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return accuracy


# ========================================
# EXPERIMENT: EWC Training
# ========================================

print("\n" + "="*80)
print("EWC DEMONSTRATION - REDUCING CATASTROPHIC FORGETTING")
print("="*80)
print("\nTask 1: Digits 1, 2, 3")
print("Task 2: Digits 4, 5, 6")
print("Task 3: Digits 7, 8, 9")
print("\nWe'll train sequentially with EWC to protect previous task knowledge.")

# Define tasks
task1_labels = [1, 2, 3]  # Task 1
task2_labels = [4, 5, 6]  # Task 2
task3_labels = [7, 8, 9]  # Task 3

# Create datasets
print("\n" + "="*60)
print("CREATING DATASETS")
print("="*60)

task1_train = FilteredMNIST(train_ds_full, task1_labels)
task2_train = FilteredMNIST(train_ds_full, task2_labels)
task3_train = FilteredMNIST(train_ds_full, task3_labels)

task1_test = FilteredMNIST(test_ds_full, task1_labels)
task2_test = FilteredMNIST(test_ds_full, task2_labels)
task3_test = FilteredMNIST(test_ds_full, task3_labels)

# Create data loaders
train_loader_task1 = DataLoader(task1_train, batch_size=32, shuffle=True)
train_loader_task2 = DataLoader(task2_train, batch_size=32, shuffle=True)
train_loader_task3 = DataLoader(task3_train, batch_size=32, shuffle=True)

test_loader_task1 = DataLoader(task1_test, batch_size=128, shuffle=False)
test_loader_task2 = DataLoader(task2_test, batch_size=128, shuffle=False)
test_loader_task3 = DataLoader(task3_test, batch_size=128, shuffle=False)

# Initialize model
model = MLPModel(num_classes=3).to(DEVICE)

# EWC hyperparameter
LAMBDA_EWC = 5000  # Strength of EWC regularization (higher = more protection)
EPOCHS = 5

# Storage for Fisher Information and optimal parameters
fisher_dict = None
optimal_params_dict = None

# ========================================
# TASK 1: Train on digits 1,2,3
# ========================================

print("\n" + "="*80)
print("TASK 1: Training on digits 1, 2, 3")
print("="*80)

train_with_ewc(model, train_loader_task1, EPOCHS, "Task 1")

print("\n📊 Evaluating after Task 1:")
task1_acc_after_task1 = evaluate(model, test_loader_task1, "Task 1")

# Save Fisher Information and optimal parameters for Task 1
print("\n🔍 Saving Task 1 knowledge (Fisher + optimal params)...")
fisher_dict = compute_fisher_information(model, train_loader_task1)
optimal_params_dict = {name: param.clone().detach() 
                       for name, param in model.named_parameters()}

# ========================================
# TASK 2: Train on digits 4,5,6 with EWC
# ========================================

print("\n" + "="*80)
print("TASK 2: Training on digits 4, 5, 6 (with EWC protection)")
print("="*80)

train_with_ewc(model, train_loader_task2, EPOCHS, "Task 2", 
               fisher_dict, optimal_params_dict, LAMBDA_EWC)

print("\n📊 Evaluating after Task 2:")
task1_acc_after_task2 = evaluate(model, test_loader_task1, "Task 1")
task2_acc_after_task2 = evaluate(model, test_loader_task2, "Task 2")

# Update Fisher and optimal params (combine Task 1 and Task 2 knowledge)
print("\n🔍 Updating Fisher Information with Task 2 knowledge...")
fisher_task2 = compute_fisher_information(model, train_loader_task2)

# Combine Fisher information from both tasks
for name in fisher_dict:
    fisher_dict[name] = fisher_dict[name] + fisher_task2[name]

# Update optimal parameters
optimal_params_dict = {name: param.clone().detach() 
                       for name, param in model.named_parameters()}

# ========================================
# TASK 3: Train on digits 7,8,9 with EWC
# ========================================

print("\n" + "="*80)
print("TASK 3: Training on digits 7, 8, 9 (with EWC protection)")
print("="*80)

train_with_ewc(model, train_loader_task3, EPOCHS, "Task 3", 
               fisher_dict, optimal_params_dict, LAMBDA_EWC)

print("\n📊 Evaluating after Task 3:")
task1_acc_after_task3 = evaluate(model, test_loader_task1, "Task 1")
task2_acc_after_task3 = evaluate(model, test_loader_task2, "Task 2")
task3_acc_after_task3 = evaluate(model, test_loader_task3, "Task 3")

# ========================================
# RESULTS SUMMARY
# ========================================

print("\n" + "="*80)
print("FINAL RESULTS WITH EWC")
print("="*80)

print(f"\nTask 1 (1,2,3):")
print(f"  After Task 1: {task1_acc_after_task1:.4f}")
print(f"  After Task 2: {task1_acc_after_task2:.4f}")
print(f"  After Task 3: {task1_acc_after_task3:.4f}")
task1_forgetting = (task1_acc_after_task1 - task1_acc_after_task3)
task1_forgetting_pct = task1_forgetting * 100
print(f"  Forgetting: {task1_forgetting:.4f} ({task1_forgetting_pct:.1f}%)")

print(f"\nTask 2 (4,5,6):")
print(f"  After Task 2: {task2_acc_after_task2:.4f}")
print(f"  After Task 3: {task2_acc_after_task3:.4f}")
task2_forgetting = (task2_acc_after_task2 - task2_acc_after_task3)
task2_forgetting_pct = task2_forgetting * 100
print(f"  Forgetting: {task2_forgetting:.4f} ({task2_forgetting_pct:.1f}%)")

print(f"\nTask 3 (7,8,9):")
print(f"  After Task 3: {task3_acc_after_task3:.4f}")

print(f"\n" + "="*80)
print("EWC SUCCESS!")
print("="*80)
print(f"EWC reduces catastrophic forgetting by protecting important weights.")
print(f"Compare these results with vanilla training to see the improvement!")

# ========================================
# VISUALIZATION
# ========================================

def plot_results():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram: Original vs Final performance comparison
    x_pos = [0, 1, 2]
    original_accs = [task1_acc_after_task1, task2_acc_after_task2, task3_acc_after_task3]
    final_accs = [task1_acc_after_task3, task2_acc_after_task3, task3_acc_after_task3]
    
    width = 0.35
    bars1 = ax.bar([p - width/2 for p in x_pos], original_accs, width, 
                    label='Original Accuracy', color='green', alpha=0.7)
    bars2 = ax.bar([p + width/2 for p in x_pos], final_accs, width, 
                    label='Final Accuracy (After Task 3)', color='blue', alpha=0.7)
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Task', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title(f'EWC: Original vs Final Accuracy (λ={LAMBDA_EWC})', fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Task 1\n(1,2,3)', 'Task 2\n(4,5,6)', 'Task 3\n(7,8,9)'], fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('simple_catastrophic_forgetting_ewc.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n📊 Visualization saved as 'simple_catastrophic_forgetting_ewc.png'")


plot_results()

print(f"\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)
print(f"🎯 EWC protected previous task knowledge!")
print(f"🎯 Compare with vanilla training to see the difference!")

