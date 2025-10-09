import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy

# Optimize PyTorch for CPU multithreading
torch.set_num_threads(8)  # Use 8 threads for PyTorch operations
print(f"PyTorch using {torch.get_num_threads()} threads")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ----- 1) Data Preparation -----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# Load full MNIST datasets
train_ds_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds_full = datasets.MNIST(root="./data", train=False, download=True, transform=transform)


print(f"Train dataset size: {len(train_ds_full)}")
print(f"Test dataset size: {len(test_ds_full)}")

# MNIST class labels (integers 0-9)
MNIST_LABELS = {
    0: 'Zero',
    1: 'One', 
    2: 'Two',
    3: 'Three',
    4: 'Four',
    5: 'Five',
    6: 'Six',
    7: 'Seven',
    8: 'Eight',
    9: 'Nine'
}

class FilteredMNIST:
    def __init__(self, original_dataset, target_labels):
        self.data = []
        self.targets = []
        self.target_labels = target_labels
        
        print(f"Creating dataset with labels {target_labels}...")
        print(f"Class names: {[MNIST_LABELS[label] for label in target_labels]}")
        
        # Filter for specific labels and remap to 0-based indexing  
        for i in range(len(original_dataset)):
            image, label = original_dataset[i]
            if label in target_labels:
                self.data.append(image)
                # Remap labels: e.g., [1,2,3,4,5] → [0,1,2,3,4]
                new_label = target_labels.index(label)
                self.targets.append(new_label)
        
        print(f"Dataset created: {len(self.data)} samples, labels remapped to {set(self.targets)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class MLPMNISTContinual(nn.Module):
    def __init__(self, num_classes=2):  # Each task has 2 classes
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),  # Output 2 classes per task
        )

    def forward(self, x):
        return self.net(x)


model = MLPMNISTContinual(num_classes=2).to(DEVICE)  # 2 classes per task

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ========================================
# 🧠 EWC (Elastic Weight Consolidation) Implementation
# ========================================

# EWC parameters
ewc_lambda = 1000  # Regularization strength
fisher_information = {}  # Store Fisher Information Matrix
optimal_params = {}  # Store optimal parameters after each task

def compute_fisher_information(model, data_loader, num_samples=1000):
    """
    Compute Fisher Information Matrix for EWC regularization.
    Fisher Information measures the importance of each parameter.
    """
    print("🔍 Computing Fisher Information Matrix...")
    
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param.data)
    
    model.eval()
    sample_count = 0
    
    for images, labels in data_loader:
        if sample_count >= num_samples:
            break
            
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass to get gradients
        model.zero_grad()
        loss.backward()
        
        # Accumulate squared gradients (Fisher Information approximation)
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data ** 2
        
        sample_count += images.size(0)
    
    # Average over samples
    for name in fisher:
        fisher[name] /= sample_count
    
    print(f"✅ Fisher Information computed using {sample_count} samples")
    return fisher

def ewc_loss(model, fisher_information, optimal_params, ewc_lambda):
    """
    Compute EWC regularization loss.
    Penalizes changes to important parameters (high Fisher Information).
    """
    ewc_loss_value = 0
    
    for name, param in model.named_parameters():
        if name in fisher_information and name in optimal_params:
            # EWC loss: λ/2 * Σ F_i * (θ_i - θ*_i)^2
            ewc_loss_value += (ewc_lambda / 2) * torch.sum(
                fisher_information[name] * (param - optimal_params[name]) ** 2
            )
    
    return ewc_loss_value

def train_task_ewc(model, train_loader, epochs=5, task_name="", task_num=1):
    """
    Train model with EWC regularization to prevent catastrophic forgetting.
    """
    import time
    model.train()
    print(f"🚀 Training on {task_name} with EWC regularization...")
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        total_ewc_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            task_loss = criterion(outputs, labels)
            
            # Add EWC regularization loss for previous tasks
            ewc_reg_loss = 0
            if task_num > 1:  # Only apply EWC after first task
                ewc_reg_loss = ewc_loss(model, fisher_information, optimal_params, ewc_lambda)
            
            total_loss = task_loss + ewc_reg_loss
            total_ewc_loss += ewc_reg_loss.item() if isinstance(ewc_reg_loss, torch.Tensor) else ewc_reg_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        epoch_time = time.time() - start_time
        avg_task_loss = (total_loss.item() - total_ewc_loss) / len(train_loader)
        avg_ewc_loss = total_ewc_loss / len(train_loader)
        
        print(f"  Epoch {epoch+1}: Task Loss = {avg_task_loss:.4f}, EWC Loss = {avg_ewc_loss:.4f}, Time = {epoch_time:.2f}s")


def evaluate_task(model, test_loader, task_name=""):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    accuracy = correct/total
    print(f"📊 {task_name} - Loss: {total_loss/total:.4f}, Accuracy: {accuracy:.4f}")
    return total_loss/total, accuracy

def print_network_weights(model, task_name):
    """Print network weights for each layer"""
    print(f"\n🔍 NETWORK WEIGHTS AFTER {task_name}")
    print("=" * 80)
    
    for name, param in model.named_parameters():
        print(f"\n{name}:")
        print(f"  Shape: {param.shape}")
        print(f"  Weights:")
        print(f"  {param.data}")
        print("-" * 40)

# ========================================
# 🧠 EWC CONTINUAL LEARNING EXPERIMENT
# ========================================

print(f"\n🎯 Setting up 5-Task Continual Learning Experiment with EWC")
print(f"We'll train on 5 sequential tasks with 2 classes each using EWC regularization")

# Define 5 tasks with 2 classes each
tasks = [
    [1, 2],  # Task 1: One, Two
    [3, 4],  # Task 2: Three, Four
    [5, 6],  # Task 3: Five, Six
    [7, 8],  # Task 4: Seven, Eight
    [9, 0]   # Task 5: Nine, Zero
]

print(f"\nTask definitions:")
for i, task_labels in enumerate(tasks, 1):
    print(f"Task {i}: {[MNIST_LABELS[label] for label in task_labels]}")

# Create filtered datasets for all tasks
task_datasets = {}
for i, task_labels in enumerate(tasks, 1):
    task_datasets[f'task{i}_train'] = FilteredMNIST(train_ds_full, task_labels)
    task_datasets[f'task{i}_test'] = FilteredMNIST(test_ds_full, task_labels)

# Create DataLoaders for all tasks
task_loaders = {}
for i in range(1, 6):
    task_loaders[f'task{i}_train'] = DataLoader(task_datasets[f'task{i}_train'], batch_size=64, shuffle=True)
    task_loaders[f'task{i}_test'] = DataLoader(task_datasets[f'task{i}_test'], batch_size=512, shuffle=False)

# ========================================
# 🚀 5-TASK EWC CONTINUAL LEARNING SEQUENCE
# ========================================

# Store results for analysis
task_results_ewc = {}
task_weights_ewc = {}

print(f"\n🔧 EWC Parameters:")
print(f"   Lambda (regularization strength): {ewc_lambda}")
print(f"   Fisher Information samples: 1000")

# Train on all 5 tasks sequentially with EWC
for task_num in range(1, 6):
    print(f"\n" + "="*60)
    print(f"PHASE {task_num}: Learning Task {task_num} ({[MNIST_LABELS[label] for label in tasks[task_num-1]]}) with EWC")
    print(f"="*60)
    
    # Train on current task with EWC
    train_task_ewc(model, task_loaders[f'task{task_num}_train'], epochs=3, 
                   task_name=f"Task {task_num}", task_num=task_num)
    
    # Evaluate current task
    loss, accuracy = evaluate_task(model, task_loaders[f'task{task_num}_test'], f"Task {task_num} Performance")
    task_results_ewc[f'task{task_num}_after_task{task_num}'] = accuracy
    
    # Store optimal parameters after training this task
    optimal_params[f'task{task_num}'] = {name: param.data.clone().cpu() for name, param in model.named_parameters()}
    
    # Compute Fisher Information for this task
    fisher_information[f'task{task_num}'] = compute_fisher_information(
        model, task_loaders[f'task{task_num}_train'], num_samples=1000
    )
    
    # Store weights after current task
    task_weights_ewc[f'task{task_num}'] = {name: param.data.clone().cpu() for name, param in model.named_parameters()}
    
    # Evaluate all previous tasks to check forgetting
    print(f"\n📊 EVALUATING ALL PREVIOUS TASKS:")
    print(f"-" * 60)
    for prev_task in range(1, task_num):
        prev_loss, prev_acc = evaluate_task(model, task_loaders[f'task{prev_task}_test'], 
                                          f"Task {prev_task} Performance (AFTER Task {task_num})")
        task_results_ewc[f'task{prev_task}_after_task{task_num}'] = prev_acc

# ========================================
# 📊 EWC FORGETTING ANALYSIS
# ========================================

print(f"\n" + "="*80)
print(f"FINAL EWC CATASTROPHIC FORGETTING ANALYSIS")
print(f"="*80)

# Calculate forgetting for each task with EWC
forgetting_analysis_ewc = {}
for task_num in range(1, 6):
    original_acc = task_results_ewc[f'task{task_num}_after_task{task_num}']
    final_acc = task_results_ewc[f'task{task_num}_after_task5']  # After all tasks
    forgetting = original_acc - final_acc
    forgetting_analysis_ewc[task_num] = {
        'original': original_acc,
        'final': final_acc,
        'forgetting': forgetting,
        'forgetting_pct': forgetting * 100
    }

# Print EWC forgetting analysis
print(f"\n📊 TASK-BY-TASK FORGETTING ANALYSIS (WITH EWC):")
print(f"=" * 80)
print(f"{'Task':<6} {'Original Acc':<12} {'Final Acc':<12} {'Forgetting':<12} {'Forgetting %':<12}")
print(f"-" * 80)

total_forgetting_ewc = 0
for task_num in range(1, 6):
    data = forgetting_analysis_ewc[task_num]
    print(f"Task {task_num:<4} {data['original']:<12.4f} {data['final']:<12.4f} {data['forgetting']:<12.4f} {data['forgetting_pct']:<12.1f}%")
    total_forgetting_ewc += data['forgetting']

avg_forgetting_ewc = total_forgetting_ewc / 5
print(f"-" * 80)
print(f"Average Forgetting (EWC): {avg_forgetting_ewc:.4f} ({avg_forgetting_ewc*100:.1f}%)")

# EWC Summary
if avg_forgetting_ewc > 0.3:
    print(f"\n❌ SEVERE CATASTROPHIC FORGETTING DETECTED (even with EWC)!")
    print(f"   Average forgetting: {avg_forgetting_ewc*100:.1f}%")
elif avg_forgetting_ewc > 0.1:
    print(f"\n⚠️  MODERATE CATASTROPHIC FORGETTING DETECTED (with EWC)!")
    print(f"   Average forgetting: {avg_forgetting_ewc*100:.1f}%")
else:
    print(f"\n✅ MINIMAL CATASTROPHIC FORGETTING (EWC is working)!")
    print(f"   Average forgetting: {avg_forgetting_ewc*100:.1f}%")

print(f"\nEWC (Elastic Weight Consolidation) helps prevent catastrophic forgetting by:")
print(f"1. Computing Fisher Information Matrix to identify important parameters")
print(f"2. Adding regularization loss that penalizes changes to important weights")
print(f"3. Preserving knowledge from previous tasks while learning new ones")

# 📈 VISUALIZATION - Plot EWC Forgetting Effect
def plot_ewc_forgetting_effect():
    """Create a bar chart showing catastrophic forgetting with EWC across 5 tasks"""
    
    # Data for plotting
    task_names = [f'Task {i}' for i in range(1, 6)]
    original_accs = [forgetting_analysis_ewc[i]['original'] for i in range(1, 6)]
    final_accs = [forgetting_analysis_ewc[i]['final'] for i in range(1, 6)]
    forgetting_amounts = [forgetting_analysis_ewc[i]['forgetting'] for i in range(1, 6)]
    
    # Create bar plot
    x = np.arange(len(task_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, original_accs, width, label='Original Accuracy', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, final_accs, width, label='Final Accuracy (with EWC)', color='blue', alpha=0.7)
    
    # Add accuracy values on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Tasks', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Catastrophic Forgetting with EWC Regularization', fontsize=14, fontweight='bold', loc='center', y=1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(task_names)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('catastrophic_forgetting_ewc_5tasks.png', dpi=150, bbox_inches='tight')
    plt.show()

# Create the EWC forgetting visualization
print(f"\n📊 Creating EWC Catastrophic Forgetting Visualization...")
plot_ewc_forgetting_effect()

print(f"\n🎉 EWC Continual Learning Experiment Complete!")
print(f"Compare the results with the standard training to see EWC's effectiveness.")
