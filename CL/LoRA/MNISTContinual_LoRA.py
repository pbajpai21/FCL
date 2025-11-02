import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy

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

# MNIST class labels
MNIST_LABELS = {
    0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four',
    5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'
}

class FilteredMNIST:
    def __init__(self, original_dataset, target_labels):
        self.data = []
        self.targets = []
        self.target_labels = target_labels
        
        print(f"Creating dataset with labels {target_labels}...")
        print(f"Class names: {[MNIST_LABELS[label] for label in target_labels]}")
        
        for i in range(len(original_dataset)):
            image, label = original_dataset[i]
            if label in target_labels:
                self.data.append(image)
                new_label = target_labels.index(label)
                self.targets.append(new_label)
        
        print(f"Dataset created: {len(self.data)} samples, labels remapped to {set(self.targets)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# ========================================
# 🧠 LoRA (Low-Rank Adaptation) Implementation
# ========================================

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer.
    
    Wraps a linear layer and adds low-rank adaptation:
    output = Wx + (α/r) * (B @ A @ x)
    
    Where:
    - W: Original weight matrix (frozen)
    - A: Down-projection matrix (d × r), trainable
    - B: Up-projection matrix (r × d), trainable
    - r: Rank (typically 1-16)
    - α: Scaling factor (typically = r)
    """
    def __init__(self, linear_layer, rank=8, alpha=8):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze the original weights
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # Get dimensions
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        # Initialize LoRA matrices
        # A: random initialization
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        # B: zero initialization (so BA = 0 initially)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x):
        # Original layer output (frozen weights)
        base_output = self.linear(x)
        
        # LoRA adaptation: (α/r) * B @ (A @ x)
        # A @ x: (rank, in_features) @ (batch, in_features).T = (rank, batch)
        # B @ (A @ x): (out_features, rank) @ (rank, batch) = (out_features, batch)
        # Transpose back: (batch, out_features)
        lora_output = (self.alpha / self.rank) * torch.matmul(self.lora_B, torch.matmul(self.lora_A, x.T)).T
        
        return base_output + lora_output
    
    def get_lora_params(self):
        """Return LoRA parameters (for optimizer)"""
        return [self.lora_A, self.lora_B]
    
    def count_lora_params(self):
        """Count the number of LoRA parameters"""
        return self.lora_A.numel() + self.lora_B.numel()


class MLPWithLoRA(nn.Module):
    """
    MLP model with LoRA adapters for continual learning.
    
    Base model architecture is frozen after Task 1.
    Each subsequent task gets its own LoRA adapters.
    """
    def __init__(self, num_classes=2, rank=8, alpha=8):
        super().__init__()
        self.num_classes = num_classes
        self.rank = rank
        self.alpha = alpha
        
        # Base network (will be frozen after Task 1)
        self.base_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        
        # Store LoRA adapters per task
        self.lora_adapters = nn.ModuleDict()
        self.current_task = 0
        
    def add_task_adapters(self, task_num):
        """Add LoRA adapters for a new task"""
        task_name = f'task_{task_num}'
        
        if task_name in self.lora_adapters:
            print(f"⚠️  Adapters for Task {task_num} already exist!")
            return
        
        # Create LoRA layers for each linear layer
        lora_layers = nn.ModuleDict()
        
        # Find all Linear layers and wrap them with LoRA
        layer_idx = 0
        for i, module in enumerate(self.base_net):
            if isinstance(module, nn.Linear):
                layer_name = f'layer_{layer_idx}'
                lora_layers[layer_name] = LoRALayer(module, rank=self.rank, alpha=self.alpha)
                layer_idx += 1
        
        self.lora_adapters[task_name] = lora_layers
        print(f"✅ Added LoRA adapters for Task {task_num} (rank={self.rank}, {self._count_task_params(task_name)} params)")
    
    def _count_task_params(self, task_name):
        """Count LoRA parameters for a task"""
        total = 0
        for layer in self.lora_adapters[task_name].values():
            total += layer.count_lora_params()
        return total
    
    def freeze_base_model(self):
        """Freeze the base model (call after Task 1)"""
        for param in self.base_net.parameters():
            param.requires_grad = False
        print(f"🔒 Base model FROZEN - only LoRA adapters will be trainable")
    
    def set_current_task(self, task_num):
        """Set which task's adapters to use"""
        self.current_task = task_num
    
    def forward(self, x, task_num=None):
        """Forward pass with task-specific LoRA adapters"""
        if task_num is None:
            task_num = self.current_task
        
        task_name = f'task_{task_num}'
        
        if task_name not in self.lora_adapters:
            raise ValueError(f"LoRA adapters for Task {task_num} not found! Call add_task_adapters({task_num}) first.")
        
        # Apply base network with LoRA adapters
        h = x
        layer_idx = 0
        
        for i, module in enumerate(self.base_net):
            if isinstance(module, nn.Flatten):
                h = module(h)
            elif isinstance(module, nn.Linear):
                layer_name = f'layer_{layer_idx}'
                # Use LoRA-wrapped layer for this task
                lora_layer = self.lora_adapters[task_name][layer_name]
                h = lora_layer(h)
                layer_idx += 1
            elif isinstance(module, nn.ReLU):
                h = module(h)
        
        return h
    
    def get_task_lora_params(self, task_num):
        """Get all LoRA parameters for a specific task (for optimizer)"""
        task_name = f'task_{task_num}'
        if task_name not in self.lora_adapters:
            return []
        
        params = []
        for layer in self.lora_adapters[task_name].values():
            params.extend(layer.get_lora_params())
        
        return params
    
    def count_total_base_params(self):
        """Count total parameters in base model"""
        return sum(p.numel() for p in self.base_net.parameters())
    
    def count_total_lora_params(self):
        """Count total LoRA parameters across all tasks"""
        total = 0
        for task_adapters in self.lora_adapters.values():
            for layer in task_adapters.values():
                total += layer.count_lora_params()
        return total


# Initialize model
lora_rank = 8  # LoRA rank (hyperparameter)
lora_alpha = 8  # LoRA alpha (scaling factor)

model = MLPWithLoRA(num_classes=2, rank=lora_rank, alpha=lora_alpha).to(DEVICE)
criterion = nn.CrossEntropyLoss()

# Print model info
base_params = model.count_total_base_params()
print(f"\n📊 Model Information:")
print(f"   Base model parameters: {base_params:,}")
print(f"   LoRA rank: {lora_rank}")
print(f"   LoRA alpha: {lora_alpha}")


# ========================================
# 🚀 TRAINING WITH LoRA
# ========================================

def train_task_lora(model, train_loader, epochs=5, task_num=1):
    """
    Train using LoRA adapters:
    - Task 1: Train base model + LoRA adapters
    - Task 2+: Only train new LoRA adapters (base model frozen)
    """
    import time
    
    model.train()
    
    # Add task adapters if they don't exist
    if f'task_{task_num}' not in model.lora_adapters:
        model.add_task_adapters(task_num)
    
    model.set_current_task(task_num)
    
    # Configure training based on task number
    if task_num == 1:
        # Task 1: Train both base model and LoRA adapters
        print(f"🚀 Training Task {task_num} - Learning base model + LoRA adapters...")
        
        # Unfreeze base model for Task 1
        for param in model.base_net.parameters():
            param.requires_grad = True
        
        # Get all parameters (base + LoRA)
        base_params = list(model.base_net.parameters())
        lora_params = model.get_task_lora_params(task_num)
        optimizer = torch.optim.Adam(base_params + lora_params, lr=0.001)
    else:
        # Task 2+: Only train LoRA adapters (base model frozen)
        model.freeze_base_model()
        print(f"🚀 Training Task {task_num} - Using frozen base model, training LoRA adapters only...")
        
        # Only LoRA parameters
        lora_params = model.get_task_lora_params(task_num)
        optimizer = torch.optim.Adam(lora_params, lr=0.001)
        
        lora_param_count = sum(p.numel() for p in lora_params)
        base_param_count = model.count_total_base_params()
        print(f"   Training {lora_param_count:,} LoRA parameters (vs {base_param_count:,} base parameters)")
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images, task_num=task_num)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s")


def evaluate_task(model, test_loader, task_name="", task_num=None):
    """Evaluate model on a specific task"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    if task_num is None:
        task_num = model.current_task
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images, task_num=task_num)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"📊 {task_name} - Loss: {total_loss/total:.4f}, Accuracy: {accuracy:.4f}")
    return total_loss/total, accuracy


def print_model_stats(model):
    """Print detailed model statistics"""
    base_params = model.count_total_base_params()
    total_lora_params = model.count_total_lora_params()
    
    print(f"\n📊 MODEL STATISTICS:")
    print(f"   Base model parameters: {base_params:,}")
    print(f"   Total LoRA parameters (all tasks): {total_lora_params:,}")
    print(f"   LoRA efficiency: {total_lora_params/base_params*100:.2f}% of base model size")
    print(f"   Tasks with adapters: {len(model.lora_adapters)}")
    
    for task_name in model.lora_adapters.keys():
        task_params = model._count_task_params(task_name)
        print(f"     {task_name}: {task_params:,} LoRA parameters")


# ========================================
# 🎯 EXPERIMENT SETUP
# ========================================

print(f"\n🎯 Setting up LoRA-based Continual Learning Experiment")
print(f"Key Idea: Freeze base model after Task 1, add small LoRA adapters per task")

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
# 🚀 LoRA CONTINUAL LEARNING
# ========================================

task_results_lora = {}

print(f"\n🔧 LoRA Parameters:")
print(f"   Rank (r): {lora_rank}")
print(f"   Alpha (α): {lora_alpha}")
print(f"   Task 1: Train base model + LoRA adapters")
print(f"   Task 2+: Freeze base model, train task-specific LoRA adapters only")
print(f"\n⚠️  KEY INSIGHT:")
print(f"   LoRA adapters are tiny (~{lora_rank*2*512 + lora_rank*2*256 + lora_rank*2*2} params per task)")
print(f"   vs base model ({base_params:,} params)")
print(f"   This allows efficient multi-task learning!")

# Train on all 5 tasks sequentially
for task_num in range(1, 6):
    print(f"\n" + "="*60)
    print(f"PHASE {task_num}: Learning Task {task_num} ({[MNIST_LABELS[label] for label in tasks[task_num-1]]})")
    print(f"="*60)
    
    # Train on current task
    train_task_lora(
        model, 
        task_loaders[f'task{task_num}_train'], 
        epochs=3, 
        task_num=task_num
    )
    
    # Evaluate current task
    loss, accuracy = evaluate_task(
        model, 
        task_loaders[f'task{task_num}_test'], 
        f"Task {task_num} Performance",
        task_num=task_num
    )
    task_results_lora[f'task{task_num}_after_task{task_num}'] = accuracy
    
    # Evaluate all previous tasks to check forgetting
    print(f"\n📊 EVALUATING ALL PREVIOUS TASKS:")
    print(f"-" * 60)
    for prev_task in range(1, task_num + 1):
        prev_loss, prev_acc = evaluate_task(
            model, 
            task_loaders[f'task{prev_task}_test'], 
            f"Task {prev_task} Performance (AFTER Task {task_num})",
            task_num=prev_task
        )
        task_results_lora[f'task{prev_task}_after_task{task_num}'] = prev_acc
    
    # Print model stats
    if task_num == 1 or task_num == 5:
        print_model_stats(model)

# ========================================
# 📊 FORGETTING ANALYSIS
# ========================================

print(f"\n" + "="*80)
print(f"LoRA FORGETTING ANALYSIS")
print(f"="*80)

forgetting_analysis_lora = {}
for task_num in range(1, 6):
    original_acc = task_results_lora[f'task{task_num}_after_task{task_num}']
    final_acc = task_results_lora[f'task{task_num}_after_task5']
    forgetting = original_acc - final_acc
    forgetting_analysis_lora[task_num] = {
        'original': original_acc,
        'final': final_acc,
        'forgetting': forgetting,
        'forgetting_pct': forgetting * 100
    }

print(f"\n📊 TASK-BY-TASK FORGETTING ANALYSIS (LoRA):")
print(f"=" * 80)
print(f"{'Task':<6} {'Original Acc':<12} {'Final Acc':<12} {'Forgetting':<12} {'Forgetting %':<12}")
print(f"-" * 80)

total_forgetting_lora = 0
for task_num in range(1, 6):
    data = forgetting_analysis_lora[task_num]
    print(f"Task {task_num:<4} {data['original']:<12.4f} {data['final']:<12.4f} {data['forgetting']:<12.4f} {data['forgetting_pct']:<12.1f}%")
    total_forgetting_lora += data['forgetting']

avg_forgetting_lora = total_forgetting_lora / 5
print(f"-" * 80)
print(f"Average Forgetting (LoRA): {avg_forgetting_lora:.4f} ({avg_forgetting_lora*100:.1f}%)")

if avg_forgetting_lora < 0.05:
    print(f"\n✅ EXCELLENT! Minimal forgetting with LoRA approach!")
elif avg_forgetting_lora < 0.1:
    print(f"\n✅ GOOD! Low forgetting with LoRA approach!")
else:
    print(f"\n⚠️  Some forgetting detected, but typically less than standard training!")

print(f"\n🎓 KEY INSIGHTS:")
print(f"1. Base model learned in Task 1, then frozen")
print(f"2. Each new task gets tiny LoRA adapters (rank={lora_rank})")
print(f"3. LoRA adapters are isolated - no interference between tasks")
print(f"4. Parameter efficient: ~{model.count_total_lora_params()/base_params*100:.1f}% of base model size")
print(f"5. Perfect isolation: Each task uses its own adapter, base model unchanged")

# ========================================
# 📈 VISUALIZATION
# ========================================

def plot_lora_forgetting():
    """Create visualization of forgetting with LoRA approach"""
    task_names = [f'Task {i}' for i in range(1, 6)]
    original_accs = [forgetting_analysis_lora[i]['original'] for i in range(1, 6)]
    final_accs = [forgetting_analysis_lora[i]['final'] for i in range(1, 6)]
    
    x = np.arange(len(task_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, original_accs, width, label='Original Accuracy', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, final_accs, width, label='Final Accuracy (LoRA)', color='orange', alpha=0.7)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Tasks', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('LoRA-based Continual Learning', fontsize=14, fontweight='bold', loc='center', y=1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(task_names)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('catastrophic_forgetting_lora_5tasks.png', dpi=150, bbox_inches='tight')
    plt.show()

print(f"\n📊 Creating LoRA Forgetting Visualization...")
plot_lora_forgetting()

print(f"\n🎉 LoRA Continual Learning Experiment Complete!")
print(f"\n💡 LoRA Advantages:")
print(f"   - Parameter efficient: Only train tiny adapters per task")
print(f"   - Perfect isolation: Each task's adapter is independent")
print(f"   - Scalable: Can handle many tasks efficiently")
print(f"   - Fast training: Small adapters train quickly")
print(f"\n🔍 Compare with:")
print(f"   - EWC: Regularizes weight changes")
print(f"   - ER: Replays old data")
print(f"   - Shared-Private: Freezes encoder, adds heads")
print(f"   - LoRA: Adds tiny adapters to frozen base model")

