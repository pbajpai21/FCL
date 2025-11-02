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
# 🧠 REPRESENTATION-BASED APPROACH
# ========================================
# This implementation uses a Shared-Private Architecture:
# - Shared Encoder: Learns common features across all tasks
# - Task-Specific Heads: Task-specific classification layers
# - Representation Freezing: Protect shared representations when learning new tasks

class SharedEncoder(nn.Module):
    """
    Shared encoder that learns task-agnostic representations.
    This is the key component - it learns features useful across all tasks.
    """
    def __init__(self, input_size=28*28, hidden_size=512):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),  # Shared representation space
            nn.ReLU(),
        )
        self.repr_size = 256
        
    def forward(self, x):
        return self.shared_net(x)  # Returns shared representation


class TaskSpecificHead(nn.Module):
    """
    Task-specific classification head.
    Each task gets its own head, but shares the encoder.
    """
    def __init__(self, input_size=256, num_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.head(x)


class RepresentationBasedModel(nn.Module):
    """
    Representation-Based Continual Learning Model.
    
    Key Design:
    1. Shared encoder learns common features
    2. Each task has its own head
    3. We can freeze/shield the encoder when learning new tasks
    """
    def __init__(self, num_classes_per_task=2, freeze_encoder_after_task=1):
        super().__init__()
        self.shared_encoder = SharedEncoder()
        self.task_heads = nn.ModuleDict()  # Store task-specific heads
        self.num_classes_per_task = num_classes_per_task
        self.freeze_encoder_after_task = freeze_encoder_after_task
        self.current_task = 0
        
    def add_task_head(self, task_num):
        """Add a new task-specific head when learning a new task"""
        task_name = f'task_{task_num}'
        self.task_heads[task_name] = TaskSpecificHead(
            input_size=self.shared_encoder.repr_size,
            num_classes=self.num_classes_per_task
        )
        print(f"✅ Added head for Task {task_num}")
        
    def freeze_shared_encoder(self):
        """Freeze shared encoder to protect learned representations"""
        for param in self.shared_encoder.parameters():
            param.requires_grad = False
        print(f"🔒 Shared encoder FROZEN - representations are protected!")
        
    def unfreeze_shared_encoder(self):
        """Unfreeze shared encoder (for initial task training)"""
        for param in self.shared_encoder.parameters():
            param.requires_grad = True
        print(f"🔓 Shared encoder UNFROZEN - learning representations!")
    
    def forward(self, x, task_num=None):
        """
        Forward pass:
        1. Extract shared representation
        2. Pass through task-specific head
        """
        if task_num is None:
            task_num = self.current_task
            
        # Extract shared representation
        shared_repr = self.shared_encoder(x)
        
        # Use appropriate task head
        task_name = f'task_{task_num}'
        if task_name in self.task_heads:
            output = self.task_heads[task_name](shared_repr)
        else:
            raise ValueError(f"Task {task_num} head not found! Call add_task_head({task_num}) first.")
        
        return output
    
    def set_current_task(self, task_num):
        """Set which task we're currently working on"""
        self.current_task = task_num


# Initialize model
model = RepresentationBasedModel(
    num_classes_per_task=2,
    freeze_encoder_after_task=1  # Freeze encoder after first task
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

# ========================================
# 🚀 TRAINING WITH REPRESENTATION PROTECTION
# ========================================

def train_task_representation_based(model, train_loader, epochs=5, task_num=1):
    """
    Train using representation-based approach:
    - Task 1: Train both encoder and head (learn representations)
    - Task 2+: Freeze encoder, only train task-specific head
    """
    import time
    
    model.train()
    task_name = f'task_{task_num}'
    
    # Add task head if it doesn't exist
    if task_name not in model.task_heads:
        model.add_task_head(task_num)
    
    model.set_current_task(task_num)
    
    # Configure encoder freezing based on task number
    if task_num == 1:
        # First task: learn both encoder and head
        model.unfreeze_shared_encoder()
        print(f"🚀 Training Task {task_num} - Learning shared representations...")
        
        # Create optimizer for both encoder and head
        encoder_params = list(model.shared_encoder.parameters())
        head_params = list(model.task_heads[task_name].parameters())
        optimizer = torch.optim.Adam(encoder_params + head_params, lr=0.001)
    else:
        # Subsequent tasks: freeze encoder, only train head
        model.freeze_shared_encoder()
        print(f"🚀 Training Task {task_num} - Using frozen representations, training task head only...")
        
        # Optimizer only for task head
        head_params = list(model.task_heads[task_name].parameters())
        optimizer = torch.optim.Adam(head_params, lr=0.001)
    
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


def analyze_representations(model, task_num, test_loader):
    """
    Analyze the learned representations to show how they're shared.
    This demonstrates that tasks use the same representation space.
    """
    model.eval()
    representations = []
    labels_list = []
    
    print(f"🔍 Analyzing representations for Task {task_num}...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            # Extract shared representation (before task head)
            reprs = model.shared_encoder(images)
            representations.append(reprs.cpu().numpy())
            labels_list.append(labels.numpy())
    
    representations = np.concatenate(representations, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    
    # Compute representation statistics
    mean_repr = np.mean(representations, axis=0)
    std_repr = np.std(representations, axis=0)
    
    print(f"  Representation shape: {representations.shape}")
    print(f"  Mean representation norm: {np.linalg.norm(mean_repr):.4f}")
    print(f"  Std representation norm: {np.linalg.norm(std_repr):.4f}")
    
    return representations, labels_list


# ========================================
# 🎯 EXPERIMENT SETUP
# ========================================

print(f"\n🎯 Setting up Representation-Based Continual Learning Experiment")
print(f"Key Idea: Learn shared representations in first task, freeze them, then reuse for all tasks")

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
# 🚀 REPRESENTATION-BASED CONTINUAL LEARNING
# ========================================

task_results_rep = {}
task_representations = {}

print(f"\n🔧 Representation-Based Parameters:")
print(f"   Task 1: Train encoder + head (learn shared representations)")
print(f"   Task 2+: Freeze encoder, train task-specific heads only")

# Train on all 5 tasks sequentially
for task_num in range(1, 6):
    print(f"\n" + "="*60)
    print(f"PHASE {task_num}: Learning Task {task_num} ({[MNIST_LABELS[label] for label in tasks[task_num-1]]})")
    print(f"="*60)
    
    # Train on current task
    train_task_representation_based(
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
    task_results_rep[f'task{task_num}_after_task{task_num}'] = accuracy
    
    # Analyze representations
    reprs, labels = analyze_representations(model, task_num, task_loaders[f'task{task_num}_test'])
    task_representations[f'task{task_num}'] = {
        'representations': reprs,
        'labels': labels,
        'mean_norm': np.linalg.norm(np.mean(reprs, axis=0))
    }
    
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
        task_results_rep[f'task{prev_task}_after_task{task_num}'] = prev_acc

# ========================================
# 📊 FORGETTING ANALYSIS
# ========================================

print(f"\n" + "="*80)
print(f"REPRESENTATION-BASED FORGETTING ANALYSIS")
print(f"="*80)

forgetting_analysis_rep = {}
for task_num in range(1, 6):
    original_acc = task_results_rep[f'task{task_num}_after_task{task_num}']
    final_acc = task_results_rep[f'task{task_num}_after_task5']
    forgetting = original_acc - final_acc
    forgetting_analysis_rep[task_num] = {
        'original': original_acc,
        'final': final_acc,
        'forgetting': forgetting,
        'forgetting_pct': forgetting * 100
    }

print(f"\n📊 TASK-BY-TASK FORGETTING ANALYSIS (REPRESENTATION-BASED):")
print(f"=" * 80)
print(f"{'Task':<6} {'Original Acc':<12} {'Final Acc':<12} {'Forgetting':<12} {'Forgetting %':<12}")
print(f"-" * 80)

total_forgetting_rep = 0
for task_num in range(1, 6):
    data = forgetting_analysis_rep[task_num]
    print(f"Task {task_num:<4} {data['original']:<12.4f} {data['final']:<12.4f} {data['forgetting']:<12.4f} {data['forgetting_pct']:<12.1f}%")
    total_forgetting_rep += data['forgetting']

avg_forgetting_rep = total_forgetting_rep / 5
print(f"-" * 80)
print(f"Average Forgetting (Representation-Based): {avg_forgetting_rep:.4f} ({avg_forgetting_rep*100:.1f}%)")

if avg_forgetting_rep < 0.05:
    print(f"\n✅ EXCELLENT! Minimal forgetting with Representation-Based approach!")
elif avg_forgetting_rep < 0.1:
    print(f"\n✅ GOOD! Low forgetting with Representation-Based approach!")
else:
    print(f"\n⚠️  Some forgetting detected, but typically less than standard training!")

print(f"\n🎓 KEY INSIGHTS:")
print(f"1. Shared encoder learns common features in Task 1")
print(f"2. Encoder is frozen after Task 1 to protect representations")
print(f"3. Each new task gets its own head, but reuses the frozen encoder")
print(f"4. This prevents interference because encoder parameters don't change")

# ========================================
# 📈 VISUALIZATION
# ========================================

def plot_representation_based_forgetting():
    """Create visualization of forgetting with representation-based approach"""
    task_names = [f'Task {i}' for i in range(1, 6)]
    original_accs = [forgetting_analysis_rep[i]['original'] for i in range(1, 6)]
    final_accs = [forgetting_analysis_rep[i]['final'] for i in range(1, 6)]
    
    x = np.arange(len(task_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, original_accs, width, label='Original Accuracy', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, final_accs, width, label='Final Accuracy (Rep-Based)', color='purple', alpha=0.7)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Tasks', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Representation-Based Continual Learning', fontsize=14, fontweight='bold', loc='center', y=1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(task_names)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('catastrophic_forgetting_repbased_5tasks.png', dpi=150, bbox_inches='tight')
    plt.show()

print(f"\n📊 Creating Representation-Based Forgetting Visualization...")
plot_representation_based_forgetting()

print(f"\n🎉 Representation-Based Continual Learning Experiment Complete!")
print(f"\n💡 Learn More:")
print(f"   - Progressive Neural Networks: Add new columns per task")
print(f"   - PackNet: Task-specific weight masks")
print(f"   - Subspace Methods: Constrain updates to subspaces")
print(f"   - All share the core idea: protect learned representations!")

