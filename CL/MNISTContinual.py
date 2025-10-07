import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

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


def train_task(model, train_loader, epochs=5, task_name=""):
    import time
    model.train()
    print(f"🚀 Training on {task_name}...")
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        for images, labels in train_loader:  # Back to 2 values
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_time = time.time() - start_time
        print(f"  Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, Time = {epoch_time:.2f}s")


def evaluate_task(model, test_loader, task_name=""):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:  # Back to 2 values
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
# 🧠 CATASTROPHIC FORGETTING DEMONSTRATION
# ========================================

print(f"\n🎯 Setting up 5-Task Continual Learning Experiment")
print(f"We'll train on 5 sequential tasks with 2 classes each")

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
# 🚀 5-TASK CONTINUAL LEARNING SEQUENCE
# ========================================

# Store results for analysis
task_results = {}
task_weights = {}

# Train on all 5 tasks sequentially
for task_num in range(1, 6):
    print(f"\n" + "="*60)
    print(f"PHASE {task_num}: Learning Task {task_num} ({[MNIST_LABELS[label] for label in tasks[task_num-1]]})")
    print(f"="*60)
    
    # Train on current task
    train_task(model, task_loaders[f'task{task_num}_train'], epochs=3, task_name=f"Task {task_num}")
    
    # Evaluate current task
    loss, accuracy = evaluate_task(model, task_loaders[f'task{task_num}_test'], f"Task {task_num} Performance")
    task_results[f'task{task_num}_after_task{task_num}'] = accuracy
    
    # Print weights after current task
    # print_network_weights(model, f"TASK {task_num}")
    task_weights[f'task{task_num}'] = {name: param.data.clone().cpu() for name, param in model.named_parameters()}
    
    # Evaluate all previous tasks to check forgetting
    print(f"\n📊 EVALUATING ALL PREVIOUS TASKS:")
    print(f"-" * 60)
    for prev_task in range(1, task_num):
        prev_loss, prev_acc = evaluate_task(model, task_loaders[f'task{prev_task}_test'], 
                                          f"Task {prev_task} Performance (AFTER Task {task_num})")
        task_results[f'task{prev_task}_after_task{task_num}'] = prev_acc

# ========================================
# 📊 FINAL ANALYSIS
# ========================================

print(f"\n" + "="*80)
print(f"FINAL CATASTROPHIC FORGETTING ANALYSIS")
print(f"="*80)

# Calculate forgetting for each task
forgetting_analysis = {}
for task_num in range(1, 6):
    original_acc = task_results[f'task{task_num}_after_task{task_num}']
    final_acc = task_results[f'task{task_num}_after_task5']  # After all tasks
    forgetting = original_acc - final_acc
    forgetting_analysis[task_num] = {
        'original': original_acc,
        'final': final_acc,
        'forgetting': forgetting,
        'forgetting_pct': forgetting * 100
    }

# Print forgetting analysis
print(f"\n📊 TASK-BY-TASK FORGETTING ANALYSIS:")
print(f"=" * 80)
print(f"{'Task':<6} {'Original Acc':<12} {'Final Acc':<12} {'Forgetting':<12} {'Forgetting %':<12}")
print(f"-" * 80)

total_forgetting = 0
for task_num in range(1, 6):
    data = forgetting_analysis[task_num]
    print(f"Task {task_num:<4} {data['original']:<12.4f} {data['final']:<12.4f} {data['forgetting']:<12.4f} {data['forgetting_pct']:<12.1f}%")
    total_forgetting += data['forgetting']

avg_forgetting = total_forgetting / 5
print(f"-" * 80)
print(f"Average Forgetting: {avg_forgetting:.4f} ({avg_forgetting*100:.1f}%)")

# Summary
if avg_forgetting > 0.3:
    print(f"\n❌ SEVERE CATASTROPHIC FORGETTING DETECTED!")
    print(f"   Average forgetting: {avg_forgetting*100:.1f}%")
elif avg_forgetting > 0.1:
    print(f"\n⚠️  MODERATE CATASTROPHIC FORGETTING DETECTED!")
    print(f"   Average forgetting: {avg_forgetting*100:.1f}%")
else:
    print(f"\n✅ MINIMAL CATASTROPHIC FORGETTING!")
    print(f"   Average forgetting: {avg_forgetting*100:.1f}%")

print(f"\nThis demonstrates the challenge of continual learning across multiple tasks:")
print(f"Neural networks tend to 'forget' old tasks when learning new ones.")

# 📈 VISUALIZATION - Plot Forgetting Percentage
def plot_forgetting_effect():
    """Create a bar chart showing catastrophic forgetting across 5 tasks"""
    
    # Data for plotting
    task_names = [f'Task {i}' for i in range(1, 6)]
    original_accs = [forgetting_analysis[i]['original'] for i in range(1, 6)]
    final_accs = [forgetting_analysis[i]['final'] for i in range(1, 6)]
    forgetting_amounts = [forgetting_analysis[i]['forgetting'] for i in range(1, 6)]
    
    # Create bar plot
    x = np.arange(len(task_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, original_accs, width, label='Original Accuracy', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, final_accs, width, label='Final Accuracy', color='red', alpha=0.7)
    
    # Add accuracy values on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Tasks', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Catastrophic Forgetting Across 5 Tasks', fontsize=14, fontweight='bold', loc='center', y=1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(task_names)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('catastrophic_forgetting_5tasks.png', dpi=150, bbox_inches='tight')
    plt.show()

# Create the forgetting visualization
print(f"\n📊 Creating Catastrophic Forgetting Visualization...")
plot_forgetting_effect()

