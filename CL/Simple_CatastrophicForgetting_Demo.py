"""
Simple Catastrophic Forgetting Demonstration
Train on 3 tasks sequentially and observe catastrophic forgetting:
Task 1: digits 1,2,3
Task 2: digits 4,5,6
Task 3: digits 7,8,9
"""

import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
                # Remap labels: e.g., [1,2,3,4,5] → [0,1,2,3,4]
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


# Training function
def train(model, train_loader, epochs, task_name):
    print(f"\n{'='*60}")
    print(f"Training on {task_name}")
    print(f"{'='*60}")
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
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
# EXPERIMENT: Catastrophic Forgetting
# ========================================

print("\n" + "="*80)
print("CATASTROPHIC FORGETTING DEMONSTRATION")
print("="*80)
print("\nTask 1: Digits 1, 2, 3")
print("Task 2: Digits 4, 5, 6")
print("Task 3: Digits 7, 8, 9")
print("\nWe'll train sequentially and observe catastrophic forgetting.")

# Define tasks
task1_labels = [1, 2, 3]  # Task 1
task2_labels = [4, 5, 6]  # Task 2
task3_labels = [7, 8, 9]  # Task 3

# Create datasets
print("\n" + "="*60)
print("CREATING DATASETS")
print("="*60)

task1_train = FilteredMNIST(train_ds_full, task1_labels)
task1_test = FilteredMNIST(test_ds_full, task1_labels)

task2_train = FilteredMNIST(train_ds_full, task2_labels)
task2_test = FilteredMNIST(test_ds_full, task2_labels)

task3_train = FilteredMNIST(train_ds_full, task3_labels)
task3_test = FilteredMNIST(test_ds_full, task3_labels)

# Create DataLoaders
task1_train_loader = DataLoader(task1_train, batch_size=64, shuffle=True)
task1_test_loader = DataLoader(task1_test, batch_size=512, shuffle=False)

task2_train_loader = DataLoader(task2_train, batch_size=64, shuffle=True)
task2_test_loader = DataLoader(task2_test, batch_size=512, shuffle=False)

task3_train_loader = DataLoader(task3_train, batch_size=64, shuffle=True)
task3_test_loader = DataLoader(task3_test, batch_size=512, shuffle=False)

# Initialize model
model = MLPModel(num_classes=3).to(DEVICE)  # 3 classes per task

# ========================================
# PHASE 1: Train on Task 1 (digits 1,2,3)
# ========================================

train(model, task1_train_loader, epochs=5, task_name="Task 1 (digits 1,2,3)")

print(f"\n📊 After Training Task 1:")
task1_acc_after_task1 = evaluate(model, task1_test_loader, "Task 1 Performance")

# ========================================
# PHASE 2: Train on Task 2 (digits 4,5,6)
# ========================================

train(model, task2_train_loader, epochs=5, task_name="Task 2 (digits 4,5,6)")

print(f"\n📊 After Training Task 2:")
task2_acc_after_task2 = evaluate(model, task2_test_loader, "Task 2 Performance")
task1_acc_after_task2 = evaluate(model, task1_test_loader, "Task 1 Performance")

# ========================================
# PHASE 3: Train on Task 3 (digits 7,8,9)
# ========================================

train(model, task3_train_loader, epochs=5, task_name="Task 3 (digits 7,8,9)")

print(f"\n📊 After Training Task 3:")
task3_acc_after_task3 = evaluate(model, task3_test_loader, "Task 3 Performance")
task2_acc_after_task3 = evaluate(model, task2_test_loader, "Task 2 Performance")
task1_acc_after_task3 = evaluate(model, task1_test_loader, "Task 1 Performance")

# ========================================
# ANALYSIS: Catastrophic Forgetting
# ========================================

print("\n" + "="*80)
print("CATASTROPHIC FORGETTING ANALYSIS")
print("="*80)

# Task 1 forgetting
task1_forgetting = task1_acc_after_task1 - task1_acc_after_task3
task1_forgetting_pct = task1_forgetting * 100

# Task 2 forgetting
task2_forgetting = task2_acc_after_task2 - task2_acc_after_task3
task2_forgetting_pct = task2_forgetting * 100

print(f"\nTask 1 (digits 1,2,3):")
print(f"  After Task 1 training: {task1_acc_after_task1:.4f} ({task1_acc_after_task1*100:.2f}%)")
print(f"  After Task 2 training: {task1_acc_after_task2:.4f} ({task1_acc_after_task2*100:.2f}%)")
print(f"  After Task 3 training: {task1_acc_after_task3:.4f} ({task1_acc_after_task3*100:.2f}%)")
print(f"  Total Forgetting: {task1_forgetting:.4f} ({task1_forgetting_pct:.1f}%)")

print(f"\nTask 2 (digits 4,5,6):")
print(f"  After Task 2 training: {task2_acc_after_task2:.4f} ({task2_acc_after_task2*100:.2f}%)")
print(f"  After Task 3 training: {task2_acc_after_task3:.4f} ({task2_acc_after_task3*100:.2f}%)")
print(f"  Forgetting: {task2_forgetting:.4f} ({task2_forgetting_pct:.1f}%)")

print(f"\nTask 3 (digits 7,8,9):")
print(f"  After Task 3 training: {task3_acc_after_task3:.4f} ({task3_acc_after_task3*100:.2f}%)")

avg_forgetting = (task1_forgetting + task2_forgetting) / 2
avg_forgetting_pct = avg_forgetting * 100

print(f"\nAverage Forgetting: {avg_forgetting:.4f} ({avg_forgetting_pct:.1f}%)")

if avg_forgetting > 0.3:
    print(f"\n❌ SEVERE CATASTROPHIC FORGETTING!")
    print(f"   The model forgot {avg_forgetting_pct:.1f}% on average!")
elif avg_forgetting > 0.1:
    print(f"\n⚠️  MODERATE CATASTROPHIC FORGETTING!")
    print(f"   The model forgot {avg_forgetting_pct:.1f}% on average!")
else:
    print(f"\n✅ MINIMAL FORGETTING")
    print(f"   Only {avg_forgetting_pct:.1f}% forgetting on average")

print(f"\n💡 Why did this happen?")
print(f"   When training on Task 2, the model updates weights to minimize Task 2 loss.")
print(f"   These updates overwrite what it learned for Task 1.")
print(f"   Same happens with Task 3 - it overwrites both Task 1 and Task 2!")
print(f"   The model has no memory of previous tasks during new task training!")

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
                    label='Final Accuracy (After Task 3)', color='red', alpha=0.7)
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Task', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Catastrophic Forgetting: Original vs Final Accuracy', fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Task 1\n(1,2,3)', 'Task 2\n(4,5,6)', 'Task 3\n(7,8,9)'], fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('simple_catastrophic_forgetting_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n📊 Visualization saved as 'simple_catastrophic_forgetting_demo.png'")


plot_results()

print(f"\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)
print(f"\nKey Observations:")
print(f"  1. Task 1: {task1_acc_after_task1*100:.1f}% → {task1_acc_after_task3*100:.1f}% (dropped {task1_forgetting_pct:.1f}%)")
print(f"  2. Task 2: {task2_acc_after_task2*100:.1f}% → {task2_acc_after_task3*100:.1f}% (dropped {task2_forgetting_pct:.1f}%)")
print(f"  3. Task 3: {task3_acc_after_task3*100:.1f}% (just trained, still good)")
print(f"\n💥 This is Catastrophic Forgetting in action!")
print(f"   Older tasks are forgotten when learning new tasks!")

