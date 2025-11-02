# Representation-Based Approaches in Continual Learning

## 📚 Overview

**Representation-Based Approaches** are a family of continual learning methods that focus on **learning and protecting good representations** rather than just preserving specific weight values. These methods prevent catastrophic forgetting by maintaining reusable feature spaces across tasks.

---

## 🎯 Core Philosophy

### Traditional Approaches (EWC, LwF) vs. Representation-Based

| Approach | What They Protect | How They Work |
|----------|------------------|---------------|
| **EWC** | Individual weight values | Regularizes weight changes based on importance |
| **LwF** | Output distributions | Distillation loss on old task outputs |
| **Representation-Based** | **Learned feature representations** | Separates shared features from task-specific ones |

### Key Insight

> "Instead of protecting all weights, learn good **shared representations** that can be reused across tasks, and only learn task-specific components."

---

## 🏗️ Architecture Pattern: Shared-Private Model

The fundamental architecture pattern in representation-based methods:

```
Input → [Shared Encoder] → [Task-Specific Head]
             ↓                      ↓
    Learns common        Task 1: Head 1
    features across      Task 2: Head 2
    all tasks            Task 3: Head 3
```

### Components:

1. **Shared Encoder** (Feature Extractor)
   - Learns task-agnostic representations
   - Trained on first task(s)
   - Frozen after initial training to protect representations

2. **Task-Specific Heads**
   - One head per task
   - Only this part is trained for new tasks
   - Reuses frozen shared encoder

---

## 🔬 Key Methods in Representation-Based Approaches

### ⚠️ Important Distinction: Architecture Modification vs. Fixed Architecture

**NOT all representation-based methods modify architecture!** They fall into two categories:

#### **Category A: Methods That Modify Architecture** 🔨
- **Progressive Neural Networks**: Adds new columns
- **Dynamic Expansion**: Grows network size

#### **Category B: Methods With Fixed Architecture** 🔒
- **PackNet**: Uses masks (same architecture)
- **Subspace Methods**: Same architecture, constrained updates
- **Shared-Private**: Adds small heads but core architecture fixed
- **Low-Rank Adaptation (LoRA)**: Adds small adapter modules

---

### 1. **Progressive Neural Networks (PNN)** 🔨 *[Modifies Architecture]*
**Paper**: Rusu et al., "Progressive Neural Networks" (2016)

**Idea**: Add new columns (networks) for each task while keeping old columns frozen.

```
Task 1: Column 1 (frozen)
Task 2: Column 1 (frozen) → Column 2 (new)      ← NEW COLUMN ADDED
Task 3: Column 1,2 (frozen) → Column 3 (new)   ← ANOTHER COLUMN ADDED
```

**Architecture Changes**: ✅ YES - Adds new full columns per task

**Pros**: 
- Zero forgetting (old columns never change)
- Can leverage knowledge via lateral connections

**Cons**: 
- Network size grows linearly with number of tasks
- Expensive memory-wise
- Architecture explicitly changes

---

### 2. **PackNet** 🔒 *[Fixed Architecture]*
**Paper**: Mallya et al., "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning" (2018)

**Idea**: Use task-specific binary masks on weights. Each task uses different subset of weights.

**Process**:
1. Train Task 1 → Learn weights
2. Prune unimportant weights → Create mask
3. Train Task 2 → Use remaining weights (frozen Task 1 weights)
4. Repeat

**Architecture Changes**: ❌ NO - Same network, just masks different weights

**Key Point**: The network architecture (layers, sizes) stays the same. Only which weights are "active" changes via masks.

**Pros**:
- Fixed network capacity
- Efficient weight reuse
- Architecture doesn't grow

**Cons**:
- Requires pruning step
- Limited capacity per task

---

### 3. **Subspace Methods** 🔒 *[Fixed Architecture]*
**Idea**: Constrain updates to specific subspaces in parameter space.

**Example**: Each task's updates lie in a different subspace:
- Task 1 uses subspace S₁
- Task 2 uses subspace S₂ (orthogonal or near-orthogonal to S₁)

**Architecture Changes**: ❌ NO - Same architecture, just constrains WHERE updates happen

**Mathematical Intuition**:
- Instead of: `θ_new = θ_old + Δθ`
- Use: `θ_new = θ_old + Pₜ(Δθ)` where `Pₜ` projects to task-specific subspace

**Key Point**: Same network structure, but gradient updates are projected into task-specific subspaces.

---

### 4. **Shared-Private Architecture** 🔒 *[Mostly Fixed]* (Our Implementation)

**Idea**: 
- **Shared component**: Learns common features (frozen after first task)
- **Private components**: Task-specific heads (trained per task)

**Training Process**:
```
Task 1: Train [Shared Encoder + Head 1] together
Task 2: Freeze Encoder, Train [Head 2] only     ← Small head added
Task 3: Freeze Encoder, Train [Head 3] only     ← Small head added
...
```

**Architecture Changes**: ⚠️ MINIMAL - Only adds small task-specific heads (typically 2-3 layers, < 1% of total parameters)

**Key Distinction**:
- **Core encoder**: Fixed architecture, frozen after Task 1
- **Task heads**: Small additions (like adding a classifier head)
- **Not architectural expansion** like PNN (doesn't add new feature layers)

**Analogy**: 
- PNN: Building new floors on a skyscraper (major change)
- Shared-Private: Adding small rooms at the top (minor addition)

**Why This Works**:
- Shared encoder captures common patterns (edges, shapes, etc.)
- Task heads specialize for specific classes
- No interference because encoder never changes after Task 1

---

## 💡 Why Representation-Based Methods Work

### 1. **Transfer Learning Principle**
Shared representations capture general features (e.g., edges, textures) useful across tasks.

### 2. **Isolation of Updates**
Task-specific components are isolated, so learning Task N doesn't affect Tasks 1...N-1.

### 3. **Capacity Management**
- Shared encoder: Fixed size (doesn't grow)
- Task heads: Small, efficient
- Better than growing network size linearly

### 4. **Natural Feature Hierarchy**
Deep networks naturally learn hierarchical features:
- Early layers: General (edges, textures)
- Later layers: Task-specific (class discriminators)

Representation-based methods exploit this!

---

## 🔬 Mathematical Foundation

### Shared-Private Model Formulation

For task `t`:

```
h_t = f_shared(x; θ_shared)     # Shared encoder (frozen after t=1)
y_t = f_private(h_t; θ_t)       # Task-specific head
```

**Loss for Task 1**:
```
L₁ = L(f_private(f_shared(x; θ_shared); θ₁), y₁)
```
→ Updates both `θ_shared` and `θ₁`

**Loss for Task t > 1**:
```
L_t = L(f_private(f_shared(x; θ_shared_FROZEN); θ_t), y_t)
```
→ Updates only `θ_t`

### Why No Forgetting?

For Task 1 after training Task 2:
```
ŷ₁ = f_private₁(f_shared_FROZEN(x); θ₁_FIXED)
```

Since both `f_shared` and `θ₁` are frozen, Task 1 performance remains unchanged! ✨

---

## 📊 Comparison Table: Architecture Modification

| Method | Architecture Changes? | What Changes? | Memory Impact |
|--------|----------------------|---------------|---------------|
| **Progressive Neural Networks** | ✅ **YES** | Adds full new columns per task | Linear growth |
| **Shared-Private** | ⚠️ **MINIMAL** | Adds small heads (~1% params) | Constant |
| **PackNet** | ❌ **NO** | Same architecture, masks weights | Constant |
| **Subspace Methods** | ❌ **NO** | Same architecture, projects updates | Constant |
| **LoRA** | ⚠️ **MINIMAL** | Adds small adapter modules | Constant |

## 📊 Comparison with Other Methods

| Method | Memory Growth | Forgetting | Transfer | Complexity | Architecture Modifies? |
|--------|---------------|------------|----------|------------|----------------------|
| **Finetuning** | Constant | High | Low | Low | ❌ No |
| **EWC** | Constant | Medium | Medium | Medium | ❌ No |
| **ER** | Grows (buffer) | Low | High | Medium | ❌ No |
| **LwF** | Constant | Medium | Medium | Medium | ❌ No |
| **PNN** | Linear | None | High | High | ✅ **YES** |
| **PackNet** | Constant | Low | Medium | High | ❌ No |
| **Rep-Based (Shared-Private)** | Constant | Low | High | Low | ⚠️ **MINIMAL** |

---

## 🎨 Visual Comparison: Architecture Changes

### Progressive Neural Networks (PNN) - Architecture Grows
```
Task 1:  [Column 1] ────────────→ Output 1
         (frozen)

Task 2:  [Column 1] ────────────→ Output 1
         (frozen)
         [Column 2] ────────────→ Output 2  ← NEW COLUMN!
         (active)

Task 3:  [Column 1] ────────────→ Output 1
         [Column 2] ────────────→ Output 2
         [Column 3] ────────────→ Output 3  ← ANOTHER NEW COLUMN!
```
**Architecture Size**: Grows linearly (3x original after 3 tasks)

---

### Shared-Private (Our Implementation) - Minimal Addition
```
Task 1:  [Shared Encoder] ────→ [Head 1] ──→ Output 1
         (512→256 params)       (256→128→2)

Task 2:  [Shared Encoder] ────→ [Head 1] ──→ Output 1 (frozen)
         (FROZEN)              [Head 2] ──→ Output 2  ← Small head only
                                (256→128→2, ~0.5K params)

Task 3:  [Shared Encoder] ────→ [Head 1] ──→ Output 1 (frozen)
         (FROZEN)              [Head 2] ──→ Output 2 (frozen)
                                [Head 3] ──→ Output 3  ← Small head only
```
**Architecture Size**: ~0.5K new params per task (vs. 100K+ for new column in PNN)

---

### PackNet - No Architecture Change
```
Task 1:  [Network] ────→ Output 1
         All weights trainable

Task 2:  [Network] ────→ Output 2
         Same architecture!
         But weights 1-50% are MASKED (frozen for Task 1)
         Only weights 51-100% trainable for Task 2
```
**Architecture Size**: Constant - same network, different active weights

---

## 🎓 Key Takeaways

1. **NOT all representation-based methods modify architecture**
   - **PNN**: Yes, adds full columns
   - **PackNet/Subspace**: No, same architecture
   - **Shared-Private**: Minimal (tiny heads only)

2. **Focus on Representations, Not Weights**
   - Protect learned feature spaces
   - Reuse across tasks

3. **Separation of Concerns**
   - Shared: Common features
   - Private: Task-specific decisions

4. **Freezing Strategy**
   - Learn shared features early
   - Freeze to protect them
   - Only learn task-specific parts later

5. **Natural Architecture Fit**
   - Exploits natural feature hierarchy in deep networks
   - Aligns with how CNNs learn (general → specific)

---

## 🚀 When to Use Representation-Based Methods

### ✅ Best For:
- Tasks with **similar domains** (e.g., all image classification)
- Tasks that can **share low-level features**
- Scenarios where you want **guaranteed no forgetting** (with proper freezing)

### ❌ Less Suitable For:
- Tasks from **very different domains** (images → text)
- Tasks requiring **very different architectures**
- Scenarios where **shared features are minimal**

---

## 🔍 Advanced Topics

### 1. **When to Freeze?**
- **Option 1**: Freeze after first task (simplest)
- **Option 2**: Freeze after first K tasks
- **Option 3**: Gradually freeze layers (bottom-up)

### 2. **Shared Representation Size**
- Too small: May not capture enough
- Too large: Overfitting risk
- Rule of thumb: 256-512 dimensions often works well

### 3. **Multi-Head vs. Single Head**
- **Multi-head**: Separate head per task (what we did)
- **Single head**: One head for all tasks (requires more careful training)

### 4. **Hybrid Approaches**
- Combine with EWC on shared encoder
- Use distillation on representations
- Add experience replay for shared encoder training

---

## 📚 Further Reading

1. **Progressive Neural Networks**: Rusu et al., 2016
2. **PackNet**: Mallya et al., 2018
3. **Subspace Regularization**: Deng et al., 2021
4. **Avalanche Framework**: Continual learning library with many implementations

---

## 🎯 Summary

Representation-Based Approaches offer a powerful paradigm for continual learning by:
- **Learning reusable features** rather than task-specific weights
- **Isolating task-specific components** to prevent interference
- **Exploiting natural feature hierarchies** in deep networks

The key insight: **Protect the representations, not the weights!**

