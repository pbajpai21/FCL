# LoRA (Low-Rank Adaptation) for Continual Learning

## 📚 Overview

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique that was originally developed for adapting large language models (like GPT) to new tasks. However, it's also highly effective for continual learning!

**Key Idea**: Instead of updating all weights, add small "adapter" matrices that can be efficiently learned and swapped.

---

## 🎯 Core Concept

### The Problem It Solves

When you have a large model with millions of parameters:
- **Full fine-tuning**: Update all weights → Expensive, can cause catastrophic forgetting
- **Freeze everything**: No learning → Can't adapt to new tasks
- **LoRA**: Add tiny learnable adapters → Best of both worlds!

### The Mathematical Foundation

LoRA is based on a fundamental insight: **weight updates in neural networks often have low intrinsic rank**.

#### What Does "Low Rank" Mean?

In linear algebra, a matrix has "low rank" if it can be decomposed into smaller matrices:

```
Large matrix W (d × d) ≈ Small matrix A (d × r) × Small matrix B (r × d)
                        where r << d
```

**Example**:
- Original weight matrix: `W` = 1000 × 1000 = **1M parameters**
- Low-rank decomposition: `A` = 1000 × 10 + `B` = 10 × 1000 = **20K parameters**
- Reduction: 50x fewer parameters!

---

## 🔬 How LoRA Works

### Standard Training (Without LoRA)

```
Input → W (full weight matrix) → Output
       ↑
    All weights updated during training
```

### With LoRA

```
Input → W (frozen) → + → Output
       ↑            ↑
    Frozen      LoRA adapter (A × B)
                (small, trainable)
```

**The Forward Pass**:
```
h = Wx + ΔWx
  = Wx + (BA)x
  = Wx + B(Ax)
```

Where:
- `W`: Original frozen weight matrix
- `ΔW = BA`: Low-rank adaptation (product of two small matrices)
- `A`: Down-projection matrix (d × r), trainable
- `B`: Up-projection matrix (r × d), trainable
- `r`: Rank (typically 1-16, very small!)

---

## 🎨 Visual Understanding

### Standard Fine-Tuning:
```
Task 1: [Full Model] ──→ Output 1
        All 10M params trainable

Task 2: [Full Model] ──→ Output 2  
        All 10M params trainable (overwrites Task 1!)
```
❌ Problem: Task 1 knowledge overwritten

### LoRA Approach:
```
Task 1: [Base Model (frozen)] + [LoRA₁ (0.1M params)] ──→ Output 1
        (10M params)         (tiny adapter)

Task 2: [Base Model (frozen)] + [LoRA₂ (0.1M params)] ──→ Output 2
        (10M params)         (different adapter, Task 1 adapter preserved)
```
✅ Solution: Base model frozen, only small adapters learned per task!

---

## 🔍 Detailed Mechanism

### Step-by-Step:

1. **Initialize Base Model**
   - Train or load a pretrained model
   - Freeze all original weights `W`

2. **Add LoRA Adapters**
   - For each linear layer, add two small matrices:
     - `A` (d × r): Random initialization
     - `B` (r × d): Zero initialization (so `BA = 0` initially)
   - Only `A` and `B` are trainable

3. **Forward Pass with LoRA**
   ```
   Original: h = Wx
   With LoRA: h = Wx + BAx
                ↑    ↑
            frozen  trainable
   ```

4. **Training**
   - Only update `A` and `B` matrices
   - Original weights `W` remain frozen

5. **Task Switching**
   - Task 1: Use `W + B₁A₁`
   - Task 2: Use `W + B₂A₂`
   - Can store/load adapters per task!

---

## 💡 Why LoRA Works for Continual Learning

### 1. **Parameter Efficiency**
- Only train ~0.1-1% of parameters
- Can store many task adapters
- Fast training

### 2. **Isolation**
- Each task gets its own adapter
- Tasks don't interfere (base model frozen)
- Can switch adapters at inference time

### 3. **Low-Rank Hypothesis**
- Weight updates often lie in low-dimensional subspaces
- Small adapters capture task-specific changes efficiently

### 4. **Flexibility**
- Can apply to any layer
- Can apply to all layers or just some
- Tunable rank `r` for efficiency/performance tradeoff

---

## 📊 Comparison with Other Methods

| Method | Params per Task | Memory | Forgetting | Flexibility |
|--------|----------------|--------|------------|-------------|
| **Full Fine-tuning** | All (10M) | Low | High | High |
| **Freeze Everything** | 0 | Low | None | None |
| **Shared-Private** | Small heads (~1K) | Low | Low | Medium |
| **PNN** | Full column (~10M) | High | None | High |
| **LoRA** | Tiny adapters (~100K) | Low | Low | High |

### Key Advantages of LoRA:

✅ **Parameter Efficient**: Only train tiny adapters
✅ **No Forgetting**: Base model never changes
✅ **Scalable**: Store many adapters (one per task)
✅ **Fast**: Small adapters train quickly
✅ **Flexible**: Can apply selectively to layers

---

## 🎯 When to Use LoRA

### ✅ Best For:
1. **Large Models**: When full fine-tuning is expensive
2. **Many Tasks**: When you'll learn many tasks sequentially
3. **Parameter Constraints**: Limited memory/storage
4. **Quick Adaptation**: Need fast task switching
5. **Transfer Learning**: Strong base model available

### ❌ Less Suitable For:
1. **Very Small Models**: Overhead not worth it
2. **Dramatically Different Tasks**: Base model may be insufficient
3. **Tasks Requiring Major Changes**: Low-rank may be too restrictive

---

## 🔬 Mathematical Deep Dive

### Low-Rank Decomposition Theorem

**Given**: A weight update matrix `ΔW` of size (d × d)

**Low-Rank Approximation**:
```
ΔW ≈ BA

Where:
- B: (d × r) matrix
- A: (r × d) matrix  
- r << d (rank, typically 1-16)
```

**Parameter Count**:
- Original `ΔW`: d² parameters
- LoRA `BA`: 2dr parameters
- **Reduction factor**: d² / 2dr = d / 2r

**Example**: d=1000, r=10
- Original: 1,000,000 parameters
- LoRA: 20,000 parameters
- **50x reduction!**

### Why Zero Initialize B?

```python
# Standard LoRA initialization
A = random_normal(mean=0, std=1/r)  # Random
B = zeros()                          # Zeros!
```

**Reason**: So that `BA = 0` initially:
- Model starts identical to base model
- No disruption from random initialization
- Clean task adaptation

### Gradient Flow

During backpropagation:
```
Loss → Output → (Wx + BAx) → Input
                ↑    ↑
              frozen trainable
```

Only `A` and `B` receive gradients, `W` does not!

---

## 🏗️ Architecture Design Choices

### 1. **Which Layers to Apply LoRA?**

**Option A: All Linear Layers**
- Most comprehensive
- Higher parameter count
- Best performance potentially

**Option B: Only Key Layers**
- Attention layers (in transformers)
- Final classification layers
- More efficient

**Option C: Task-Specific Choice**
- Analyze which layers matter most
- Apply LoRA selectively

### 2. **Rank Selection (r)**

| Rank (r) | Params | Performance | Speed |
|----------|--------|-------------|-------|
| 1 | Minimal | May be limiting | Fastest |
| 4-8 | Low | Good balance | Fast |
| 16 | Moderate | High | Medium |
| 32+ | High | Very high | Slower |

**Rule of thumb**: Start with r=8, adjust based on task complexity

### 3. **Alpha Parameter (α)**

LoRA often includes a scaling factor:
```
h = Wx + (α/r) · BAx
```

Where `α` is a hyperparameter (typically = r or 2r)
- Controls adapter influence
- Helps with numerical stability

---

## 🔄 LoRA in Continual Learning Workflow

### Training Phase:

```
Task 1:
1. Base model W (frozen)
2. Create LoRA₁: A₁, B₁
3. Train: Only update A₁, B₁
4. Save LoRA₁

Task 2:
1. Base model W (same, frozen)
2. Create LoRA₂: A₂, B₂ (new adapters)
3. Train: Only update A₂, B₂
4. Save LoRA₂
5. LoRA₁ preserved!

Task 3:
... (repeat)
```

### Inference Phase:

```
For Task 1: Use W + B₁A₁
For Task 2: Use W + B₂A₂
For Task 3: Use W + B₃A₃
...
```

Can switch adapters dynamically!

---

## 📈 Advantages for Continual Learning

### 1. **Zero Forgetting Guarantee**
- Base model `W` never changes
- Old adapters preserved
- Perfect isolation

### 2. **Efficiency**
- Small adapters = fast training
- Minimal memory overhead
- Scalable to many tasks

### 3. **Modularity**
- Adapters are independent modules
- Easy to add/remove tasks
- Can compose adapters (multi-task)

### 4. **Transfer Learning Friendly**
- Start with strong pretrained base
- Only adapt what's needed
- Leverages pretrained knowledge

---

## ⚠️ Limitations & Considerations

### 1. **Low-Rank Assumption**
- Assumes updates are low-rank
- May not hold for all tasks
- Some tasks need higher rank

### 2. **Base Model Dependency**
- Quality depends on base model
- If base is poor, adapters can't fix it
- May need better base for diverse tasks

### 3. **Rank Selection**
- Need to tune rank `r`
- Too small: insufficient capacity
- Too large: defeats efficiency purpose

### 4. **Layer Selection**
- Need to decide which layers to adapt
- Not always obvious
- Some experimentation needed

---

## 🎓 Key Insights

1. **Low-Rank Hypothesis**: Most weight updates lie in low-dimensional subspaces
2. **Parameter Efficiency**: Train tiny adapters instead of full model
3. **Task Isolation**: Each task gets independent adapter
4. **Scalability**: Can handle many tasks efficiently
5. **Practical**: Works well in practice, especially for large models

---

## 🔗 Relationship to Other Methods

### LoRA vs. Shared-Private:
- **Shared-Private**: Adds heads, shares encoder
- **LoRA**: Adds adapters to existing layers
- **Both**: Parameter-efficient, representation-based

### LoRA vs. EWC:
- **EWC**: Regularizes weight changes
- **LoRA**: Avoids weight changes entirely (adapter approach)
- **LoRA**: More isolation, less interference

### LoRA vs. PackNet:
- **PackNet**: Masks weights in same architecture
- **LoRA**: Adds adapters, base weights unchanged
- **Both**: Fixed architecture, parameter-efficient

---

## 📚 Further Reading

1. **Original LoRA Paper**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
2. **QLoRA**: Quantized LoRA for even more efficiency
3. **AdaLoRA**: Adaptive rank selection
4. **DoRA**: Decomposed LoRA for better performance

---

## 🎯 Summary

**LoRA** is a powerful technique that:
- Adds small trainable adapters to frozen base model
- Reduces parameters by 10-100x
- Provides perfect task isolation
- Scales efficiently to many tasks
- Works especially well with large pretrained models

**Key Formula**: `output = base_model(x) + adapter(x) = Wx + BAx`

**Philosophy**: "Don't change the base model, just add tiny task-specific tweaks!"

