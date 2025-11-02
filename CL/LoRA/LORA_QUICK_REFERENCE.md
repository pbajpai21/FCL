# LoRA Quick Reference

## 🎯 One-Sentence Summary

**LoRA adds tiny trainable adapter matrices to a frozen base model, allowing task-specific adaptation with minimal parameters.**

---

## 📐 The Math in Simple Terms

### Standard Training:
```
h = W × x
↑  ↑   ↑
│  │   └─── Input
│  └─────── Weight matrix (trainable, large)
└────────── Output
```

### With LoRA:
```
h = W × x + (B × A) × x
  ↑   ↑     ↑   ↑    ↑
  │   │     │   │    └─── Input
  │   │     │   └──────── Small adapter matrix A
  │   │     └──────────── Small adapter matrix B
  │   └────────────────── Frozen weight matrix
  └────────────────────── Output
```

**Key**: `B × A` is much smaller than `W`!

---

## 🔢 Concrete Example

### Scenario: Linear Layer with 1000 inputs, 1000 outputs

**Standard Approach:**
- Weight matrix `W`: 1000 × 1000 = **1,000,000 parameters**
- All parameters trainable

**LoRA Approach:**
- Base weight `W`: 1000 × 1000 = **1,000,000 parameters** (frozen)
- Adapter `A`: 1000 × 10 = **10,000 parameters** (trainable)
- Adapter `B`: 10 × 1000 = **10,000 parameters** (trainable)
- **Total trainable**: 20,000 parameters (50x reduction!)

**Memory Savings:**
- Standard: 1M params × 4 bytes = 4MB per task
- LoRA: 20K params × 4 bytes = 80KB per task
- **50x less memory per task!**

---

## 🎨 Visual Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    BASE MODEL (Frozen)                  │
│                                                          │
│   Input ───→ [Linear] ───→ [Linear] ───→ [Linear] ───→ Output │
│              (W₁)         (W₂)         (W₃)            │
│              ❄️ frozen    ❄️ frozen    ❄️ frozen       │
└─────────────────────────────────────────────────────────┘
                           +
┌─────────────────────────────────────────────────────────┐
│              LoRA ADAPTERS (Trainable)                    │
│                                                          │
│   Input ───→ [B₁A₁] ───→ [B₂A₂] ───→ [B₃A₃] ───→ Add    │
│              (tiny)      (tiny)      (tiny)            │
│              🔥 train    🔥 train    🔥 train           │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 Continual Learning Workflow

### Task 1:
```
1. Start with base model W (pretrained or trained on Task 1)
2. Freeze W
3. Add LoRA adapters: A₁, B₁
4. Train only A₁, B₁ on Task 1
5. Save adapters: LoRA₁ = {A₁, B₁}
```

### Task 2:
```
1. Use same base model W (still frozen)
2. Add NEW LoRA adapters: A₂, B₂
3. Train only A₂, B₂ on Task 2
4. Save adapters: LoRA₂ = {A₂, B₂}
5. LoRA₁ still exists and works!
```

### Inference:
```
Task 1: Forward pass with W + B₁A₁
Task 2: Forward pass with W + B₂A₂
Task 3: Forward pass with W + B₃A₃
...
```

---

## 📊 Parameter Comparison Table

| Method | Params (1000×1000 layer) | Trainable | Memory/Task |
|--------|-------------------------|-----------|-------------|
| **Full Fine-tune** | 1M | 1M | 4MB |
| **Freeze All** | 1M | 0 | 0MB (but can't learn) |
| **LoRA (r=4)** | 1M (frozen) + 8K | 8K | 32KB |
| **LoRA (r=8)** | 1M (frozen) + 16K | 16K | 64KB |
| **LoRA (r=16)** | 1M (frozen) + 32K | 32K | 128KB |

**Key Insight**: Even with r=16, LoRA uses **31x fewer trainable parameters!**

---

## 🎯 Key Hyperparameters

### 1. **Rank (r)**
- **Low (1-4)**: Maximum efficiency, may limit performance
- **Medium (8-16)**: Good balance (recommended starting point)
- **High (32+)**: Better performance, less efficient

### 2. **Alpha (α)**
- Scaling factor: `output = Wx + (α/r) × BAx`
- Typically: `α = r` or `α = 2r`
- Controls adapter influence strength

### 3. **Which Layers**
- **All layers**: Maximum adaptation
- **Attention only**: Efficient (for transformers)
- **Last layers only**: Task-specific adaptation

---

## 💡 Why It Works

### 1. **Low-Rank Hypothesis**
Research shows: Weight updates in fine-tuning often have **low intrinsic dimensionality**.

Example:
- You have 1M parameters
- But the "direction" of updates lies in ~10-dimensional space
- So you only need to learn 10 dimensions worth of changes!

### 2. **Efficient Representation**
Instead of:
```
ΔW = [1M values]
```

LoRA uses:
```
ΔW ≈ BA = [small A] × [small B] = [~20K values total]
```

### 3. **Task Isolation**
- Each task's adapter is independent
- No interference between tasks
- Can remove/add tasks easily

---

## ✅ Advantages

1. **Efficient**: Train 10-100x fewer parameters
2. **Fast**: Small adapters train quickly
3. **Isolated**: Tasks don't interfere
4. **Scalable**: Can handle many tasks
5. **Modular**: Easy to add/remove tasks

---

## ⚠️ Limitations

1. **Rank Selection**: Need to choose appropriate rank
2. **Base Model**: Quality depends on base model
3. **Low-Rank Assumption**: May not hold for all tasks
4. **Hyperparameter Tuning**: Need to tune r and α

---

## 🔗 Comparison with Other Methods

| Method | How It Works | When Tasks Differ |
|--------|--------------|-------------------|
| **EWC** | Regularize weight changes | Still updates all weights |
| **ER** | Replay old data | Uses memory buffer |
| **PNN** | Add full columns | Works but very expensive |
| **Shared-Private** | Freeze encoder, add heads | Assumes general features |
| **LoRA** | Add tiny adapters | Works efficiently |

---

## 🎓 Key Takeaway

**LoRA = "Surgical Updates"**

Instead of changing the entire model (expensive, risky), LoRA makes tiny targeted changes via small adapter matrices. It's like:
- **Full fine-tuning**: Renovating entire house
- **LoRA**: Just updating the door handles (much cheaper, faster, less disruptive!)

---

## 📝 Code Pseudocode

```python
# Forward pass with LoRA
def forward_with_lora(x, W, A, B, alpha=8, rank=8):
    # Base model output (frozen)
    base_output = W @ x
    
    # LoRA adapter output (trainable)
    adapter_output = (alpha / rank) * (B @ (A @ x))
    
    # Combined
    return base_output + adapter_output

# Training: Only update A and B, not W
optimizer = Adam([A, B])  # W not in optimizer!
```

---

## 🚀 Next Steps

1. Understand the low-rank decomposition concept
2. See how adapters are initialized (A random, B zero)
3. Learn about rank selection strategies
4. Implement LoRA for continual learning!

