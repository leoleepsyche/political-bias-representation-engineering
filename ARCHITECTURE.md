# Technical Architecture

## System Design Overview

```
┌─────────────────────────────────────────────────────────┐
│              Input: Political Datasets                  │
│  ┌───────────────┬────────────────┬──────────────────┐  │
│  │ Political     │  Non-Political │  Style Variants  │  │
│  │ Statements    │  Statements    │  (Academic/     │  │
│  │ (L/R pairs)   │  (20 topics)   │   Activist)     │  │
│  └───────────────┴────────────────┴──────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   Model Inference + Hooks   │
        │   (Qwen2.5-7B-Instruct)     │
        │ Extract hidden states per   │
        │ layer (32 layers total)     │
        └──────────────┬──────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
┌─────────┐      ┌──────────┐      ┌──────────┐
│ Step 1  │      │ Step 2   │      │ Step 3   │
│Locate   │ ───► │ Analyze  │ ───► │ Topic    │
│Political│      │L/R/N Bias│      │Analysis  │
│Layers   │      │          │      │          │
└────┬────┘      └────┬─────┘      └────┬─────┘
     │                │                  │
     ▼                ▼                  ▼
 [P-Layers]   [Dir. Vectors]      [Topic Scores]
     │                │                  │
     └────────────────┼──────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │  Step 4:     │
              │  Steering    │
              │  Intervention│
              └──────┬───────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
    ┌────────┐  ┌────────┐  ┌─────────┐
    │ Repr   │  │Behav   │  │Capab    │
    │Eval    │  │Eval    │  │Eval     │
    │(Gap→0) │  │(Stance)│  │(QA)     │
    └────────┘  └────────┘  └─────────┘
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   Results & Plots    │
          │ - Visualizations     │
          │ - Bias scores        │
          │ - Optimal steering α │
          └──────────────────────┘
```

---

## Detailed Component Architecture

### 1. Data Processing Layer

```python
# Input standardization
political_dataset.py
  ├─ get_left_statements()       # 20 (topic, statement) pairs
  ├─ get_right_statements()      # 20 (topic, statement) pairs
  └─ get_prompt_template(stmt)   # Standardized prompt format

nonpolitical_dataset.py
  ├─ get_nonpolitical_statements()      # 20 factual statements
  └─ get_political_statements_mixed()   # Mixed for probing

control_dataset.py
  ├─ lexical_neutral_pairs()            # Control: vocabulary only
  ├─ topic_shuffled_pairs()             # Control: cross-topic
  └─ base_model_statements()            # Control: uninstruct model
```

**Design Pattern**: Immutable datasets (functions return lists of tuples)

**Why**: Ensures reproducibility, easy versioning, supports multiple languages/topics

---

### 2. Model Interface Layer

```python
run_experiment.py

load_model_and_tokenizer(model_name, quantize=False, device="auto")
  ├─ Detects device (auto: CUDA > MPS > CPU)
  ├─ Handles 4-bit quantization (if requested)
  ├─ Returns: (model, tokenizer, device, num_layers)
  └─ Configurable: model_name, quantization, device

extract_hidden_states(model, tokenizer, prompt, device)
  ├─ Input: prompt (string)
  ├─ Tokenize → forward pass → extract last token's hidden state
  ├─ Returns: dict {layer_idx: hidden_state_tensor}
  └─ Uses MPS/CUDA acceleration when available

cosine_similarity(vec1, vec2)
  ├─ Input: two hidden state tensors
  ├─ Computation: cos(θ) = (a·b) / (||a|| ||b||)
  └─ Returns: scalar in [-1, 1]
```

**Design Pattern**: Stateless functions (no side effects)

**Why**: Composable, testable, parallelizable

---

### 3. Analysis Pipeline

#### Step 1: Political Layer Localization

```python
step1_locate_political_layers.py

cosine_gap_analysis()
  ├─ Extract hidden states: Political, Non-Political
  ├─ 500 random pairings:
  │   ├─ P-P (political-political)
  │   ├─ NP-NP (non-political-non-political)
  │   └─ P-NP (cross-category)
  ├─ Compute per-layer cosine similarities
  └─ Angular gap = gap(P-NP) - gap(P-P, NP-NP)

weak_classifier_probing()
  ├─ SVM (linear kernel) per layer
  │   └─ Classify: Political vs Non-Political
  ├─ MLP (1 hidden layer, 100 neurons) per layer
  │   └─ Cross-validate (5-fold)
  └─ Find layer where accuracy exceeds 90%

estimate_political_layers()
  ├─ Combine two signals:
  │   ├─ Angular gap onset
  │   └─ Classifier 90% threshold
  ├─ Lower bound: min(onset_layer, classifier_layer)
  ├─ Upper bound: where gap falls to 50% of peak
  └─ Output: [lower, upper] layer indices
```

**Key Metric**: `angular_gap[layer] = arccos(sim_P-NP) - mean(arccos(sim_P-P), arccos(sim_NP-NP))`

---

#### Step 2: L/R/N Bias Analysis

```python
step2_analyze_bias.py

triangulation_in_political_layers()
  ├─ Extract: Left, Right, Neutral hidden states
  ├─ 6 pair types (500 rounds):
  │   ├─ NN, LL, RR (within-class)
  │   └─ NL, NR, LR (cross-class, same-topic)
  ├─ Compute per-layer angles
  └─ Core metrics:
      ├─ nl_gap[layer] = angle(N-L) - angle(N-N)
      ├─ nr_gap[layer] = angle(N-R) - angle(N-N)
      └─ bias_direction = nl_gap - nr_gap

three_class_probing()
  ├─ Train L vs R vs N classifier per layer
  ├─ SVM (linear) + MLP (100 neurons)
  ├─ Cross-validate within [pol_lower, pol_upper]
  └─ Compare: pol_acc vs nonpol_acc

compute_political_direction_vector()
  ├─ Within political layers:
  │   ├─ left_mean = mean(left hidden states)
  │   ├─ right_mean = mean(right hidden states)
  │   └─ direction = right_mean - left_mean
  ├─ Project neutral onto L-R axis
  └─ Per-layer direction vectors for Step 4
```

**Key Insight**: Neutral's projection on L-R axis reveals default leaning

---

#### Step 3: Topic-Level Analysis

```python
step3_topic_analysis.py

topic_layer_heatmap()
  ├─ For each (topic, layer):
  │   ├─ Compute L-R gap (per-topic)
  │   ├─ Compute bias_direction (nl_gap - nr_gap)
  │   └─ Store in matrix [topics × layers]
  ├─ Heatmap visualization
  └─ Rank topics by political strength

content_style_decomposition()
  ├─ For each topic, 4 variants:
  │   ├─ left_academic, left_activist
  │   ├─ right_academic, right_activist
  ├─ Gaps:
  │   ├─ Content gap = L-vs-R (same style)
  │   ├─ Style gap = academic-vs-activist (same stance)
  └─ Ratio = content_gap / style_gap
```

**Key Hypothesis**: content_gap >> style_gap → we measure stance, not tone

---

#### Step 4: Steering Intervention

```python
step4_steering.py

PoliticalSteeringHook (forward hook)
  ├─ Registered on layers [pol_lower, pol_upper]
  ├─ On forward pass:
  │   └─ hidden_new = hidden_old - alpha * direction_vector
  ├─ Alpha: steering strength
  │   ├─ 0: no steering (baseline)
  │   ├─ 1: moderate steering
  │   ├─ >2: strong steering (may degrade capability)
  └─ CAA method (Contrastive Activation Addition)

evaluate_representation()
  ├─ For each alpha:
  │   ├─ Compute bias_direction with steering
  │   └─ Track: abs(bias_direction) → 0?
  └─ Plot: bias curves for alpha = 0, 0.5, 1.0, 2.0, 3.0, 5.0

evaluate_behavioral()
  ├─ Generation with steering:
  │   ├─ 5 political prompts (healthcare, climate, etc.)
  │   ├─ Generate text at each alpha
  │   └─ Qualitatively assess stance changes
  └─ Human review recommended

evaluate_capability()
  ├─ Simple QA tasks:
  │   ├─ Math: "17 × 23 = ?"
  │   ├─ Science: "Chemical formula of water?"
  │   ├─ Geography, History, Logic
  ├─ Measure: % correct at each alpha
  └─ Guard: alpha where capability drops >10%
```

**Metric**: Bias-Capability Score = 0.6 × capability + 0.4 × bias_reduction

---

## Data Flow

### Example: One "Round" of Analysis

```
Input prompt: "Government should provide universal healthcare"
    │
    ├─ Tokenize: ['Government', 'should', ..., '</s>']
    ├─ Forward pass through model (32 layers)
    ├─ Extract hidden states:
    │   ├─ Layer 0: shape (1, seq_len, 4096)
    │   ├─ Layer 15: shape (1, seq_len, 4096)
    │   └─ Layer 31: shape (1, seq_len, 4096)
    ├─ Last token hidden state: shape (4096,)
    │   └─ At each layer: hidden_states[layer_idx, -1, :]
    └─ Output: {0: tensor(...), 1: tensor(...), ..., 31: tensor(...)}

Per-layer computation:
    hidden_left[15]  shape (4096,)
    hidden_right[15] shape (4096,)
    cos_sim = dot(hidden_left, hidden_right) / (norm_l * norm_r)
    angle = arccos(cos_sim)  in degrees
```

---

## Computational Complexity

| Step | Time | Memory | Notes |
|------|------|--------|-------|
| Extract 1 prompt | O(layers × seq_len × hidden_dim) | ~2GB | Qwen-7B, full precision |
| Step 1 (500 rounds) | 30-60 min | 6-12GB | Includes SVM/MLP training |
| Step 2 (500 rounds) | 20-40 min | 8-16GB | Direction vector computation |
| Step 3 (4 variants) | 20-40 min | 8-16GB | Heatmap + decomposition |
| Step 4 (5 alphas) | 30-60 min | 4-8GB | Generation + QA inference |
| **Total** | **2-4 hours** | **8-16GB** | **M1 Mac, T4 GPU** |

---

## Extensibility Points

### Adding New Datasets
```python
# New file: political_dataset_multilingual.py
def get_french_statements():
    return [
        ("santé", ("Le gouvernement devrait...", "Les marchés privés offrent...")),
        ...
    ]
```

### Adding New Metrics
```python
# In step4_steering.py or new file
def evaluate_entailment(model, tokenizer, device):
    # Measure: does neutral text entail L and R equally?
    ...
```

### Adding New Steering Methods
```python
class DirectionalSteeringHook:
    # Subtract direction only in specific layers
    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if layer_idx in self.target_layers:
                output = output - self.alpha * self.direction
        return hook_fn
```

---

## Testing Strategy

```python
# tests/test_run_experiment.py
def test_cosine_similarity():
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([1.0, 0.0, 0.0])
    assert cosine_similarity(v1, v2) == pytest.approx(1.0)

def test_extract_hidden_states():
    model, tokenizer, device, _ = load_model_and_tokenizer("gpt2")
    states = extract_hidden_states(model, tokenizer, "Hello", device)
    assert len(states) == 12  # GPT2 has 12 layers

def test_angular_gap():
    # Ensure angular gap is positive for cross-category
    ...
```

---

## Performance Optimization

### GPU Acceleration
```python
# Automatic mixed precision (faster)
with torch.cuda.amp.autocast():
    hidden_states = model(...)

# Batch processing
for topic in chunked_topics(chunk_size=4):
    states = parallel_extract(topic)
```

### Memory Optimization
```python
# Gradient checkpointing (slower, less memory)
model.gradient_checkpointing_enable()

# Delete intermediate results
del hidden_states_per_round
torch.cuda.empty_cache()
```

---

<div align="center">

**Architecture designed for clarity, extensibility, and reproducibility.**

</div>
