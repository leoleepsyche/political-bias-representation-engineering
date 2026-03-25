# Political Bias Representation Engineering

<div align="center">

**A hierarchical approach to locating and steering political biases in LLMs using Representation Engineering**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[📖 Documentation](#documentation) • [🚀 Quick Start](#quick-start) • [📊 Results](#results) • [🤝 Contributing](#contributing)

</div>

---

## 📌 Overview

This research project investigates **political biases in Large Language Models (LLMs)** using **Representation Engineering** techniques. We adapt the Safety Layers methodology (ICLR 2025) to locate and analyze political bias layers, then develop steering interventions to reduce political skew while preserving model capabilities.

### Key Innovation: Hierarchical 4-Layer Framework

Instead of directly analyzing left/right bias, we:
1. **Layer 1**: Locate "political layers" (Political vs Non-Political classification)
2. **Layer 2**: Analyze Left/Right/Neutral bias *within* those layers
3. **Layer 3**: Perform topic-level fine-grained analysis
4. **Layer 4**: Apply steering interventions + evaluate (representation, behavioral, capability)

---

## 🎯 Research Goals

- ✅ Develop a **reproducible methodology** for detecting political biases in LLM representations
- ✅ Distinguish **political stance** from **lexical differences** and **expression style**
- ✅ Create **topic-specific bias profiles** (e.g., healthcare bias ≠ climate bias)
- ✅ Design **inference-time steering** to reduce political skew without harming capabilities
- ✅ Enable **multi-person collaboration** on political AI safety research

---

## 🏗️ Project Structure

```
political_bias_cosine_gap/
│
├── 📚 CORE EXPERIMENTS (4-Layer Framework)
│   ├── step1_locate_political_layers.py       # Locate political layers
│   ├── step2_analyze_bias.py                  # L/R/N triangulation within political layers
│   ├── step3_topic_analysis.py                # Topic-level fine-grained analysis
│   └── step4_steering.py                      # Steering intervention + evaluation
│
├── 📊 DATASETS
│   ├── political_dataset.py                   # 20 US political topics (L/R pairs)
│   ├── nonpolitical_dataset.py                # Same topics, factual statements (non-political)
│   └── control_dataset.py                     # Control experiments (lexical, shuffled, base model)
│
├── 🔬 BASELINE EXPERIMENTS
│   ├── run_experiment.py                      # Basic cosine gap method
│   ├── run_triangulation.py                   # N-L-R triangulation (published approach)
│   ├── run_enhanced.py                        # Bang et al. (ACL 2024) enhancements
│   ├── run_control_experiment.py              # Control experiments
│   └── run_quick_demo.py                      # Fast validation (1.5B model)
│
├── 📓 NOTEBOOKS & DOCS
│   ├── Colab_Notebook.ipynb                   # Self-contained Colab version
│   ├── README.md                              # This file
│   ├── GETTING_STARTED.md                     # Step-by-step guide
│   ├── PAPER_SUMMARY.md                       # Research methodology
│   └── ARCHITECTURE.md                        # Technical architecture
│
├── 🛠️ SETUP & CONFIG
│   ├── setup.sh                               # Environment setup
│   ├── requirements.txt                       # Python dependencies
│   └── .gitignore
│
├── 📈 RESULTS (generated during experiments)
│   ├── results/                               # Basic experiment results
│   ├── results_step1/                         # Political layer localization
│   ├── results_step2/                         # Bias analysis results
│   ├── results_step3/                         # Topic analysis results
│   ├── results_step4/                         # Steering evaluation results
│   ├── results_demo/                          # Demo run results
│   └── model_results/                         # Archived multi-model runs
│
└── 🧪 TESTS
    └── tests/
        └── test_run_experiment.py             # Unit tests
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **GPU/MPS** (NVIDIA CUDA, Apple Silicon, or CPU - slower)
- **16GB+ RAM** (for 7B model inference)

### 1. Environment Setup

```bash
# Clone and navigate to project
cd political_bias_cosine_gap

# Install dependencies
pip install -r requirements.txt

# Or use the setup script (Mac)
bash setup.sh
```

### 2. Run the Full 4-Layer Pipeline

```bash
# Step 1: Locate political layers (30-60 min on M1 Mac)
python step1_locate_political_layers.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num_rounds 500

# Step 2: Analyze L/R/N bias within those layers (20-40 min)
python step2_analyze_bias.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --step1_dir ./results_step1

# Step 3: Topic-level analysis with content vs style (20-40 min)
python step3_topic_analysis.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --step1_dir ./results_step1

# Step 4: Steering intervention + evaluation (30-60 min)
python step4_steering.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --step2_dir ./results_step2 \
  --alpha 0.5 1.0 2.0 3.0 5.0
```

### 3. Quick Demo (Fast Validation)

```bash
# Run with smaller model (1.5B, ~5 minutes)
python run_quick_demo.py

# Or use the Colab notebook
# Open: Colab_Notebook.ipynb in Google Colab
```

---

## 📊 Results & Interpretation

### Step 1: Political Layer Localization
**Output**: `results_step1/`
- `step1_cosine_gap.png` — Three cosine similarity curves showing political vs non-political patterns
- `step1_classifier_probing.png` — SVM/MLP accuracy per layer (90% threshold marks political onset)
- `step1_localization_summary.png` — Combined summary with political layer boundaries
- `political_layers.npz` — Quantitative results for downstream steps

**What it shows**:
- Which layers encode "political content" vs general knowledge
- Typically: layers 15-25 (out of 32) for 7B models
- Early layers: task/language features; Middle: political content; Late: output generation

---

### Step 2: L/R/N Bias Analysis
**Output**: `results_step2/`
- `step2_bias_direction.png` — Bias direction indicator (positive = right lean, negative = left lean)
- `step2_three_class_probing.png` — L vs R vs N classification accuracy
- `step2_neutral_projection.png` — Where does "neutral" sit on the L-R spectrum?
- `direction_vectors.npz` — Political direction vector for steering

**What it shows**:
- Model's implicit political leaning (if any)
- Whether L/R/N are clearly distinguished in representation space
- Bias magnitude and direction across layers

---

### Step 3: Topic-Level Analysis
**Output**: `results_step3/`
- `step3_topic_layer_heatmap.png` — Which topics are most "political" in which layers?
- `step3_topic_bias_direction.png` — Per-topic political leaning
- `step3_content_vs_style.png` — Does the gap encode stance or just expression style?

**What it shows**:
- Some topics (e.g., abortion) have stronger political signals than others
- Content bias is stronger than style bias (validating stance measurement)
- Topic-specific interventions are possible

---

### Step 4: Steering Evaluation
**Output**: `results_step4/`
- `step4_representation_steering.png` — Bias direction changes with different steering strengths (α)
- `step4_bias_capability_tradeoff.png` — Pareto frontier: bias reduction vs capability preservation
- `behavioral_generations.json` — Example generations at different α values
- `step4_results.json` — Numerical results

**What it shows**:
- Steering strength (α) vs bias reduction curve
- Optimal α that balances bias reduction and capability
- Generated text changes with steering (qualitative)
- Capability scores on factual QA (no degradation with light steering)

### Archived Multi-Model Runs

Archived runs for Qwen, Llama, ChatGLM, and Mistral now live in [`model_results/`](model_results/README.md).

Current multi-model evidence suggests that political information is broadly distributed across the network rather than confined to a tiny block of layers. In practice, the detected `political layers` should be interpreted as politically discriminative layer ranges, not exclusive political-only modules.

---

## 🔬 Methodology

### Adapted from Literature

| Component | Based On | Adaptation |
|-----------|----------|-----------|
| Cosine Gap Method | Safety Layers (ICLR 2025) | Normal/Malicious → Political/Non-Political |
| Weak Classifier Probing | Zhou et al. (EMNLP 2024) | Applied per-layer to political classification |
| Neutral Anchor | Bang et al. (ACL 2024) | Triangulation: N-L, N-R gaps as bias indicators |
| Content vs Style | Bang et al. (ACL 2024) | Academic vs Activist tone decomposition |
| Steering (CAA) | Zou et al. (RepE) | Political direction vector subtraction at inference |

### Key Hypotheses

1. **Political information is localized** in specific transformer layers (not distributed)
2. **Neutral is distinct from L/R** in representation space (not ambiguous)
3. **Content gap > Style gap** (political stance is encoded, not just tone)
4. **Steering is low-rank** (a single direction vector suffices across topics)

---

## 📈 Expected Results (Qwen2.5-7B-Instruct)

| Metric | Value | Meaning |
|--------|-------|---------|
| Political layers | [15, 25] | Layers 15-25 encode political content |
| Bias direction | −0.5° to +1.5° | Slight right lean (topic-dependent) |
| Content/Style ratio | 2.0-3.0x | Gap mainly encodes stance, not style |
| Steering optimal α | 1.0-2.0 | Modest steering works best |
| Capability preservation | >90% | QA performance maintained |

---

## 🤝 Contributing

We welcome contributions! Here's how to collaborate:

### 1. Fork & Clone
```bash
git clone https://github.com/YOUR_USERNAME/political-bias-representation-engineering.git
cd political-bias-representation-engineering
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# Examples:
# - feature/new-model-qwen-32b
# - feature/multilingual-topics
# - feature/streaming-generation
# - bugfix/steering-hook-cuda
```

### 3. Make Changes & Test
```bash
# Make your changes
python step1_locate_political_layers.py --model YOUR_MODEL

# Run tests
pytest tests/

# Commit
git add .
git commit -m "Add: description of your change"
```

### 4. Push & Create Pull Request
```bash
git push origin feature/your-feature-name
# Then create PR on GitHub with:
# - Description of changes
# - Results/visualizations if applicable
# - Tests added/updated
```

### Areas for Contribution

- [ ] **New models**: Test on Llama, Mistral, GPT2-XL, etc.
- [ ] **Multilingual**: Extend to non-English political topics
- [ ] **Finer topics**: Add more granular political issues (16 → 50+ topics)
- [ ] **Streaming**: Support streaming generation with steering
- [ ] **Probing tasks**: Add more downstream tasks (sentiment, entailment, etc.)
- [ ] **Visualization**: Interactive dashboards for bias exploration
- [ ] **Documentation**: API docs, tutorials, papers

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Step-by-step first-time setup |
| [PAPER_SUMMARY.md](PAPER_SUMMARY.md) | Detailed research methodology |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Technical architecture & design |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues & solutions |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Test a specific step
pytest tests/test_run_experiment.py::test_cosine_similarity -v

# Test with coverage
pytest tests/ --cov=. --cov-report=html
```

---

## 💾 Reproducing Results

### Exactly Reproduce Our Results
```bash
python step1_locate_political_layers.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --seed 42 \
  --num_rounds 500

# All results saved to results_step1/, results_step2/, etc.
```

### Test on Your Model
```bash
python step1_locate_political_layers.py \
  --model llama-7b \  # or your model
  --device cuda \     # or mps, cpu
  --num_rounds 200    # fewer rounds for speed
```

---

## 📞 Support & Questions

- **Issues**: Open a GitHub issue with details and error messages
- **Discussions**: Use GitHub Discussions for methodology questions
- **Email**: [your email if public]

---

## 📜 Citation

If you use this code or methodology, please cite:

```bibtex
@software{political_bias_repe_2025,
  title={Political Bias Representation Engineering: A Hierarchical Approach to Detecting and Steering Political Biases in LLMs},
  author={You, Geng},
  year={2025},
  url={https://github.com/YOUR_USERNAME/political-bias-representation-engineering},
  note={Adapted from Safety Layers (ICLR 2025), Bang et al. (ACL 2024), Zhou et al. (EMNLP 2024)}
}
```

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- **Safety Layers** (ICLR 2025) — Methodology foundation for layer localization
- **Bang et al.** (ACL 2024) — Topic-specific analysis and content/style decomposition
- **Zhou et al.** (EMNLP 2024) — Weak classifier probing approach
- **Zou et al.** — Representation Engineering (RepE) and CAA steering technique

---

<div align="center">

**Made with ❤️ for responsible AI research**

⭐ If you find this useful, please star the repo!

</div>
