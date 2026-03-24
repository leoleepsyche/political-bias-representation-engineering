# Getting Started Guide

## 5-Minute Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/political-bias-representation-engineering.git
cd political-bias-representation-engineering
```

### 2. Install Dependencies
```bash
# For Apple Silicon (Mac M1/M2)
pip install -r requirements.txt --break-system-packages

# For Linux/Windows with CUDA
pip install -r requirements.txt

# Minimal install (CPU only, faster install)
pip install torch transformers numpy matplotlib tqdm scikit-learn
```

### 3. Run Quick Demo
```bash
# Takes ~5 minutes, uses 1.5B model
python run_quick_demo.py
```

**Expected output**:
```
Loading model: Qwen/Qwen2.5-1.5B-Instruct
Extracting hidden states... [████████████████████] 100%
Computing cosine similarities... [████████████████████] 100%
Results saved to: ./results_demo
  Saved: ./results_demo/cosine_similarity_curves.png
  Saved: ./results_demo/angular_gap_analysis.png
  Saved: ./results_demo/angular_difference_curves.png
```

View the generated plots in `results_demo/` folder!

---

## Full Pipeline (2-4 hours)

### Setup Environment
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run All 4 Steps
```bash
# Step 1: Locate political layers (30-60 min)
echo "Step 1: Locating political layers..."
python step1_locate_political_layers.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num_rounds 500 \
  --seed 42

# Step 2: Analyze L/R/N bias (20-40 min)
echo "Step 2: Analyzing L/R/N bias..."
python step2_analyze_bias.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --step1_dir ./results_step1 \
  --seed 42

# Step 3: Topic-level analysis (20-40 min)
echo "Step 3: Topic-level analysis..."
python step3_topic_analysis.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --step1_dir ./results_step1 \
  --seed 42

# Step 4: Steering intervention (30-60 min)
echo "Step 4: Steering intervention..."
python step4_steering.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --step2_dir ./results_step2 \
  --alpha 0.5 1.0 2.0 3.0 5.0 \
  --seed 42

echo "All steps complete! Check results_step*/ directories."
```

### Or Run with Bash Script
```bash
# Save the above as run_pipeline.sh
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## Using Google Colab

### Free GPU Access
1. Open [Colab_Notebook.ipynb](Colab_Notebook.ipynb) in Google Colab
2. Click "Copy to Drive" → "Open in Colab"
3. Run cells sequentially (each takes 10-30 minutes with T4 GPU)
4. Download results when complete

**Advantages**:
- Free T4 GPU (2x faster than M1 Mac)
- No setup required
- Can interrupt and resume

**Limitations**:
- Runtime resets after 12 hours
- 50GB disk space limit
- 12 hours continuous time limit

---

## Custom Model

### Test on Your Own Model
```bash
python step1_locate_political_layers.py \
  --model meta-llama/Llama-2-7b-chat-hf \  # Any HF model
  --num_rounds 300 \
  --device cuda \
  --output-dir ./results_llama2
```

### Supported Models
- ✅ Any HuggingFace transformers model (7B-70B)
- ✅ Llama, Llama-2, Mistral, Qwen, Phi, etc.
- ✅ Open source only (requires HF token for Llama)
- ❌ OpenAI API (no hidden state access)

### Get HuggingFace Token
```bash
# 1. Create account at https://huggingface.co
# 2. Create token at https://huggingface.co/settings/tokens
# 3. Login locally:
huggingface-cli login
# 4. Paste token when prompted
```

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Use smaller model
python step1_locate_political_layers.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num_rounds 200

# Or reduce num_rounds
python step1_locate_political_layers.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num_rounds 200
```

### "MPS out of memory" (Mac)
```bash
# Use CPU instead
python step1_locate_political_layers.py \
  --device cpu \
  --num_rounds 200
```

### "Model not found"
```bash
# Ensure huggingface-cli login worked
huggingface-cli login

# Or use local path
python step1_locate_political_layers.py \
  --model /path/to/local/model
```

### "ImportError: No module named 'transformers'"
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

---

## Understanding Results

### After Step 1
```
results_step1/
├── political_layers.npz          # Data: layer boundaries, gap values
├── step1_cosine_gap.png          # Three similarity curves
├── step1_classifier_probing.png   # Accuracy per layer
└── step1_localization_summary.png # Combined visualization
```

**What to look for**:
- Political layers should be in middle layers (e.g., 15-25 out of 32)
- Classifier accuracy should jump >90% in political layers
- Cosine gap should have clear peak

### After Step 2
```
results_step2/
├── step2_results.npz              # Numerical results
├── direction_vectors.npz          # For Step 4
├── step2_bias_direction.png       # Main bias indicator
├── step2_three_class_probing.png   # L/R/N classification
└── step2_neutral_projection.png    # Neutral position
```

**What to look for**:
- Bias direction (positive = right lean, negative = left lean)
- If symmetric, model treats L/R equally
- Neutral's position on L-R axis

### After Step 3
```
results_step3/
├── step3_results.npz                # Topic strengths
├── step3_topic_layer_heatmap.png    # Which topics are political where
├── step3_topic_bias_direction.png   # Per-topic leaning
└── step3_content_vs_style.png       # Content vs style ratio
```

**What to look for**:
- Some topics more political than others (e.g., abortion > economics)
- Content > Style confirms we're measuring stance
- Topics show different bias directions

### After Step 4
```
results_step4/
├── step4_results.json                    # Numerical summary
├── behavioral_generations.json           # Generated text samples
├── step4_representation_steering.png     # Bias curves
└── step4_bias_capability_tradeoff.png    # Optimal alpha
```

**What to look for**:
- Find "elbow point" in bias-capability curve
- Optimal alpha (typically 1.0-2.0)
- Generated text changes subtly with steering

---

## Next Steps

### If You Want To...

**Contribute a new model**
```bash
git checkout -b feature/llama-analysis
python step1_locate_political_layers.py --model meta-llama/Llama-2-7b-chat-hf
# Create PR with results
```

**Add more topics**
- Edit `political_dataset.py`
- Add 5 more (topic_name, left_stmt, right_stmt) tuples
- Update `run_triangulation.py` and `step3_topic_analysis.py`
- Run full pipeline

**Extend to multilingual**
- Create `political_dataset_zh.py` for Chinese politics
- Create `political_dataset_fr.py` for French politics
- Run all steps on each language

**Implement new steering method**
- Copy `step4_steering.py` → `step4_steering_advanced.py`
- Modify `PoliticalSteeringHook` class
- Compare results with baseline

---

## Join the Community

- **Questions?** Open a GitHub Discussion
- **Found a bug?** File an Issue
- **Have an improvement?** Create a Pull Request
- **Want to chat?** Email or Discord (if available)

---

<div align="center">

**Happy researching! 🚀**

Remember to cite the original papers when you use this work.

</div>
