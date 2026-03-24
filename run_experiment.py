"""
Political Bias Cosine Similarity Gap Experiment
================================================
基于 Safety Layers (ICLR 2025) 的方法，检测 LLM 内部的政治倾向表征。

使用方法:
    python run_experiment.py [--model MODEL_NAME] [--quantize] [--num_rounds 500]

硬件要求:
    - Mac Studio M1 / Apple Silicon Mac
    - 至少 16GB 内存（使用 4-bit 量化时）
    - 或 32GB 内存（使用 float16 时）
"""

import argparse
import os
import random
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# 1. 配置
# ============================================================

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = Path("./results")


def parse_args():
    parser = argparse.ArgumentParser(description="Political Bias Cosine Similarity Gap")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model name or path")
    parser.add_argument("--quantize", action="store_true", default=False,
                        help="Use 4-bit quantization (saves memory, needs bitsandbytes)")
    parser.add_argument("--num_rounds", type=int, default=500,
                        help="Number of random pairing rounds (r in the paper)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, mps, cpu, cuda")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Directory to save raw results and plots")
    return parser.parse_args()


# ============================================================
# 2. 模型加载
# ============================================================

def load_model_and_tokenizer(model_name: str, quantize: bool, device: str):
    """加载模型和分词器"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"Quantization: {'4-bit' if quantize else 'float16/float32'}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 确定设备和精度
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    model_kwargs = {
        "trust_remote_code": True,
        "output_hidden_states": True,  # 关键：输出所有隐藏层状态
    }
    load_to_device = None

    if quantize:
        if device != "cuda":
            print("WARNING: 4-bit quantization is only supported on CUDA, falling back to non-quantized loading")
            quantize = False
        else:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["device_map"] = "auto"
            except ImportError:
                print("WARNING: bitsandbytes not available, falling back to float16")
                quantize = False

    if not quantize:
        model_kwargs["torch_dtype"] = torch.float16 if device in {"mps", "cuda"} else torch.float32
        if device in {"mps", "cuda"}:
            load_to_device = device

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if load_to_device is not None:
        model = model.to(load_to_device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model loaded successfully! Layers: {num_layers}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    return model, tokenizer, device, num_layers


# ============================================================
# 3. 隐藏状态提取
# ============================================================

@torch.no_grad()
def extract_hidden_states(model, tokenizer, text: str, device: str) -> list[torch.Tensor]:
    """
    提取输入文本在每一层最后一个 token 位置的隐藏状态向量。

    与 Safety Layers 论文一致：
    - 使用 last position 的 output vector
    - 在 first autoregressive step（即只做一次前向传播，不生成）
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # 移动到正确设备
    if device == "mps":
        inputs = {k: v.to("mps") for k, v in inputs.items()}
    elif device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = model(**inputs)

    # outputs.hidden_states: tuple of (num_layers + 1) tensors
    # 第 0 个是 embedding layer, 后面是每个 transformer layer 的输出
    # 每个 tensor shape: (batch_size, seq_len, hidden_dim)
    hidden_states = outputs.hidden_states

    # 取每层最后一个 token 的向量 (与 Safety Layers 一致)
    last_token_vectors = []
    for layer_idx, hs in enumerate(hidden_states):
        vec = hs[0, -1, :].detach().cpu().float()  # (hidden_dim,)
        last_token_vectors.append(vec)

    return last_token_vectors  # list of (num_layers+1) tensors


# ============================================================
# 4. 余弦相似度计算
# ============================================================

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """计算两个向量的余弦相似度"""
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def angular_difference(cos_sim: float) -> float:
    """将余弦相似度转换为角度差（度）"""
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return np.degrees(np.arccos(cos_sim))


# ============================================================
# 5. 核心实验逻辑
# ============================================================

def run_experiment(model, tokenizer, device: str, num_layers: int,
                   num_rounds: int = 500, seed: int = 42):
    """
    执行完整的余弦相似度鸿沟实验。

    步骤:
    1. 提取所有 left/right 语句的逐层隐藏状态
    2. 随机配对计算三种类型的余弦相似度
    3. 统计每层的均值和标准差
    """
    from political_dataset import (
        get_left_statements, get_right_statements, get_prompt_template
    )

    random.seed(seed)
    np.random.seed(seed)

    # --- Step 1: 提取所有隐藏状态 ---
    print("\n[Step 1] Extracting hidden states for all statements...")

    left_stmts = get_left_statements()
    right_stmts = get_right_statements()

    print(f"  Left statements: {len(left_stmts)}")
    print(f"  Right statements: {len(right_stmts)}")

    left_hidden = {}   # {topic: [vec_layer0, vec_layer1, ...]}
    right_hidden = {}

    for topic, stmt in tqdm(left_stmts, desc="  Extracting LEFT"):
        prompt = get_prompt_template(stmt)
        vecs = extract_hidden_states(model, tokenizer, prompt, device)
        left_hidden[topic] = vecs

    for topic, stmt in tqdm(right_stmts, desc="  Extracting RIGHT"):
        prompt = get_prompt_template(stmt)
        vecs = extract_hidden_states(model, tokenizer, prompt, device)
        right_hidden[topic] = vecs

    total_layers = num_layers + 1  # including embedding layer
    print(f"  Total layers (including embedding): {total_layers}")

    # --- Step 2: 随机配对计算余弦相似度 ---
    print(f"\n[Step 2] Computing cosine similarities ({num_rounds} rounds)...")

    left_topics = list(left_hidden.keys())
    right_topics = list(right_hidden.keys())

    # 存储每轮每层的余弦相似度
    ll_sims = np.zeros((num_rounds, total_layers))  # Left-Left
    rr_sims = np.zeros((num_rounds, total_layers))  # Right-Right
    lr_sims = np.zeros((num_rounds, total_layers))  # Left-Right

    for r in tqdm(range(num_rounds), desc="  Random pairing"):
        # Left-Left: 随机选两个不同的左翼话题
        t1, t2 = random.sample(left_topics, 2)
        for layer in range(total_layers):
            ll_sims[r, layer] = cosine_similarity(
                left_hidden[t1][layer], left_hidden[t2][layer]
            )

        # Right-Right: 随机选两个不同的右翼话题
        t1, t2 = random.sample(right_topics, 2)
        for layer in range(total_layers):
            rr_sims[r, layer] = cosine_similarity(
                right_hidden[t1][layer], right_hidden[t2][layer]
            )

        # Left-Right: 随机选一个左翼和一个右翼话题
        lt = random.choice(left_topics)
        rt = random.choice(right_topics)
        for layer in range(total_layers):
            lr_sims[r, layer] = cosine_similarity(
                left_hidden[lt][layer], right_hidden[rt][layer]
            )

    # --- Step 3: 计算统计量 ---
    print("\n[Step 3] Computing statistics...")

    results = {
        "ll_mean": np.mean(ll_sims, axis=0),
        "ll_std": np.std(ll_sims, axis=0),
        "rr_mean": np.mean(rr_sims, axis=0),
        "rr_std": np.std(rr_sims, axis=0),
        "lr_mean": np.mean(lr_sims, axis=0),
        "lr_std": np.std(lr_sims, axis=0),
        "num_layers": total_layers,
        "num_rounds": num_rounds,
    }

    # 计算 Angular Difference (如 Safety Layers Figure 2)
    # gap = angle(L-R pair) - angle(L-L pair 或 R-R pair 的平均)
    ll_angles = np.degrees(np.arccos(np.clip(ll_sims, -1, 1)))
    rr_angles = np.degrees(np.arccos(np.clip(rr_sims, -1, 1)))
    lr_angles = np.degrees(np.arccos(np.clip(lr_sims, -1, 1)))

    same_angles = (ll_angles + rr_angles) / 2  # 同类平均
    angular_gap = lr_angles - same_angles       # 跨类 - 同类

    results["angular_gap_mean"] = np.mean(angular_gap, axis=0)
    results["angular_gap_std"] = np.std(angular_gap, axis=0)
    results["ll_angle_mean"] = np.mean(ll_angles, axis=0)
    results["rr_angle_mean"] = np.mean(rr_angles, axis=0)
    results["lr_angle_mean"] = np.mean(lr_angles, axis=0)

    # 找到 gap 最大的层
    max_gap_layer = np.argmax(results["angular_gap_mean"])
    print(f"\n  Max angular gap at layer {max_gap_layer}: "
          f"{results['angular_gap_mean'][max_gap_layer]:.2f}° "
          f"(± {results['angular_gap_std'][max_gap_layer]:.2f}°)")

    return results


# ============================================================
# 6. 可视化
# ============================================================

def plot_results(results: dict, model_name: str, output_dir: Path):
    """生成与 Safety Layers 论文 Figure 1 & 2 类似的可视化"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    layers = np.arange(results["num_layers"])
    model_short = model_name.split("/")[-1]

    # ========================================
    # Figure 1: 三条余弦相似度曲线 (核心图)
    # ========================================
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Left-Left
    ax.plot(layers, results["ll_mean"], "b-", linewidth=2, label="Left-Left Pairs")
    ax.fill_between(layers,
                    results["ll_mean"] - results["ll_std"],
                    results["ll_mean"] + results["ll_std"],
                    alpha=0.15, color="blue")

    # Right-Right
    ax.plot(layers, results["rr_mean"], "r-", linewidth=2, label="Right-Right Pairs")
    ax.fill_between(layers,
                    results["rr_mean"] - results["rr_std"],
                    results["rr_mean"] + results["rr_std"],
                    alpha=0.15, color="red")

    # Left-Right
    ax.plot(layers, results["lr_mean"], "g-", linewidth=2, label="Left-Right Pairs")
    ax.fill_between(layers,
                    results["lr_mean"] - results["lr_std"],
                    results["lr_mean"] + results["lr_std"],
                    alpha=0.15, color="green")

    ax.set_xlabel("Layer Index", fontsize=13)
    ax.set_ylabel("Cosine Similarity", fontsize=13)
    ax.set_title(f"Layer-wise Cosine Similarity Analysis\n{model_short}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, results["num_layers"] - 1)

    plt.tight_layout()
    path1 = output_dir / "cosine_similarity_curves.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ========================================
    # Figure 2: 双面板 - 余弦相似度 + Angular Gap
    # (与 Safety Layers Figure 2 对应)
    # ========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])

    # 上半: 余弦相似度 (L-L/R-R vs L-R)
    same_mean = (results["ll_mean"] + results["rr_mean"]) / 2
    ax1.plot(layers, same_mean, "b-", linewidth=2,
             label="Same-Side Pairs (avg of L-L & R-R)")
    ax1.plot(layers, results["lr_mean"], "r-", linewidth=2,
             label="Cross-Side Pairs (L-R)")
    ax1.fill_between(layers,
                     results["lr_mean"] - results["lr_std"],
                     results["lr_mean"] + results["lr_std"],
                     alpha=0.15, color="red")
    ax1.set_ylabel("Cosine Similarity", fontsize=13)
    ax1.set_title(f"Political Bias Detection via Cosine Similarity Gap\n{model_short}",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 下半: Angular Gap
    gap = results["angular_gap_mean"]
    ax2.plot(layers, gap, "purple", linewidth=2.5, label="Angular Gap (Cross - Same)")
    ax2.fill_between(layers,
                     gap - results["angular_gap_std"],
                     gap + results["angular_gap_std"],
                     alpha=0.15, color="purple")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # 标注最大 gap 的位置
    max_idx = np.argmax(gap)
    ax2.annotate(f"Max gap: {gap[max_idx]:.1f}° at layer {max_idx}",
                 xy=(max_idx, gap[max_idx]),
                 xytext=(max_idx + 2, gap[max_idx] + 1),
                 arrowprops=dict(arrowstyle="->", color="red"),
                 fontsize=11, color="red", fontweight="bold")

    ax2.set_xlabel("Layer Index", fontsize=13)
    ax2.set_ylabel("Angular Difference (degrees)", fontsize=13)
    ax2.set_title("Mean Angular Difference (Left-Right Gap)", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = output_dir / "angular_gap_analysis.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path2}")

    # ========================================
    # Figure 3: 三条 Angular Difference 曲线
    # ========================================
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot(layers, results["ll_angle_mean"], "b-", linewidth=2,
            label="Left-Left Angular Diff")
    ax.plot(layers, results["rr_angle_mean"], "r-", linewidth=2,
            label="Right-Right Angular Diff")
    ax.plot(layers, results["lr_angle_mean"], "g-", linewidth=2,
            label="Left-Right Angular Diff")

    ax.set_xlabel("Layer Index", fontsize=13)
    ax.set_ylabel("Mean Angular Difference (degrees)", fontsize=13)
    ax.set_title(f"Layer-wise Angular Difference\n{model_short}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path3 = output_dir / "angular_difference_curves.png"
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path3}")

    return [path1, path2, path3]


# ============================================================
# 7. 统计检验
# ============================================================

def statistical_analysis(results: dict):
    """对关键层的 gap 进行统计显著性分析"""
    from scipy import stats

    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    gap = results["angular_gap_mean"]
    gap_std = results["angular_gap_std"]
    n = results["num_rounds"]

    # 找到 gap 最大的 5 层
    top5_layers = np.argsort(gap)[-5:][::-1]

    print(f"\nTop 5 layers with largest angular gap:")
    print(f"{'Layer':>6} {'Gap (°)':>10} {'Std (°)':>10} {'t-stat':>10} {'p-value':>12}")
    print("-" * 52)

    for layer in top5_layers:
        # 单样本 t 检验: gap 是否显著大于 0
        t_stat = gap[layer] / (gap_std[layer] / np.sqrt(n))
        p_value = 1 - stats.t.cdf(t_stat, df=n-1)  # 单尾
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"{layer:>6} {gap[layer]:>10.2f} {gap_std[layer]:>10.2f} "
              f"{t_stat:>10.2f} {p_value:>11.2e} {sig}")

    # 总体分析
    print(f"\n--- Overall Summary ---")
    layers_with_positive_gap = np.sum(gap > 0)
    print(f"Layers with positive gap: {layers_with_positive_gap}/{len(gap)}")
    print(f"Max gap: {np.max(gap):.2f}° at layer {np.argmax(gap)}")
    print(f"Mean gap (all layers): {np.mean(gap):.2f}°")
    print(f"Mean gap (middle 50% layers): {np.mean(gap[len(gap)//4:3*len(gap)//4]):.2f}°")

    return top5_layers


# ============================================================
# 8. 主程序
# ============================================================

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("  Political Bias Cosine Similarity Gap Experiment")
    print("  Based on: Safety Layers (ICLR 2025)")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Rounds:     {args.num_rounds}")
    print(f"  Quantize:   {args.quantize}")
    print(f"  Seed:       {args.seed}")
    print(f"  Device:     {args.device}")
    print("=" * 60)

    # 加载模型
    model, tokenizer, device, num_layers = load_model_and_tokenizer(
        args.model, args.quantize, args.device
    )

    # 运行实验
    results = run_experiment(
        model, tokenizer, device, num_layers,
        num_rounds=args.num_rounds,
        seed=args.seed
    )

    # 保存原始数据
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "raw_results.npz",
        **{k: v for k, v in results.items() if isinstance(v, np.ndarray)}
    )
    print(f"\nRaw results saved to {output_dir / 'raw_results.npz'}")

    # 可视化
    print("\n[Step 4] Generating visualizations...")
    plot_paths = plot_results(results, args.model, output_dir)

    # 统计分析
    try:
        statistical_analysis(results)
    except ImportError:
        print("\n(scipy not installed, skipping statistical tests)")
        print("Install with: pip install scipy --break-system-packages")

    print("\n" + "=" * 60)
    print("  Experiment Complete!")
    print(f"  Results saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
