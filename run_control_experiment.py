"""
Control Experiment: Disentangle Political Bias from Lexical Difference
======================================================================

三个控制实验，验证 cosine gap 的来源：

  Control 1: Neutral Lexical Control
    - 用中性语句（词汇风格相似但无立场）替换 left/right
    - 如果 gap 消失 → 原始 gap 来自政治立场
    - 如果 gap 保持 → 原始 gap 来自词汇差异

  Control 2: Topic-Shuffled Control
    - 打乱 left-right 的话题配对
    - 如果 gap 保持 → 存在系统性的 left/right 词汇风格差异
    - 如果 gap 缩小 → 模型在做同话题内的语义区分

  Control 3: Base Model Control
    - 在未 aligned 的 base 模型上重复实验
    - 如果 gap 缩小 → gap 部分来自 alignment
    - 如果 gap 不变 → gap 来自预训练阶段的表征

用法:
    python run_control_experiment.py --model Qwen/Qwen2.5-7B-Instruct
    python run_control_experiment.py --model Qwen/Qwen2.5-7B-Instruct --control all
    python run_control_experiment.py --model Qwen/Qwen2.5-7B-Instruct --control neutral
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from run_experiment import (
    load_model_and_tokenizer,
    extract_hidden_states,
    cosine_similarity,
)
from political_dataset import (
    get_left_statements,
    get_right_statements,
    get_prompt_template,
)
from control_dataset import (
    get_neutral_a_statements,
    get_neutral_b_statements,
    get_base_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--control", type=str, default="all",
                        choices=["all", "neutral", "shuffle", "base"],
                        help="Which control to run")
    parser.add_argument("--num_rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


# ============================================================
# 通用：提取一组语句的 hidden states
# ============================================================

def extract_all_hidden(statements, model, tokenizer, device, label=""):
    """提取一组语句的所有隐藏状态"""
    hidden = {}
    for topic, stmt in tqdm(statements, desc=f"  Extracting {label}"):
        prompt = get_prompt_template(stmt)
        vecs = extract_hidden_states(model, tokenizer, prompt, device)
        hidden[topic] = vecs
    return hidden


def compute_pairwise_sims(hidden_a, hidden_b, num_rounds, total_layers, cross=False):
    """
    计算配对余弦相似度
    cross=False: 从同一个 dict 随机选两个不同 topic
    cross=True: 从两个不同 dict 各选一个 topic
    """
    topics_a = list(hidden_a.keys())
    topics_b = list(hidden_b.keys()) if cross else topics_a

    sims = np.zeros((num_rounds, total_layers))
    for r in range(num_rounds):
        if cross:
            ta = random.choice(topics_a)
            tb = random.choice(topics_b)
        else:
            ta, tb = random.sample(topics_a, 2)
            hidden_b = hidden_a  # 同一组内配对

        for layer in range(total_layers):
            sims[r, layer] = cosine_similarity(
                hidden_a[ta][layer], hidden_b[tb][layer]
            )
    return sims


# ============================================================
# Control 1: Neutral Lexical Control
# ============================================================

def run_neutral_control(model, tokenizer, device, num_layers, num_rounds, seed):
    """
    用词汇风格相似但立场中性的语句替换 left/right。
    如果原始 gap 来自政治立场而非词汇，这个 control 的 gap 应该显著更小。
    """
    print("\n" + "=" * 60)
    print("  CONTROL 1: Neutral Lexical Control")
    print("=" * 60)

    random.seed(seed)
    np.random.seed(seed)

    neutral_a = get_neutral_a_statements()
    neutral_b = get_neutral_b_statements()

    na_hidden = extract_all_hidden(neutral_a, model, tokenizer, device, "Neutral-A")
    nb_hidden = extract_all_hidden(neutral_b, model, tokenizer, device, "Neutral-B")

    total_layers = num_layers + 1

    # A-A, B-B, A-B 配对
    print(f"\n  Computing cosine similarities ({num_rounds} rounds)...")
    aa_sims = compute_pairwise_sims(na_hidden, na_hidden, num_rounds, total_layers, cross=False)
    bb_sims = compute_pairwise_sims(nb_hidden, nb_hidden, num_rounds, total_layers, cross=False)
    ab_sims = compute_pairwise_sims(na_hidden, nb_hidden, num_rounds, total_layers, cross=True)

    # Angular gap
    aa_angles = np.degrees(np.arccos(np.clip(aa_sims, -1, 1)))
    bb_angles = np.degrees(np.arccos(np.clip(bb_sims, -1, 1)))
    ab_angles = np.degrees(np.arccos(np.clip(ab_sims, -1, 1)))

    neutral_gap = np.mean(ab_angles - (aa_angles + bb_angles) / 2, axis=0)

    return {
        "aa_mean": np.mean(aa_sims, axis=0),
        "bb_mean": np.mean(bb_sims, axis=0),
        "ab_mean": np.mean(ab_sims, axis=0),
        "neutral_gap": neutral_gap,
        "num_layers": total_layers,
    }


# ============================================================
# Control 2: Topic-Shuffled Control
# ============================================================

def run_shuffle_control(model, tokenizer, device, num_layers, num_rounds, seed):
    """
    使用原始 left/right 数据，但打乱话题配对。
    Left-Right 配对时，故意让 left 和 right 来自不同话题。
    """
    print("\n" + "=" * 60)
    print("  CONTROL 2: Topic-Shuffled Control")
    print("=" * 60)

    random.seed(seed)
    np.random.seed(seed)

    left_stmts = get_left_statements()
    right_stmts = get_right_statements()

    left_hidden = extract_all_hidden(left_stmts, model, tokenizer, device, "LEFT")
    right_hidden = extract_all_hidden(right_stmts, model, tokenizer, device, "RIGHT")

    total_layers = num_layers + 1
    left_topics = list(left_hidden.keys())
    right_topics = list(right_hidden.keys())

    # Same-topic L-R (原始实验)
    print(f"\n  Computing SAME-TOPIC L-R similarities...")
    same_topic_sims = np.zeros((num_rounds, total_layers))
    for r in range(num_rounds):
        topic = random.choice(left_topics)
        for layer in range(total_layers):
            same_topic_sims[r, layer] = cosine_similarity(
                left_hidden[topic][layer], right_hidden[topic][layer]
            )

    # Cross-topic L-R (打乱话题)
    print(f"  Computing CROSS-TOPIC L-R similarities...")
    cross_topic_sims = np.zeros((num_rounds, total_layers))
    for r in range(num_rounds):
        # 确保选择不同话题
        lt = random.choice(left_topics)
        rt = random.choice([t for t in right_topics if t != lt])
        for layer in range(total_layers):
            cross_topic_sims[r, layer] = cosine_similarity(
                left_hidden[lt][layer], right_hidden[rt][layer]
            )

    return {
        "same_topic_mean": np.mean(same_topic_sims, axis=0),
        "same_topic_std": np.std(same_topic_sims, axis=0),
        "cross_topic_mean": np.mean(cross_topic_sims, axis=0),
        "cross_topic_std": np.std(cross_topic_sims, axis=0),
        "num_layers": total_layers,
    }


# ============================================================
# Control 3: Base Model Control
# ============================================================

def run_base_model_control(aligned_model, device, num_layers_aligned, num_rounds, seed):
    """
    在 base 模型上重复实验，对比 aligned 和 base 的 gap 差异。
    """
    print("\n" + "=" * 60)
    print("  CONTROL 3: Base Model Comparison")
    print("=" * 60)

    base_model_name = get_base_model(aligned_model)
    if base_model_name is None:
        print(f"  No known base model for {aligned_model}, skipping.")
        return None

    print(f"  Aligned: {aligned_model}")
    print(f"  Base:    {base_model_name}")

    random.seed(seed)
    np.random.seed(seed)

    model, tokenizer, device, num_layers = load_model_and_tokenizer(
        base_model_name, quantize=False, device=device
    )

    left_stmts = get_left_statements()
    right_stmts = get_right_statements()

    left_hidden = extract_all_hidden(left_stmts, model, tokenizer, device, "LEFT (base)")
    right_hidden = extract_all_hidden(right_stmts, model, tokenizer, device, "RIGHT (base)")

    total_layers = num_layers + 1

    ll_sims = compute_pairwise_sims(left_hidden, left_hidden, num_rounds, total_layers, cross=False)
    rr_sims = compute_pairwise_sims(right_hidden, right_hidden, num_rounds, total_layers, cross=False)
    lr_sims = compute_pairwise_sims(left_hidden, right_hidden, num_rounds, total_layers, cross=True)

    ll_angles = np.degrees(np.arccos(np.clip(ll_sims, -1, 1)))
    rr_angles = np.degrees(np.arccos(np.clip(rr_sims, -1, 1)))
    lr_angles = np.degrees(np.arccos(np.clip(lr_sims, -1, 1)))

    base_gap = np.mean(lr_angles - (ll_angles + rr_angles) / 2, axis=0)

    # 清理 GPU 内存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "base_model": base_model_name,
        "base_gap": base_gap,
        "base_ll_mean": np.mean(ll_sims, axis=0),
        "base_rr_mean": np.mean(rr_sims, axis=0),
        "base_lr_mean": np.mean(lr_sims, axis=0),
        "num_layers": total_layers,
    }


# ============================================================
# 可视化：对比图
# ============================================================

def plot_all_controls(original_results_path, neutral_res, shuffle_res, base_res,
                      model_name, output_dir):
    """生成控制实验的对比可视化"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]

    # 加载原始实验结果
    original = np.load(original_results_path)
    orig_gap = original["angular_gap_mean"]
    layers = np.arange(len(orig_gap))

    # ========================================
    # 综合对比图: Original vs All Controls
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (0,0) 原始实验
    ax = axes[0, 0]
    ax.plot(layers, original["ll_mean"], "b-", lw=2, label="L-L")
    ax.plot(layers, original["rr_mean"], "r-", lw=2, label="R-R")
    ax.plot(layers, original["lr_mean"], "g-", lw=2, label="L-R")
    ax.set_title("Original: Left vs Right", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cosine Similarity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,1) Control 1: Neutral
    ax = axes[0, 1]
    if neutral_res:
        nl = np.arange(neutral_res["num_layers"])
        ax.plot(nl, neutral_res["aa_mean"], "b-", lw=2, label="NeutA-NeutA")
        ax.plot(nl, neutral_res["bb_mean"], "r-", lw=2, label="NeutB-NeutB")
        ax.plot(nl, neutral_res["ab_mean"], "g-", lw=2, label="NeutA-NeutB")
        ax.set_title("Control 1: Neutral (same vocab, no stance)", fontsize=12, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Not run", transform=ax.transAxes, ha="center")
    ax.set_ylabel("Cosine Similarity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,0) Control 2: Shuffle
    ax = axes[1, 0]
    if shuffle_res:
        nl = np.arange(shuffle_res["num_layers"])
        ax.plot(nl, shuffle_res["same_topic_mean"], "r-", lw=2, label="Same-Topic L-R")
        ax.plot(nl, shuffle_res["cross_topic_mean"], "b--", lw=2, label="Cross-Topic L-R")
        ax.set_title("Control 2: Same-Topic vs Cross-Topic L-R", fontsize=12, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Not run", transform=ax.transAxes, ha="center")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Cosine Similarity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,1) Angular Gap 对比
    ax = axes[1, 1]
    ax.plot(layers, orig_gap, "purple", lw=2.5, label="Original (Left vs Right)")
    if neutral_res:
        nl = np.arange(neutral_res["num_layers"])
        ax.plot(nl, neutral_res["neutral_gap"], "orange", lw=2, ls="--",
                label="Control 1 (Neutral)")
    if base_res:
        nl = np.arange(base_res["num_layers"])
        ax.plot(nl, base_res["base_gap"], "cyan", lw=2, ls="-.",
                label=f"Control 3 (Base: {base_res['base_model'].split('/')[-1]})")
    ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
    ax.set_title("Angular Gap Comparison", fontsize=12, fontweight="bold")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Angular Difference (°)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Control Experiments: {model_short}\n"
                 f"Disentangling Political Bias from Lexical Difference",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "control_experiments_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")
    return path


# ============================================================
# 结论分析
# ============================================================

def analyze_controls(original_gap, neutral_res, shuffle_res, base_res):
    """基于控制实验给出结论"""
    print("\n" + "=" * 60)
    print("  CONTROL EXPERIMENT CONCLUSIONS")
    print("=" * 60)

    orig_max = np.max(original_gap)
    orig_mean = np.mean(original_gap)

    # Control 1 分析
    if neutral_res:
        neut_max = np.max(neutral_res["neutral_gap"])
        neut_mean = np.mean(neutral_res["neutral_gap"])
        ratio = neut_max / orig_max if orig_max > 0 else float("inf")

        print(f"\n[Control 1] Neutral Lexical Control:")
        print(f"  Original max gap:  {orig_max:.2f}°")
        print(f"  Neutral max gap:   {neut_max:.2f}°")
        print(f"  Ratio:             {ratio:.2f}")

        if ratio < 0.5:
            print(f"  ✅ CONCLUSION: Gap is primarily from POLITICAL STANCE, not lexical difference")
            print(f"     (Neutral gap is <50% of original → vocabulary alone doesn't explain the gap)")
        elif ratio < 0.8:
            print(f"  ⚠️  CONCLUSION: Gap is PARTIALLY from political stance, partially lexical")
            print(f"     (Neutral gap is 50-80% of original → some lexical confounding)")
        else:
            print(f"  ❌ CONCLUSION: Gap is primarily from LEXICAL DIFFERENCE")
            print(f"     (Neutral gap is >80% of original → vocabulary drives the gap)")

    # Control 2 分析
    if shuffle_res:
        same_mean = np.mean(shuffle_res["same_topic_mean"])
        cross_mean = np.mean(shuffle_res["cross_topic_mean"])
        diff = same_mean - cross_mean

        print(f"\n[Control 2] Topic-Shuffled Control:")
        print(f"  Same-topic L-R mean sim:  {same_mean:.4f}")
        print(f"  Cross-topic L-R mean sim: {cross_mean:.4f}")
        print(f"  Difference:               {diff:.4f}")

        if abs(diff) < 0.01:
            print(f"  → Same-topic and cross-topic L-R are similar")
            print(f"     → Gap comes from systematic L/R vocabulary style, not topic-specific comparison")
        else:
            print(f"  → Same-topic L-R differs from cross-topic L-R")
            print(f"     → Model does topic-specific political comparison (good for your thesis!)")

    # Control 3 分析
    if base_res:
        base_max = np.max(base_res["base_gap"])
        ratio = base_max / orig_max if orig_max > 0 else float("inf")

        print(f"\n[Control 3] Base Model Control:")
        print(f"  Aligned model max gap: {orig_max:.2f}°")
        print(f"  Base model max gap:    {base_max:.2f}°")
        print(f"  Ratio:                 {ratio:.2f}")

        if ratio < 0.5:
            print(f"  ✅ CONCLUSION: Alignment AMPLIFIED the political representation gap")
            print(f"     (Base gap is <50% of aligned → RLHF contributes significantly)")
        elif ratio < 0.8:
            print(f"  ⚠️  CONCLUSION: Alignment partially amplified the gap")
        else:
            print(f"  → Gap exists in base model too (pre-training already encodes political structure)")

    print("\n" + "=" * 60)


# ============================================================
# 主程序
# ============================================================

def main():
    args = parse_args()
    output_dir = Path("./results_controls")

    model, tokenizer, device, num_layers = load_model_and_tokenizer(
        args.model, quantize=False, device=args.device
    )

    neutral_res = None
    shuffle_res = None
    base_res = None

    if args.control in ("all", "neutral"):
        neutral_res = run_neutral_control(
            model, tokenizer, device, num_layers, args.num_rounds, args.seed
        )

    if args.control in ("all", "shuffle"):
        shuffle_res = run_shuffle_control(
            model, tokenizer, device, num_layers, args.num_rounds, args.seed
        )

    # 释放 aligned 模型内存（base model 需要空间）
    if args.control in ("all", "base"):
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        base_res = run_base_model_control(
            args.model, args.device, num_layers, args.num_rounds, args.seed
        )

    # 可视化和分析
    orig_path = Path("./results/raw_results.npz")
    if orig_path.exists():
        original = np.load(orig_path)
        orig_gap = original["angular_gap_mean"]

        print("\n[Plotting] Generating comparison charts...")
        plot_all_controls(orig_path, neutral_res, shuffle_res, base_res,
                          args.model, output_dir)
        analyze_controls(orig_gap, neutral_res, shuffle_res, base_res)
    else:
        print(f"\n⚠️  Original results not found at {orig_path}")
        print(f"   Run `python run_experiment.py` first!")

    # 保存控制实验结果
    save_dict = {}
    if neutral_res:
        save_dict.update({f"ctrl1_{k}": v for k, v in neutral_res.items()
                          if isinstance(v, np.ndarray)})
    if shuffle_res:
        save_dict.update({f"ctrl2_{k}": v for k, v in shuffle_res.items()
                          if isinstance(v, np.ndarray)})
    if base_res:
        save_dict.update({f"ctrl3_{k}": v for k, v in base_res.items()
                          if isinstance(v, np.ndarray)})

    if save_dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / "control_results.npz", **save_dict)
        print(f"\nControl results saved to {output_dir / 'control_results.npz'}")


if __name__ == "__main__":
    main()
