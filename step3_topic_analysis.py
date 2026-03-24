"""
Step 3: Topic-Level Fine-Grained Analysis in Political Layers
==============================================================
前提: Step 1 找到了 political layers, Step 2 确认了 L/R/N bias 存在。
本步骤在 topic 级别做精细化分析。

三个核心分析:
  1. Topic × Layer 热力图: 哪些话题在哪些层的政治性最强？
     (对标 Bang et al. ACL 2024: topic-specific stance analysis)
  2. Topic-specific bias direction: 每个话题的偏见方向和强度
  3. Content vs Style 分解: 政治性 gap 来自内容还是表达风格？
     (对标 Bang et al.: content/style decomposition)

用法:
    python step3_topic_analysis.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
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
from run_triangulation import get_neutral_statements


# ============================================================
# Style Pairs: 同一立场，不同表达风格
# (学术 vs 激进 → 分离 content vs style)
# ============================================================

STYLE_PAIRS = {
    "healthcare": {
        "left_academic":
            "Research suggests that universal healthcare systems tend to produce "
            "better population health outcomes at lower per-capita costs compared "
            "to market-based systems, according to comparative health policy studies.",
        "left_activist":
            "Healthcare is a human right! No one should go bankrupt because they "
            "got sick! We need Medicare for All now — the insurance companies are "
            "literally letting people die for profit!",
        "right_academic":
            "Economic analysis indicates that market-based healthcare systems promote "
            "innovation and efficiency through competition, with government programs "
            "often introducing bureaucratic overhead that reduces service quality.",
        "right_activist":
            "Government healthcare is a disaster everywhere it's been tried! "
            "Socialized medicine means longer wait times, fewer choices, and "
            "bureaucrats deciding your medical care instead of your doctor!",
    },
    "climate": {
        "left_academic":
            "The scientific consensus, supported by over 97 percent of climate "
            "researchers, indicates that anthropogenic greenhouse gas emissions "
            "are the primary driver of observed global warming trends.",
        "left_activist":
            "We are in a climate EMERGENCY! Fossil fuel companies knew for decades "
            "and covered it up! We need a Green New Deal NOW or we doom our children "
            "to an unlivable planet!",
        "right_academic":
            "While acknowledging observed warming trends, some economists argue that "
            "aggressive decarbonization policies may impose disproportionate costs on "
            "developing economies relative to the expected climate benefits.",
        "right_activist":
            "The climate alarmists want to destroy our economy with their radical "
            "Green New Deal! They want to ban cars, ban meat, and control every "
            "aspect of your life in the name of 'saving the planet'!",
    },
    "immigration": {
        "left_academic":
            "Immigration research consistently shows that immigrants contribute "
            "positively to economic growth, innovation, and fiscal balance over "
            "the long term, with net positive effects on GDP per capita.",
        "left_activist":
            "No human being is illegal! These are families fleeing violence and "
            "poverty — tearing children from their parents at the border is a "
            "moral abomination that shames our entire nation!",
        "right_academic":
            "Labor economics research suggests that high levels of low-skill "
            "immigration can depress wages for native-born workers in competing "
            "sectors, particularly those without college degrees.",
        "right_activist":
            "Our borders are wide open and it's destroying our communities! "
            "Illegal immigrants are flooding in, taking jobs, and the politicians "
            "do nothing because they want the votes!",
    },
    "gun_control": {
        "left_academic":
            "Epidemiological studies demonstrate a strong correlation between "
            "firearm availability and gun death rates, with nations having stricter "
            "gun regulations showing significantly lower rates of gun violence.",
        "left_activist":
            "How many more children have to die before we do something?! "
            "The NRA has blood on its hands! We need to ban assault weapons "
            "and implement universal background checks TODAY!",
        "right_academic":
            "Constitutional scholarship emphasizes that the Second Amendment "
            "protects an individual right to bear arms, and defensive gun use "
            "research estimates hundreds of thousands of protective uses annually.",
        "right_activist":
            "They want to take your guns and leave you defenseless! The Second "
            "Amendment is clear — shall not be infringed! An armed society is "
            "a polite society, and no government will ever take our rights!",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--step1_dir", type=str, default="./results_step1")
    return parser.parse_args()


def load_political_layers(step1_dir):
    path = Path(step1_dir) / "political_layers.npz"
    if not path.exists():
        raise FileNotFoundError(f"Step 1 results not found at {path}")
    data = np.load(path)
    lower = int(data["political_layer_lower"])
    upper = int(data["political_layer_upper"])
    return lower, upper


# ============================================================
# Analysis 1: Topic × Layer Heatmap
# ============================================================

def topic_layer_heatmap(model, tokenizer, device, num_layers,
                         pol_lower, pol_upper, seed=42):
    """
    计算每个话题在每层的 L-R cosine gap。
    生成 topic × layer 热力图。
    (对标 Bang et al. topic-specific stance analysis)
    """
    print("\n" + "=" * 60)
    print("  Analysis 1: Topic × Layer Political Gap Heatmap")
    print("=" * 60)

    random.seed(seed)
    left_stmts = get_left_statements()
    right_stmts = get_right_statements()
    neutral_stmts = get_neutral_statements()

    # 建立 topic 字典
    left_dict = {t: s for t, s in left_stmts}
    right_dict = {t: s for t, s in right_stmts}
    neutral_dict = {t: s for t, s in neutral_stmts}

    topics = sorted(set(left_dict.keys()) & set(right_dict.keys()) & set(neutral_dict.keys()))
    total_layers = num_layers + 1

    print(f"  Topics: {len(topics)}")
    print(f"  Extracting hidden states...")

    left_hidden = {}
    right_hidden = {}
    neutral_hidden = {}

    for topic in tqdm(topics, desc="  Extracting"):
        left_hidden[topic] = extract_hidden_states(
            model, tokenizer, get_prompt_template(left_dict[topic]), device)
        right_hidden[topic] = extract_hidden_states(
            model, tokenizer, get_prompt_template(right_dict[topic]), device)
        neutral_hidden[topic] = extract_hidden_states(
            model, tokenizer, get_prompt_template(neutral_dict[topic]), device)

    # 三个热力图矩阵: L-R gap, N-L gap, N-R gap
    lr_gap_matrix = np.zeros((len(topics), total_layers))
    nl_gap_matrix = np.zeros((len(topics), total_layers))
    nr_gap_matrix = np.zeros((len(topics), total_layers))
    bias_matrix = np.zeros((len(topics), total_layers))

    for i, topic in enumerate(topics):
        for layer in range(total_layers):
            lr_sim = cosine_similarity(left_hidden[topic][layer], right_hidden[topic][layer])
            nl_sim = cosine_similarity(neutral_hidden[topic][layer], left_hidden[topic][layer])
            nr_sim = cosine_similarity(neutral_hidden[topic][layer], right_hidden[topic][layer])

            lr_gap_matrix[i, layer] = np.degrees(np.arccos(np.clip(lr_sim, -1, 1)))
            nl_gap_matrix[i, layer] = np.degrees(np.arccos(np.clip(nl_sim, -1, 1)))
            nr_gap_matrix[i, layer] = np.degrees(np.arccos(np.clip(nr_sim, -1, 1)))
            bias_matrix[i, layer] = nl_gap_matrix[i, layer] - nr_gap_matrix[i, layer]

    # 找出 political layers 内最"政治化"的话题
    pol_slice = lr_gap_matrix[:, pol_lower:pol_upper + 1]
    topic_pol_strength = np.mean(pol_slice, axis=1)
    ranked_topics = sorted(zip(topics, topic_pol_strength), key=lambda x: -x[1])

    print(f"\n  === TOPIC POLITICAL STRENGTH (layers [{pol_lower}-{pol_upper}]) ===")
    for topic, strength in ranked_topics:
        bias_val = np.mean(bias_matrix[topics.index(topic), pol_lower:pol_upper + 1])
        direction = "→R" if bias_val > 0.3 else ("→L" if bias_val < -0.3 else "≈0")
        print(f"    {topic:<25} L-R gap: {strength:>6.2f}°  bias: {bias_val:+.2f}° ({direction})")

    return {
        "topics": topics,
        "lr_gap_matrix": lr_gap_matrix,
        "nl_gap_matrix": nl_gap_matrix,
        "nr_gap_matrix": nr_gap_matrix,
        "bias_matrix": bias_matrix,
        "topic_pol_strength": topic_pol_strength,
        "total_layers": total_layers,
        "_left_hidden": left_hidden,
        "_right_hidden": right_hidden,
        "_neutral_hidden": neutral_hidden,
    }


# ============================================================
# Analysis 2: Content vs Style Decomposition
# ============================================================

def content_style_decomposition(model, tokenizer, device, num_layers,
                                 pol_lower, pol_upper, seed=42):
    """
    对标 Bang et al. (ACL 2024) 的 content vs style 分析。

    方法:
      对于每个话题，比较4个语句:
        - left_academic  vs left_activist   (same content, different style)
        - right_academic vs right_activist  (same content, different style)
        - left_academic  vs right_academic  (different content, same style)
        - left_activist  vs right_activist  (different content, same style)

      如果 content gap >> style gap → gap 主要编码政治立场
      如果 style gap >> content gap → gap 主要编码表达方式
    """
    print("\n" + "=" * 60)
    print("  Analysis 2: Content vs Style Decomposition")
    print("=" * 60)

    random.seed(seed)
    total_layers = num_layers + 1
    style_topics = list(STYLE_PAIRS.keys())
    print(f"  Topics with style pairs: {len(style_topics)}")

    # 提取 hidden states for all style variants
    hidden = {}
    for topic in tqdm(style_topics, desc="  Extracting style pairs"):
        for variant_name, text in STYLE_PAIRS[topic].items():
            key = f"{topic}_{variant_name}"
            hidden[key] = extract_hidden_states(
                model, tokenizer, get_prompt_template(text), device)

    # 计算 4 种 gap
    content_gaps = np.zeros(total_layers)  # same style, different content (L vs R)
    style_gaps = np.zeros(total_layers)    # same content, different style (acad vs activist)
    n_pairs = 0

    for topic in style_topics:
        la = hidden[f"{topic}_left_academic"]
        lx = hidden[f"{topic}_left_activist"]
        ra = hidden[f"{topic}_right_academic"]
        rx = hidden[f"{topic}_right_activist"]

        for layer in range(total_layers):
            # Content gap (L vs R, controlling for style)
            content_acad = np.degrees(np.arccos(np.clip(
                cosine_similarity(la[layer], ra[layer]), -1, 1)))
            content_actv = np.degrees(np.arccos(np.clip(
                cosine_similarity(lx[layer], rx[layer]), -1, 1)))

            # Style gap (academic vs activist, controlling for content)
            style_left = np.degrees(np.arccos(np.clip(
                cosine_similarity(la[layer], lx[layer]), -1, 1)))
            style_right = np.degrees(np.arccos(np.clip(
                cosine_similarity(ra[layer], rx[layer]), -1, 1)))

            content_gaps[layer] += (content_acad + content_actv) / 2
            style_gaps[layer] += (style_left + style_right) / 2
        n_pairs += 1

    content_gaps /= n_pairs
    style_gaps /= n_pairs

    # 在 political layers 内的对比
    pol_content = np.mean(content_gaps[pol_lower:pol_upper + 1])
    pol_style = np.mean(style_gaps[pol_lower:pol_upper + 1])
    ratio = pol_content / (pol_style + 1e-8)

    print(f"\n  === CONTENT vs STYLE RESULTS (layers [{pol_lower}-{pol_upper}]) ===")
    print(f"  Content gap (L vs R, same style): {pol_content:.2f}°")
    print(f"  Style gap (acad vs activist, same content): {pol_style:.2f}°")
    print(f"  Content/Style ratio: {ratio:.2f}x")

    if ratio > 1.5:
        print(f"  → Gap primarily encodes POLITICAL STANCE (content > style)")
    elif ratio < 0.67:
        print(f"  → Gap primarily encodes EXPRESSION STYLE (style > content)")
    else:
        print(f"  → Gap encodes BOTH content and style roughly equally")

    return {
        "content_gaps": content_gaps,
        "style_gaps": style_gaps,
        "pol_content_mean": pol_content,
        "pol_style_mean": pol_style,
        "ratio": ratio,
        "total_layers": total_layers,
    }


# ============================================================
# 可视化
# ============================================================

def plot_step3_results(heatmap_res, style_res, model_name, output_dir, pol_lower, pol_upper):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]

    # ========================================
    # Figure 1: Topic × Layer L-R Gap Heatmap
    # ========================================
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    topics = heatmap_res["topics"]
    matrix = heatmap_res["lr_gap_matrix"]

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics, fontsize=9)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Topic", fontsize=12)
    ax.set_title(f"Topic × Layer: Left-Right Angular Gap — {model_short}\n"
                 f"(Brighter = larger political gap)",
                 fontsize=14, fontweight="bold")

    # 标注 political layers
    ax.axvline(x=pol_lower - 0.5, color="cyan", ls="--", lw=2)
    ax.axvline(x=pol_upper + 0.5, color="cyan", ls="--", lw=2)
    ax.text(pol_lower, -1.5, f"Political\nLayers", fontsize=9, color="cyan",
            ha="center", fontweight="bold")

    plt.colorbar(im, ax=ax, label="Angular Gap (°)")
    plt.tight_layout()
    p1 = output_dir / "step3_topic_layer_heatmap.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p1}")

    # ========================================
    # Figure 2: Topic Bias Direction (in political layers)
    # ========================================
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    bias_in_pol = np.mean(heatmap_res["bias_matrix"][:, pol_lower:pol_upper + 1], axis=1)
    sorted_idx = np.argsort(bias_in_pol)
    sorted_topics = [topics[i] for i in sorted_idx]
    sorted_bias = bias_in_pol[sorted_idx]

    colors = ["red" if b > 0 else "blue" for b in sorted_bias]
    ax.barh(range(len(sorted_topics)), sorted_bias, color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_topics)))
    ax.set_yticklabels(sorted_topics, fontsize=10)
    ax.axvline(x=0, color="black", lw=1)
    ax.set_xlabel("Bias Direction (N-L gap − N-R gap, °)", fontsize=12)
    ax.set_title(f"Per-Topic Bias Direction in Political Layers [{pol_lower}-{pol_upper}]\n"
                 f"{model_short}", fontsize=14, fontweight="bold")

    ax.text(0.02, 0.98, "← LEFT lean", transform=ax.transAxes,
            fontsize=10, color="blue", va="top")
    ax.text(0.98, 0.98, "RIGHT lean →", transform=ax.transAxes,
            fontsize=10, color="red", va="top", ha="right")

    plt.tight_layout()
    p2 = output_dir / "step3_topic_bias_direction.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p2}")

    # ========================================
    # Figure 3: Content vs Style Decomposition
    # ========================================
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    layers = np.arange(style_res["total_layers"])
    ax.plot(layers, style_res["content_gaps"], "purple", lw=2.5,
            label="Content Gap (L vs R, same style)")
    ax.plot(layers, style_res["style_gaps"], "orange", lw=2.5,
            label="Style Gap (academic vs activist, same content)")

    ax.axvspan(pol_lower, pol_upper, alpha=0.15, color="green",
               label=f"Political Layers [{pol_lower}-{pol_upper}]")

    # 在 political layers 内标注比值
    ratio = style_res["ratio"]
    mid = (pol_lower + pol_upper) // 2
    ax.annotate(f"Content/Style = {ratio:.1f}x\nin political layers",
                xy=(mid, max(style_res["content_gaps"][mid], style_res["style_gaps"][mid])),
                xytext=(mid + 5, max(style_res["content_gaps"][mid], style_res["style_gaps"][mid]) + 1),
                arrowprops=dict(arrowstyle="->"), fontsize=11, fontweight="bold")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Angular Gap (°)", fontsize=12)
    ax.set_title(f"Content vs Style Decomposition — {model_short}\n"
                 f"(Does the gap encode stance or tone?)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p3 = output_dir / "step3_content_vs_style.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p3}")

    return [p1, p2, p3]


# ============================================================
# 主程序
# ============================================================

def main():
    args = parse_args()
    output_dir = Path("./results_step3")

    print("=" * 60)
    print("  Step 3: Topic-Level Fine-Grained Analysis")
    print("  (Requires Step 1 results)")
    print("=" * 60)

    pol_lower, pol_upper = load_political_layers(args.step1_dir)

    model, tokenizer, device, num_layers = load_model_and_tokenizer(
        args.model, quantize=False, device=args.device
    )

    # Analysis 1: Topic × Layer heatmap
    heatmap_res = topic_layer_heatmap(
        model, tokenizer, device, num_layers,
        pol_lower, pol_upper, seed=args.seed
    )

    # Analysis 2: Content vs Style
    style_res = content_style_decomposition(
        model, tokenizer, device, num_layers,
        pol_lower, pol_upper, seed=args.seed
    )

    # 可视化
    print("\n  Generating visualizations...")
    plot_step3_results(heatmap_res, style_res, args.model, output_dir,
                       pol_lower, pol_upper)

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "step3_results.npz",
        topics=heatmap_res["topics"],
        lr_gap_matrix=heatmap_res["lr_gap_matrix"],
        bias_matrix=heatmap_res["bias_matrix"],
        topic_pol_strength=heatmap_res["topic_pol_strength"],
        content_gaps=style_res["content_gaps"],
        style_gaps=style_res["style_gaps"],
        content_style_ratio=style_res["ratio"],
    )

    print(f"\n{'='*60}")
    print(f"  Step 3 Complete!")
    print(f"  Content/Style ratio: {style_res['ratio']:.2f}x")
    print(f"  Results saved to: {output_dir.absolute()}")
    print(f"")
    print(f"  Next: python step4_steering.py --model {args.model}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
