"""
Step 1: Locate Political Layers
================================
直接对标 Safety Layers (ICLR 2025) Section 3.3 & 3.4

Safety Layers 做的事:
  - normal vs malicious → 找到 safety layers
我们做的事:
  - political vs non-political → 找到 political layers

三个分析:
  1. Cosine Similarity Gap (与 Safety Layers Figure 1 & 2 对应)
  2. Weak Classifier Probing (与 Zhou et al. EMNLP 2024 Figure 3 对应)
  3. Parameter Scaling Localization (与 Safety Layers Section 3.4 对应)

用法:
    python step1_locate_political_layers.py --model Qwen/Qwen2.5-7B-Instruct
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
    angular_difference,
)
from political_dataset import get_prompt_template
from nonpolitical_dataset import (
    get_nonpolitical_statements,
    get_political_statements_mixed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


# ============================================================
# Analysis 1: Cosine Similarity Gap
# (对标 Safety Layers Section 3.2 & 3.3)
# ============================================================

def cosine_gap_analysis(model, tokenizer, device, num_layers,
                        num_rounds=500, seed=42):
    """
    三种配对:
      P-P: 两个不同的政治性语句
      NP-NP: 两个不同的非政治性语句
      P-NP: 一个政治性 + 一个非政治性

    与 Safety Layers 的 N-N / M-M / N-M 完全对应。
    """
    print("\n" + "=" * 60)
    print("  Analysis 1: Cosine Similarity Gap")
    print("  (Political vs Non-Political)")
    print("=" * 60)

    random.seed(seed)
    np.random.seed(seed)

    # 提取 hidden states
    pol_stmts = get_political_statements_mixed()
    nonpol_stmts = get_nonpolitical_statements()

    print(f"\n  Political statements: {len(pol_stmts)}")
    print(f"  Non-political statements: {len(nonpol_stmts)}")

    pol_hidden = {}  # key: f"{topic}_{direction}"
    nonpol_hidden = {}  # key: topic

    print("\n  Extracting hidden states...")
    for topic, stmt, direction in tqdm(pol_stmts, desc="  POLITICAL"):
        key = f"{topic}_{direction}"
        pol_hidden[key] = extract_hidden_states(
            model, tokenizer, get_prompt_template(stmt), device
        )

    for topic, stmt in tqdm(nonpol_stmts, desc="  NON-POLITICAL"):
        nonpol_hidden[topic] = extract_hidden_states(
            model, tokenizer, get_prompt_template(stmt), device
        )

    total_layers = num_layers + 1
    pol_keys = list(pol_hidden.keys())
    nonpol_keys = list(nonpol_hidden.keys())

    # 三种配对
    pp_sims = np.zeros((num_rounds, total_layers))
    nn_sims = np.zeros((num_rounds, total_layers))
    pn_sims = np.zeros((num_rounds, total_layers))

    print(f"\n  Computing cosine similarities ({num_rounds} rounds)...")
    for r in tqdm(range(num_rounds), desc="  Pairing"):
        # P-P
        k1, k2 = random.sample(pol_keys, 2)
        for layer in range(total_layers):
            pp_sims[r, layer] = cosine_similarity(
                pol_hidden[k1][layer], pol_hidden[k2][layer])

        # NP-NP
        k1, k2 = random.sample(nonpol_keys, 2)
        for layer in range(total_layers):
            nn_sims[r, layer] = cosine_similarity(
                nonpol_hidden[k1][layer], nonpol_hidden[k2][layer])

        # P-NP
        pk = random.choice(pol_keys)
        nk = random.choice(nonpol_keys)
        for layer in range(total_layers):
            pn_sims[r, layer] = cosine_similarity(
                pol_hidden[pk][layer], nonpol_hidden[nk][layer])

    # Angular gap
    pp_angles = np.degrees(np.arccos(np.clip(pp_sims, -1, 1)))
    nn_angles = np.degrees(np.arccos(np.clip(nn_sims, -1, 1)))
    pn_angles = np.degrees(np.arccos(np.clip(pn_sims, -1, 1)))

    same_angles = (pp_angles + nn_angles) / 2
    angular_gap = np.mean(pn_angles - same_angles, axis=0)

    # 找到 gap 开始出现的层 (political layers 的起始点)
    # 定义: angular gap 首次超过整体标准差的层
    gap_std = np.std(angular_gap[:5])  # 前几层的噪声水平
    gap_threshold = gap_std * 2
    onset_layer = None
    for i in range(total_layers):
        if angular_gap[i] > gap_threshold:
            onset_layer = i
            break

    max_gap_layer = np.argmax(angular_gap)

    print(f"\n  === COSINE GAP RESULTS ===")
    print(f"  Gap onset layer: {onset_layer}")
    print(f"  Max gap layer: {max_gap_layer} ({angular_gap[max_gap_layer]:.2f}°)")
    print(f"  Noise level (first 5 layers): {gap_std:.2f}°")

    return {
        "pp_mean": np.mean(pp_sims, axis=0),
        "pp_std": np.std(pp_sims, axis=0),
        "nn_mean": np.mean(nn_sims, axis=0),
        "nn_std": np.std(nn_sims, axis=0),
        "pn_mean": np.mean(pn_sims, axis=0),
        "pn_std": np.std(pn_sims, axis=0),
        "angular_gap": angular_gap,
        "onset_layer": onset_layer,
        "max_gap_layer": max_gap_layer,
        "num_layers": total_layers,
        # 保留 hidden states 供后续分析使用
        "_pol_hidden": pol_hidden,
        "_nonpol_hidden": nonpol_hidden,
    }


# ============================================================
# Analysis 2: Weak Classifier Probing
# (对标 Zhou et al. EMNLP 2024 Figure 3 & Table 2)
# ============================================================

def weak_classifier_probing(pol_hidden, nonpol_hidden, num_layers, seed=42):
    """
    在每层训练 SVM 和 MLP 分类 political vs non-political。
    与 Zhou et al. 用弱分类器分类 normal/malicious 完全对应。

    如果弱分类器从很早的层就能区分，说明模型在预训练阶段
    就已经学会了识别政治内容（类似 Zhou et al. 的发现：
    ethical concepts are learned during pre-training）。
    """
    print("\n" + "=" * 60)
    print("  Analysis 2: Weak Classifier Probing")
    print("  (SVM & MLP per layer)")
    print("=" * 60)

    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    np.random.seed(seed)
    total_layers = num_layers

    # 准备数据: 每个样本是某层的 hidden state 向量
    pol_keys = list(pol_hidden.keys())
    nonpol_keys = list(nonpol_hidden.keys())

    svm_scores = []
    mlp_scores = []

    for layer in tqdm(range(total_layers), desc="  Probing layers"):
        # 构建特征矩阵
        X = []
        y = []

        for key in pol_keys:
            vec = pol_hidden[key][layer].numpy()
            X.append(vec)
            y.append(1)  # political

        for key in nonpol_keys:
            vec = nonpol_hidden[key][layer].numpy()
            X.append(vec)  # non-political
            y.append(0)

        X = np.array(X)
        y = np.array(y)

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # SVM (linear kernel, weak classifier)
        svm = SVC(kernel="linear", C=1.0, max_iter=1000)
        svm_cv = cross_val_score(svm, X_scaled, y, cv=min(5, len(y)//2), scoring="accuracy")
        svm_scores.append(np.mean(svm_cv))

        # MLP (single hidden layer with 100 neurons, weak classifier)
        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500,
                            random_state=seed, early_stopping=True)
        mlp_cv = cross_val_score(mlp, X_scaled, y, cv=min(5, len(y)//2), scoring="accuracy")
        mlp_scores.append(np.mean(mlp_cv))

    svm_scores = np.array(svm_scores)
    mlp_scores = np.array(mlp_scores)

    # 找到分类准确率超过 90% 的最早层
    svm_90_layer = None
    mlp_90_layer = None
    for i in range(total_layers):
        if svm_90_layer is None and svm_scores[i] >= 0.90:
            svm_90_layer = i
        if mlp_90_layer is None and mlp_scores[i] >= 0.90:
            mlp_90_layer = i

    print(f"\n  === CLASSIFIER PROBING RESULTS ===")
    print(f"  {'Layer':>6} {'SVM Acc':>10} {'MLP Acc':>10}")
    print(f"  {'-'*28}")
    for i in range(total_layers):
        marker = " <-- 90%+" if (svm_scores[i] >= 0.90 and
                                  (i == svm_90_layer or i == mlp_90_layer)) else ""
        print(f"  {i:>6} {svm_scores[i]:>10.3f} {mlp_scores[i]:>10.3f}{marker}")

    print(f"\n  SVM reaches 90% at layer: {svm_90_layer}")
    print(f"  MLP reaches 90% at layer: {mlp_90_layer}")

    return {
        "svm_scores": svm_scores,
        "mlp_scores": mlp_scores,
        "svm_90_layer": svm_90_layer,
        "mlp_90_layer": mlp_90_layer,
    }


# ============================================================
# Analysis 3: Political Layer Boundary Estimation
# (综合 cosine gap + classifier 确定 political layers 范围)
# ============================================================

def estimate_political_layers(cosine_results, probe_results, num_layers):
    """
    综合两种方法确定 political layers 的范围 [lower, upper]。

    逻辑 (对标 Safety Layers Section 3.4):
    - Lower bound: cosine gap 开始出现 且 分类器准确率开始显著上升的层
    - Upper bound: gap 开始趋于平稳或下降的层
    """
    print("\n" + "=" * 60)
    print("  Analysis 3: Political Layer Boundary Estimation")
    print("=" * 60)

    angular_gap = cosine_results["angular_gap"]
    svm_scores = probe_results["svm_scores"]

    total_layers = len(angular_gap)

    # Lower bound: 两种方法给出的 onset 的较早者
    cosine_onset = cosine_results["onset_layer"]
    classifier_onset = probe_results["svm_90_layer"]

    if cosine_onset is not None and classifier_onset is not None:
        lower = min(cosine_onset, classifier_onset)
    elif cosine_onset is not None:
        lower = cosine_onset
    elif classifier_onset is not None:
        lower = classifier_onset
    else:
        lower = total_layers // 4  # fallback

    # Upper bound: gap 从峰值开始下降到峰值 50% 以下的层
    max_gap = np.max(angular_gap)
    max_layer = np.argmax(angular_gap)
    upper = max_layer
    for i in range(max_layer, total_layers):
        if angular_gap[i] < max_gap * 0.5:
            upper = i
            break
    else:
        upper = total_layers - 1

    # 确保 upper > lower
    if upper <= lower:
        upper = min(lower + (total_layers // 4), total_layers - 1)

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  POLITICAL LAYERS: [{lower}, {upper}]")
    print(f"  ║  (out of {total_layers} total layers)")
    print(f"  ╠══════════════════════════════════════╣")
    print(f"  ║  Cosine gap onset:    layer {cosine_onset}")
    print(f"  ║  Classifier onset:    layer {classifier_onset}")
    print(f"  ║  Max gap:             layer {max_layer} ({max_gap:.2f}°)")
    print(f"  ║  Estimated range:     {upper - lower + 1} layers")
    print(f"  ╚══════════════════════════════════════╝")

    return {
        "lower": lower,
        "upper": upper,
        "max_gap_layer": max_layer,
    }


# ============================================================
# 可视化
# ============================================================

def plot_political_layers(cosine_res, probe_res, boundary_res, model_name, output_dir):
    """生成 political layer 定位的可视化"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]
    layers = np.arange(cosine_res["num_layers"])

    # ========================================
    # Figure 1: 三条余弦相似度曲线
    # (对标 Safety Layers Figure 1)
    # ========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 上: 余弦相似度
    ax1.plot(layers, cosine_res["pp_mean"], "r-", lw=2, label="P-P (Political-Political)")
    ax1.plot(layers, cosine_res["nn_mean"], "b-", lw=2, label="NP-NP (NonPol-NonPol)")
    ax1.plot(layers, cosine_res["pn_mean"], "g-", lw=2, label="P-NP (Political-NonPol)")

    # 标注 political layers 范围
    lower, upper = boundary_res["lower"], boundary_res["upper"]
    ax1.axvspan(lower, upper, alpha=0.15, color="orange", label=f"Political Layers [{lower}-{upper}]")
    ax1.axvline(x=lower, color="orange", ls="--", alpha=0.7)
    ax1.axvline(x=upper, color="orange", ls="--", alpha=0.7)

    ax1.set_ylabel("Cosine Similarity", fontsize=13)
    ax1.set_title(f"Step 1: Locating Political Layers in {model_short}\n"
                  f"(Analogous to Safety Layers Figure 1 & 2)",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 下: Angular gap
    gap = cosine_res["angular_gap"]
    ax2.plot(layers, gap, "purple", lw=2.5, label="Angular Gap (P-NP vs same-class)")
    ax2.axvspan(lower, upper, alpha=0.15, color="orange")
    ax2.axvline(x=lower, color="orange", ls="--", alpha=0.7)
    ax2.axvline(x=upper, color="orange", ls="--", alpha=0.7)
    ax2.axhline(y=0, color="gray", ls="--", alpha=0.5)

    max_layer = np.argmax(gap)
    ax2.annotate(f"Peak: {gap[max_layer]:.1f}° at L{max_layer}",
                 xy=(max_layer, gap[max_layer]),
                 xytext=(max_layer + 3, gap[max_layer] + 0.5),
                 arrowprops=dict(arrowstyle="->", color="red"),
                 fontsize=11, color="red", fontweight="bold")

    ax2.set_xlabel("Layer Index", fontsize=13)
    ax2.set_ylabel("Angular Difference (°)", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p1 = output_dir / "step1_cosine_gap.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p1}")

    # ========================================
    # Figure 2: 弱分类器准确率曲线
    # (对标 Zhou et al. Figure 3)
    # ========================================
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    ax.plot(layers[:len(probe_res["svm_scores"])], probe_res["svm_scores"],
            "blue", lw=2.5, marker="o", markersize=4, label="SVM (Linear)")
    ax.plot(layers[:len(probe_res["mlp_scores"])], probe_res["mlp_scores"],
            "red", lw=2.5, marker="s", markersize=4, label="MLP (100 neurons)")

    ax.axhline(y=0.9, color="gray", ls="--", alpha=0.5, label="90% threshold")
    ax.axhline(y=0.5, color="lightgray", ls=":", alpha=0.5, label="Chance level")
    ax.axvspan(lower, upper, alpha=0.15, color="orange", label=f"Political Layers [{lower}-{upper}]")

    ax.set_xlabel("Layer Index", fontsize=13)
    ax.set_ylabel("Classification Accuracy", fontsize=13)
    ax.set_title(f"Weak Classifier Probing: Can Each Layer Distinguish Political Content?\n"
                 f"{model_short} (Analogous to Zhou et al. EMNLP 2024 Figure 3)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

    plt.tight_layout()
    p2 = output_dir / "step1_classifier_probing.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p2}")

    # ========================================
    # Figure 3: 综合定位图
    # ========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 上: Angular gap (归一化到 0-1)
    gap_norm = (gap - np.min(gap)) / (np.max(gap) - np.min(gap) + 1e-8)
    ax1.fill_between(layers, 0, gap_norm, alpha=0.3, color="purple")
    ax1.plot(layers, gap_norm, "purple", lw=2, label="Angular Gap (normalized)")

    ax1.axvspan(lower, upper, alpha=0.2, color="orange")
    ax1.set_ylabel("Normalized Gap", fontsize=12)
    ax1.set_title(f"Political Layer Localization Summary — {model_short}",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 下: 分类器准确率
    avg_acc = (probe_res["svm_scores"] + probe_res["mlp_scores"]) / 2
    ax2.fill_between(layers[:len(avg_acc)], 0.5, avg_acc, alpha=0.3, color="blue")
    ax2.plot(layers[:len(avg_acc)], avg_acc, "blue", lw=2,
             label="Avg Classifier Accuracy")

    ax2.axvspan(lower, upper, alpha=0.2, color="orange",
                label=f"Political Layers [{lower}-{upper}]")
    ax2.axhline(y=0.9, color="gray", ls="--", alpha=0.5)

    ax2.set_xlabel("Layer Index", fontsize=13)
    ax2.set_ylabel("Classification Accuracy", fontsize=12)
    ax2.set_ylim(0.4, 1.05)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p3 = output_dir / "step1_localization_summary.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p3}")

    return [p1, p2, p3]


# ============================================================
# 主程序
# ============================================================

def main():
    args = parse_args()
    output_dir = Path("./results_step1")

    print("=" * 60)
    print("  Step 1: Locate Political Layers")
    print("  (Analogous to Safety Layers ICLR 2025)")
    print("=" * 60)

    # 加载模型
    model, tokenizer, device, num_layers = load_model_and_tokenizer(
        args.model, quantize=False, device=args.device
    )

    # Analysis 1: Cosine Gap
    cosine_res = cosine_gap_analysis(
        model, tokenizer, device, num_layers,
        num_rounds=args.num_rounds, seed=args.seed
    )

    # Analysis 2: Weak Classifier Probing
    probe_res = weak_classifier_probing(
        cosine_res["_pol_hidden"],
        cosine_res["_nonpol_hidden"],
        cosine_res["num_layers"],
        seed=args.seed
    )

    # Analysis 3: Boundary Estimation
    boundary_res = estimate_political_layers(cosine_res, probe_res, num_layers)

    # 可视化
    print("\n  Generating visualizations...")
    plot_political_layers(cosine_res, probe_res, boundary_res, args.model, output_dir)

    # 保存结果 (供 Step 2 使用)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "political_layers.npz",
        political_layer_lower=boundary_res["lower"],
        political_layer_upper=boundary_res["upper"],
        angular_gap=cosine_res["angular_gap"],
        svm_scores=probe_res["svm_scores"],
        mlp_scores=probe_res["mlp_scores"],
        pp_mean=cosine_res["pp_mean"],
        nn_mean=cosine_res["nn_mean"],
        pn_mean=cosine_res["pn_mean"],
    )

    print(f"\n{'='*60}")
    print(f"  Step 1 Complete!")
    print(f"  Political Layers: [{boundary_res['lower']}, {boundary_res['upper']}]")
    print(f"  Results saved to: {output_dir.absolute()}")
    print(f"")
    print(f"  Next: python step2_analyze_bias.py --model {args.model}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
