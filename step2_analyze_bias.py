"""
Step 2: Analyze Left/Right/Neutral Bias WITHIN Political Layers
================================================================
前提: Step 1 已经找到了 political layers 的范围 [lower, upper]。
本步骤在这些层内进行 L/R/N 三角测量分析。

与 Safety Layers 的对比:
  Safety Layers:  先找 safety layers → 再在这些层做 normal/malicious 分析
  我们:           先找 political layers → 再在这些层做 Left/Right/Neutral 分析

三个核心分析:
  1. Cosine gap 三角测量 (N-L, N-R, L-R) — 聚焦于 political layers
  2. Bias direction 指标 (N-L gap - N-R gap) — 衡量模型偏向
  3. Weak classifier probing (L vs R vs N) — 确认层级特异性

用法:
    python step2_analyze_bias.py --model Qwen/Qwen2.5-7B-Instruct
    # 或指定 Step 1 结果路径:
    python step2_analyze_bias.py --model Qwen/Qwen2.5-7B-Instruct --step1_dir ./results_step1
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--step1_dir", type=str, default="./results_step1",
                        help="Directory containing Step 1 results (political_layers.npz)")
    return parser.parse_args()


# ============================================================
# 载入 Step 1 的 political layers 边界
# ============================================================

def load_political_layers(step1_dir):
    """从 Step 1 的结果中读取 political layers 范围"""
    path = Path(step1_dir) / "political_layers.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Step 1 results not found at {path}.\n"
            f"Please run step1_locate_political_layers.py first."
        )
    data = np.load(path)
    lower = int(data["political_layer_lower"])
    upper = int(data["political_layer_upper"])
    print(f"  Loaded political layers from Step 1: [{lower}, {upper}]")
    return lower, upper, data


# ============================================================
# Analysis 1: L/R/N 三角测量 (聚焦 political layers)
# ============================================================

def triangulation_in_political_layers(model, tokenizer, device, num_layers,
                                       pol_lower, pol_upper,
                                       num_rounds=500, seed=42):
    """
    在 political layers 内做 Neutral-Left-Right 三角测量。
    也计算全层的结果用于对比。

    核心指标:
      bias_direction = (N-L gap) - (N-R gap)
      > 0 → 模型隐性偏右 (left 被感知为更偏离 neutral)
      < 0 → 模型隐性偏左 (right 被感知为更偏离 neutral)
    """
    print("\n" + "=" * 60)
    print("  Analysis 1: L/R/N Triangulation in Political Layers")
    print("=" * 60)

    random.seed(seed)
    np.random.seed(seed)

    left_stmts = get_left_statements()
    right_stmts = get_right_statements()
    neutral_stmts = get_neutral_statements()

    print(f"\n  Left: {len(left_stmts)}, Right: {len(right_stmts)}, Neutral: {len(neutral_stmts)}")

    # 提取 hidden states
    left_hidden = {}
    right_hidden = {}
    neutral_hidden = {}

    print("  Extracting hidden states...")
    for topic, stmt in tqdm(left_stmts, desc="  LEFT"):
        left_hidden[topic] = extract_hidden_states(
            model, tokenizer, get_prompt_template(stmt), device)
    for topic, stmt in tqdm(right_stmts, desc="  RIGHT"):
        right_hidden[topic] = extract_hidden_states(
            model, tokenizer, get_prompt_template(stmt), device)
    for topic, stmt in tqdm(neutral_stmts, desc="  NEUTRAL"):
        neutral_hidden[topic] = extract_hidden_states(
            model, tokenizer, get_prompt_template(stmt), device)

    total_layers = num_layers + 1
    common_topics = sorted(
        set(left_hidden.keys()) & set(right_hidden.keys()) & set(neutral_hidden.keys())
    )
    print(f"  Common topics: {len(common_topics)}")

    # 6 种配对的余弦相似度
    pair_configs = {
        "nn": (neutral_hidden, neutral_hidden, False),
        "ll": (left_hidden, left_hidden, False),
        "rr": (right_hidden, right_hidden, False),
        "nl": (neutral_hidden, left_hidden, True),
        "nr": (neutral_hidden, right_hidden, True),
        "lr": (left_hidden, right_hidden, True),
    }

    all_sims = {}
    for name, (dict_a, dict_b, cross) in pair_configs.items():
        sims = np.zeros((num_rounds, total_layers))
        for r in range(num_rounds):
            if cross:
                topic = random.choice(common_topics)
                for layer in range(total_layers):
                    sims[r, layer] = cosine_similarity(
                        dict_a[topic][layer], dict_b[topic][layer])
            else:
                t1, t2 = random.sample(common_topics, 2)
                for layer in range(total_layers):
                    sims[r, layer] = cosine_similarity(
                        dict_a[t1][layer], dict_a[t2][layer])
        all_sims[name] = sims

    # 转换为角度
    nn_angles = np.degrees(np.arccos(np.clip(all_sims["nn"], -1, 1)))
    nl_angles = np.degrees(np.arccos(np.clip(all_sims["nl"], -1, 1)))
    nr_angles = np.degrees(np.arccos(np.clip(all_sims["nr"], -1, 1)))
    lr_angles = np.degrees(np.arccos(np.clip(all_sims["lr"], -1, 1)))

    # Gap 指标
    nl_gap = np.mean(nl_angles - nn_angles, axis=0)
    nr_gap = np.mean(nr_angles - nn_angles, axis=0)
    lr_gap = np.mean(lr_angles - nn_angles, axis=0)
    bias_direction = nl_gap - nr_gap

    # 聚焦 political layers 范围的统计
    pol_range = range(pol_lower, pol_upper + 1)
    pol_bias_mean = np.mean(bias_direction[pol_lower:pol_upper + 1])
    pol_bias_std = np.std(bias_direction[pol_lower:pol_upper + 1])
    nonpol_bias_mean = np.mean(np.concatenate([
        bias_direction[:pol_lower],
        bias_direction[pol_upper + 1:]
    ])) if pol_lower > 0 or pol_upper < total_layers - 1 else 0

    print(f"\n  === TRIANGULATION RESULTS (Political Layers [{pol_lower}-{pol_upper}]) ===")
    print(f"  Bias direction in political layers: {pol_bias_mean:+.3f}° ± {pol_bias_std:.3f}°")
    print(f"  Bias direction outside:             {nonpol_bias_mean:+.3f}°")
    print(f"  Ratio (pol / nonpol):               {abs(pol_bias_mean / (nonpol_bias_mean + 1e-8)):.2f}x")

    if pol_bias_mean > 0.5:
        leaning = "RIGHT (Left seen as more deviant from Neutral)"
    elif pol_bias_mean < -0.5:
        leaning = "LEFT (Right seen as more deviant from Neutral)"
    else:
        leaning = "SYMMETRIC (Left and Right roughly equidistant from Neutral)"
    print(f"  Model leaning: {leaning}")

    results = {
        "total_layers": total_layers,
        "pol_lower": pol_lower, "pol_upper": pol_upper,
    }
    for name, sims in all_sims.items():
        results[f"{name}_mean"] = np.mean(sims, axis=0)
        results[f"{name}_std"] = np.std(sims, axis=0)
    results["nl_gap"] = nl_gap
    results["nr_gap"] = nr_gap
    results["lr_gap"] = lr_gap
    results["bias_direction"] = bias_direction
    results["pol_bias_mean"] = pol_bias_mean
    results["_left_hidden"] = left_hidden
    results["_right_hidden"] = right_hidden
    results["_neutral_hidden"] = neutral_hidden

    return results


# ============================================================
# Analysis 2: Weak Classifier — L vs R vs N (3-class)
# ============================================================

def three_class_probing(left_hidden, right_hidden, neutral_hidden,
                         total_layers, pol_lower, pol_upper, seed=42):
    """
    在每层训练 3-class 分类器 (Left / Right / Neutral)。

    关键假设 (对标 Zhou et al.):
      如果 political layers 内的分类准确率 > 非政治层,
      说明这些层确实编码了政治立场信息。
    """
    print("\n" + "=" * 60)
    print("  Analysis 2: 3-Class Probing (Left / Right / Neutral)")
    print("=" * 60)

    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    np.random.seed(seed)
    common_topics = sorted(
        set(left_hidden.keys()) & set(right_hidden.keys()) & set(neutral_hidden.keys())
    )

    svm_scores = []
    mlp_scores = []

    for layer in tqdm(range(total_layers), desc="  3-class probing"):
        X = []
        y = []

        for topic in common_topics:
            X.append(left_hidden[topic][layer].numpy())
            y.append(0)  # left
        for topic in common_topics:
            X.append(right_hidden[topic][layer].numpy())
            y.append(1)  # right
        for topic in common_topics:
            X.append(neutral_hidden[topic][layer].numpy())
            y.append(2)  # neutral

        X = np.array(X)
        y = np.array(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_samples = len(y)
        cv_folds = min(5, n_samples // 3)  # 至少每 fold 3 个样本
        if cv_folds < 2:
            cv_folds = 2

        svm = SVC(kernel="linear", C=1.0, max_iter=2000, decision_function_shape="ovo")
        svm_cv = cross_val_score(svm, X_scaled, y, cv=cv_folds, scoring="accuracy")
        svm_scores.append(np.mean(svm_cv))

        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500,
                            random_state=seed, early_stopping=True)
        mlp_cv = cross_val_score(mlp, X_scaled, y, cv=cv_folds, scoring="accuracy")
        mlp_scores.append(np.mean(mlp_cv))

    svm_scores = np.array(svm_scores)
    mlp_scores = np.array(mlp_scores)

    # 对比 political layers 内外的分类准确率
    pol_svm = np.mean(svm_scores[pol_lower:pol_upper + 1])
    nonpol_svm = np.mean(np.concatenate([
        svm_scores[:pol_lower], svm_scores[pol_upper + 1:]
    ])) if pol_lower > 0 or pol_upper < total_layers - 1 else 0

    pol_mlp = np.mean(mlp_scores[pol_lower:pol_upper + 1])
    nonpol_mlp = np.mean(np.concatenate([
        mlp_scores[:pol_lower], mlp_scores[pol_upper + 1:]
    ])) if pol_lower > 0 or pol_upper < total_layers - 1 else 0

    print(f"\n  === 3-CLASS PROBING RESULTS ===")
    print(f"  {'Region':<25} {'SVM':>8} {'MLP':>8}")
    print(f"  {'-'*43}")
    print(f"  {'Political layers':.<25} {pol_svm:>8.3f} {pol_mlp:>8.3f}")
    print(f"  {'Non-political layers':.<25} {nonpol_svm:>8.3f} {nonpol_mlp:>8.3f}")
    print(f"  {'Difference (pol-nonpol)':.<25} {pol_svm - nonpol_svm:>+8.3f} {pol_mlp - nonpol_mlp:>+8.3f}")
    print(f"\n  Chance level (3-class): 33.3%")

    return {
        "svm_scores": svm_scores,
        "mlp_scores": mlp_scores,
        "pol_svm_mean": pol_svm,
        "nonpol_svm_mean": nonpol_svm,
        "pol_mlp_mean": pol_mlp,
        "nonpol_mlp_mean": nonpol_mlp,
    }


# ============================================================
# Analysis 3: Political Direction Vector
# ============================================================

def compute_political_direction_vector(left_hidden, right_hidden, neutral_hidden,
                                        common_topics, pol_lower, pol_upper):
    """
    在 political layers 内计算 政治方向向量。

    方法 (对标 CAA / RepE):
      direction_vector = mean(left_hidden) - mean(right_hidden)

    这个向量后续用于 Step 4 的 steering intervention。
    同时计算 neutral 在这个方向上的投影，看模型默认位置。
    """
    print("\n" + "=" * 60)
    print("  Analysis 3: Political Direction Vector")
    print("=" * 60)

    # 在 political layers 范围内平均
    pol_layers = list(range(pol_lower, pol_upper + 1))

    left_vecs = []
    right_vecs = []
    neutral_vecs = []

    for topic in common_topics:
        for layer in pol_layers:
            left_vecs.append(left_hidden[topic][layer].numpy())
            right_vecs.append(right_hidden[topic][layer].numpy())
            neutral_vecs.append(neutral_hidden[topic][layer].numpy())

    left_mean = np.mean(left_vecs, axis=0)
    right_mean = np.mean(right_vecs, axis=0)
    neutral_mean = np.mean(neutral_vecs, axis=0)

    # 政治方向向量: Left → Right
    direction = right_mean - left_mean
    direction_norm = direction / (np.linalg.norm(direction) + 1e-8)

    # 投影: neutral 在 left-right 轴上的位置
    # 0 = center, >0 = closer to right, <0 = closer to left
    center = (left_mean + right_mean) / 2
    neutral_proj = np.dot(neutral_mean - center, direction_norm)
    left_proj = np.dot(left_mean - center, direction_norm)
    right_proj = np.dot(right_mean - center, direction_norm)

    print(f"\n  Political direction vector computed over layers [{pol_lower}-{pol_upper}]")
    print(f"  Vector norm: {np.linalg.norm(direction):.4f}")
    print(f"  Projections onto L→R axis:")
    print(f"    Left mean:    {left_proj:+.4f}")
    print(f"    Right mean:   {right_proj:+.4f}")
    print(f"    Neutral mean: {neutral_proj:+.4f}")

    if neutral_proj > 0.01:
        print(f"  → Neutral leans toward RIGHT")
    elif neutral_proj < -0.01:
        print(f"  → Neutral leans toward LEFT")
    else:
        print(f"  → Neutral is approximately centered")

    # 逐层计算方向向量和投影 (更细粒度)
    per_layer_proj = {"left": [], "right": [], "neutral": []}
    per_layer_direction = []

    total_layers = len(left_hidden[common_topics[0]])
    for layer in range(total_layers):
        l_vecs = [left_hidden[t][layer].numpy() for t in common_topics]
        r_vecs = [right_hidden[t][layer].numpy() for t in common_topics]
        n_vecs = [neutral_hidden[t][layer].numpy() for t in common_topics]

        l_m = np.mean(l_vecs, axis=0)
        r_m = np.mean(r_vecs, axis=0)
        n_m = np.mean(n_vecs, axis=0)

        d = r_m - l_m
        d_norm = d / (np.linalg.norm(d) + 1e-8)
        c = (l_m + r_m) / 2

        per_layer_proj["left"].append(np.dot(l_m - c, d_norm))
        per_layer_proj["right"].append(np.dot(r_m - c, d_norm))
        per_layer_proj["neutral"].append(np.dot(n_m - c, d_norm))
        per_layer_direction.append(d)

    return {
        "direction_vector": direction,
        "direction_norm": direction_norm,
        "neutral_projection": neutral_proj,
        "per_layer_proj": per_layer_proj,
        "per_layer_direction": per_layer_direction,
        "pol_layers": pol_layers,
    }


# ============================================================
# 可视化
# ============================================================

def plot_step2_results(tri_res, probe_res, direction_res, model_name, output_dir):
    """生成 Step 2 的全部可视化"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]
    total_layers = tri_res["total_layers"]
    layers = np.arange(total_layers)
    pol_lower = tri_res["pol_lower"]
    pol_upper = tri_res["pol_upper"]

    # ========================================
    # Figure 1: Bias Direction (focused on political layers)
    # ========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 上: N-L gap vs N-R gap
    ax1.plot(layers, tri_res["nl_gap"], "b-", lw=2.5, label="N→L Gap")
    ax1.plot(layers, tri_res["nr_gap"], "r-", lw=2.5, label="N→R Gap")
    ax1.plot(layers, tri_res["lr_gap"], "g--", lw=1.5, alpha=0.6, label="L↔R Gap")
    ax1.axvspan(pol_lower, pol_upper, alpha=0.15, color="orange",
                label=f"Political Layers [{pol_lower}-{pol_upper}]")
    ax1.axhline(y=0, color="gray", ls="--", alpha=0.5)
    ax1.set_ylabel("Angular Gap (°)", fontsize=12)
    ax1.set_title(f"Step 2: L/R/N Triangulation in Political Layers — {model_short}",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 下: Bias direction
    bias = tri_res["bias_direction"]
    colors = ["red" if b > 0 else "blue" for b in bias]
    ax2.bar(layers, bias, color=colors, alpha=0.6, width=1.0)
    ax2.axvspan(pol_lower, pol_upper, alpha=0.1, color="orange")
    ax2.axhline(y=0, color="black", lw=1)

    # 标注 political layers 内的平均值
    pol_mean = tri_res["pol_bias_mean"]
    ax2.axhline(y=pol_mean, color="orange", ls=":", lw=2,
                label=f"Mean in pol. layers: {pol_mean:+.2f}°")

    ax2.text(0.02, 0.95, "← LEFT lean (Right more deviant)",
             transform=ax2.transAxes, fontsize=9, color="blue", va="top")
    ax2.text(0.02, 0.05, "→ RIGHT lean (Left more deviant)",
             transform=ax2.transAxes, fontsize=9, color="red", va="bottom")

    ax2.set_xlabel("Layer Index", fontsize=12)
    ax2.set_ylabel("Bias Direction (°)", fontsize=12)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p1 = output_dir / "step2_bias_direction.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p1}")

    # ========================================
    # Figure 2: 3-Class Probing Accuracy
    # ========================================
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    ax.plot(layers[:len(probe_res["svm_scores"])], probe_res["svm_scores"],
            "b-", lw=2.5, marker="o", markersize=3, label="SVM (Linear)")
    ax.plot(layers[:len(probe_res["mlp_scores"])], probe_res["mlp_scores"],
            "r-", lw=2.5, marker="s", markersize=3, label="MLP (100 neurons)")

    ax.axhline(y=1/3, color="lightgray", ls=":", alpha=0.5, label="Chance (33.3%)")
    ax.axvspan(pol_lower, pol_upper, alpha=0.15, color="orange",
               label=f"Political Layers [{pol_lower}-{pol_upper}]")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("3-Class Accuracy", fontsize=12)
    ax.set_title(f"3-Class Probing: Left vs Right vs Neutral — {model_short}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.2, 1.05)

    plt.tight_layout()
    p2 = output_dir / "step2_three_class_probing.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p2}")

    # ========================================
    # Figure 3: Neutral Projection on L→R Axis
    # ========================================
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    proj = direction_res["per_layer_proj"]
    ax.plot(layers[:len(proj["left"])], proj["left"], "b-", lw=2, label="Left")
    ax.plot(layers[:len(proj["right"])], proj["right"], "r-", lw=2, label="Right")
    ax.plot(layers[:len(proj["neutral"])], proj["neutral"], "gray", lw=2.5,
            ls="--", label="Neutral (model default)")

    ax.axvspan(pol_lower, pol_upper, alpha=0.15, color="orange",
               label=f"Political Layers [{pol_lower}-{pol_upper}]")
    ax.axhline(y=0, color="black", ls=":", alpha=0.3)

    ax.fill_between(layers[:len(proj["neutral"])],
                     proj["left"][:len(layers)], proj["right"][:len(layers)],
                     alpha=0.05, color="green")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Projection on L→R Axis", fontsize=12)
    ax.set_title(f"Where Does 'Neutral' Sit Between Left and Right? — {model_short}\n"
                 f"(Political Direction Vector per Layer)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p3 = output_dir / "step2_neutral_projection.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p3}")

    return [p1, p2, p3]


# ============================================================
# 主程序
# ============================================================

def main():
    args = parse_args()
    output_dir = Path("./results_step2")

    print("=" * 60)
    print("  Step 2: Analyze L/R/N Bias in Political Layers")
    print("  (Requires Step 1 results)")
    print("=" * 60)

    # 载入 Step 1 结果
    pol_lower, pol_upper, step1_data = load_political_layers(args.step1_dir)

    # 加载模型
    model, tokenizer, device, num_layers = load_model_and_tokenizer(
        args.model, quantize=False, device=args.device
    )

    # Analysis 1: 三角测量
    tri_res = triangulation_in_political_layers(
        model, tokenizer, device, num_layers,
        pol_lower, pol_upper,
        num_rounds=args.num_rounds, seed=args.seed
    )

    # Analysis 2: 3-class probing
    common_topics = sorted(
        set(tri_res["_left_hidden"].keys()) &
        set(tri_res["_right_hidden"].keys()) &
        set(tri_res["_neutral_hidden"].keys())
    )

    probe_res = three_class_probing(
        tri_res["_left_hidden"], tri_res["_right_hidden"], tri_res["_neutral_hidden"],
        tri_res["total_layers"], pol_lower, pol_upper, seed=args.seed
    )

    # Analysis 3: Direction vector
    direction_res = compute_political_direction_vector(
        tri_res["_left_hidden"], tri_res["_right_hidden"], tri_res["_neutral_hidden"],
        common_topics, pol_lower, pol_upper
    )

    # 可视化
    print("\n  Generating visualizations...")
    plot_step2_results(tri_res, probe_res, direction_res, args.model, output_dir)

    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "step2_results.npz",
        pol_lower=pol_lower,
        pol_upper=pol_upper,
        bias_direction=tri_res["bias_direction"],
        pol_bias_mean=tri_res["pol_bias_mean"],
        nl_gap=tri_res["nl_gap"],
        nr_gap=tri_res["nr_gap"],
        lr_gap=tri_res["lr_gap"],
        svm_3class=probe_res["svm_scores"],
        mlp_3class=probe_res["mlp_scores"],
        direction_vector=direction_res["direction_vector"],
        neutral_projection=direction_res["neutral_projection"],
    )

    # 保存方向向量 (供 Step 4 steering 使用)
    np.savez(
        output_dir / "direction_vectors.npz",
        direction_vector=direction_res["direction_vector"],
        direction_norm=direction_res["direction_norm"],
        per_layer_direction=np.array(direction_res["per_layer_direction"]),
        pol_lower=pol_lower,
        pol_upper=pol_upper,
    )

    print(f"\n{'='*60}")
    print(f"  Step 2 Complete!")
    print(f"  Bias direction (political layers): {tri_res['pol_bias_mean']:+.3f}°")
    print(f"  Neutral projection: {direction_res['neutral_projection']:+.4f}")
    print(f"  Results saved to: {output_dir.absolute()}")
    print(f"")
    print(f"  Next: python step3_topic_analysis.py --model {args.model}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
