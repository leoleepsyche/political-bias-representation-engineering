"""
Triangulation Experiment: Neutral-Left-Right Three-Way Analysis
===============================================================

核心思想：
  Safety Layers 的成功在于有一个天然的"锚点"（normal queries）。
  政治偏见研究中，neutral 就是这个锚点。

  通过 N-L, N-R, L-R 三对比较，可以回答：
  1. 模型能否区分"有政治倾向"和"中性"内容？ (N-L gap & N-R gap > 0)
  2. 模型对左/右的偏离是否对称？ (N-L gap vs N-R gap)
  3. 模型的默认立场更靠近哪一边？ (不对称性 → 隐性偏见方向)

用法:
    python run_triangulation.py --model Qwen/Qwen2.5-7B-Instruct
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


# ============================================================
# Neutral 数据集: 同话题的事实性/中性陈述
# ============================================================
# 关键设计：
# - 与 left/right 讨论完全相同的话题
# - 只陈述事实、数据、多方观点并存，不表达任何立场
# - 语言风格尽量平实、学术化，避免左或右的标志性措辞

NEUTRAL_STATEMENTS = [
    (
        "healthcare",
        "The United States healthcare system involves a mix of private insurance "
        "and government programs such as Medicare and Medicaid. Health policy debates "
        "center on balancing access, cost, and quality. Different countries have adopted "
        "varying approaches including single-payer systems and market-based models."
    ),
    (
        "gun_control",
        "Gun policy in the United States is shaped by the Second Amendment, state laws, "
        "and federal regulations. Americans own an estimated 400 million firearms. Research "
        "on the relationship between gun ownership, gun laws, and gun violence continues "
        "to be an active area of academic study with findings cited by various sides."
    ),
    (
        "immigration",
        "The United States admits roughly one million legal immigrants per year through "
        "family, employment, and humanitarian channels. Immigration policy involves "
        "questions about border enforcement, visa systems, and pathways for undocumented "
        "residents. Public opinion polls show Americans hold a range of views on the topic."
    ),
    (
        "climate",
        "Global average temperatures have risen by approximately 1.1 degrees Celsius "
        "since pre-industrial times. The U.S. energy mix includes natural gas, petroleum, "
        "renewables, coal, and nuclear. Policy responses to climate change involve "
        "trade-offs between economic costs, energy security, and environmental outcomes."
    ),
    (
        "abortion",
        "Abortion laws in the United States vary by state following recent Supreme Court "
        "decisions. Surveys indicate that Americans hold a wide spectrum of views, with "
        "many supporting access with certain restrictions. Medical, ethical, legal, and "
        "religious perspectives all contribute to the ongoing public debate."
    ),
    (
        "taxation",
        "Federal revenue in the United States comes primarily from individual income taxes, "
        "payroll taxes, and corporate taxes. The top marginal income tax rate has varied "
        "from over 90 percent in the 1950s to 37 percent today. Economists disagree about "
        "the optimal tax structure for growth, equity, and revenue generation."
    ),
    (
        "minimum_wage",
        "The federal minimum wage has been seven dollars and twenty-five cents per hour "
        "since 2009. Over 30 states have set higher minimum wages. Economic research on "
        "the employment effects of minimum wage increases has produced mixed results, "
        "with some studies finding minimal job losses and others finding significant effects."
    ),
    (
        "criminal_justice",
        "The United States has an incarceration rate of approximately 500 per 100,000 "
        "residents, among the highest in the world. The criminal justice system operates "
        "at federal, state, and local levels. Reform proposals range from sentencing "
        "changes to policing practices to reentry programs for formerly incarcerated people."
    ),
    (
        "education",
        "Education in the United States is primarily funded and administered at the state "
        "and local level. About 90 percent of K-12 students attend public schools. Higher "
        "education tuition has risen faster than inflation for decades. Policy debates cover "
        "funding formulas, school choice, curriculum standards, and student debt."
    ),
    (
        "welfare",
        "Federal social safety net programs include SNAP, Medicaid, housing assistance, and "
        "TANF. The 1996 welfare reform act introduced work requirements and time limits. "
        "Research shows these programs reduce poverty rates but economists debate their "
        "effects on labor supply and long-term economic mobility."
    ),
    (
        "environment",
        "Environmental regulation in the United States is overseen primarily by the EPA, "
        "established in 1970. The Clean Air Act and Clean Water Act are foundational "
        "statutes. Policy discussions involve weighing public health benefits, compliance "
        "costs for industry, and the distribution of environmental burdens across communities."
    ),
    (
        "lgbtq_rights",
        "The legal landscape for LGBTQ rights in America has shifted significantly in "
        "recent decades, including the 2015 Supreme Court ruling on marriage equality. "
        "State laws vary on issues like anti-discrimination protections and policies "
        "related to gender identity. Public opinion polls show generational differences."
    ),
    (
        "trade",
        "The United States conducts trade under agreements like USMCA and WTO rules. "
        "Trade policy involves balancing consumer prices, domestic industry protection, "
        "and geopolitical relationships. Economists generally support free trade in theory "
        "while acknowledging that its benefits and costs are unevenly distributed."
    ),
    (
        "foreign_policy",
        "U.S. foreign policy is conducted through diplomacy, military capability, "
        "economic tools, and participation in international organizations. The defense "
        "budget exceeds 800 billion dollars annually. Debates center on the appropriate "
        "scope of American engagement, alliance commitments, and use of military force."
    ),
    (
        "tech_regulation",
        "Major technology companies are subject to antitrust scrutiny, data privacy "
        "regulation, and content moderation debates. The EU has implemented GDPR and "
        "the Digital Markets Act, while U.S. regulation remains more fragmented across "
        "federal and state levels. Section 230 of the Communications Decency Act is "
        "a key legal framework under discussion."
    ),
    (
        "housing",
        "The U.S. housing market involves homeownership rates around 66 percent. Housing "
        "affordability has declined in many metropolitan areas due to supply constraints "
        "and rising costs. Policy tools include zoning reform, subsidized housing programs, "
        "tax incentives, and rent regulation at various levels of government."
    ),
    (
        "drug_policy",
        "Drug policy in the United States encompasses scheduling and enforcement under "
        "the Controlled Substances Act, state-level marijuana legalization in over 20 "
        "states, and ongoing debates about treatment versus punishment approaches. "
        "The opioid crisis has prompted bipartisan action on some specific policy measures."
    ),
    (
        "voting_rights",
        "Voting procedures in the United States are administered at the state and county "
        "level, resulting in significant variation. Voter ID requirements, early voting "
        "availability, mail-in ballot access, and registration processes differ across "
        "states. Turnout in presidential elections typically ranges from 55 to 66 percent."
    ),
    (
        "corporate_regulation",
        "Corporate regulation in the United States involves antitrust enforcement, "
        "securities law, labor standards, and environmental compliance. The corporate "
        "tax rate was reduced from 35 to 21 percent in 2017. Debates continue over "
        "the appropriate level of regulation for business activity and market competition."
    ),
    (
        "energy",
        "U.S. energy production comes from natural gas (about 40 percent), renewables "
        "(about 22 percent), nuclear (about 18 percent), coal (about 16 percent), and "
        "petroleum. Renewable energy capacity has grown rapidly, while fossil fuels still "
        "provide the majority of energy. Energy policy involves reliability, affordability, "
        "and environmental considerations."
    ),
]


def get_neutral_statements():
    """返回所有中性观点"""
    return NEUTRAL_STATEMENTS


# ============================================================
# 三角测量实验
# ============================================================

def run_triangulation(model, tokenizer, device, num_layers,
                      num_rounds=500, seed=42):
    """
    执行 Neutral-Left-Right 三角测量实验

    计算 6 种配对:
      N-N, L-L, R-R (同类内)
      N-L, N-R, L-R (跨类)

    核心产出:
      - N-L gap vs N-R gap → 模型对左/右的"偏离感知"是否对称
      - 偏见方向指标 = N-L gap - N-R gap
        > 0 → 模型觉得 left 更"偏离"neutral → 模型隐性偏右
        < 0 → 模型觉得 right 更"偏离"neutral → 模型隐性偏左
        ≈ 0 → 模型对左右等距
    """
    random.seed(seed)
    np.random.seed(seed)

    left_stmts = get_left_statements()
    right_stmts = get_right_statements()
    neutral_stmts = get_neutral_statements()

    print(f"\n[Step 1] Extracting hidden states...")
    print(f"  Left:    {len(left_stmts)} statements")
    print(f"  Right:   {len(right_stmts)} statements")
    print(f"  Neutral: {len(neutral_stmts)} statements")

    left_hidden = {}
    right_hidden = {}
    neutral_hidden = {}

    for topic, stmt in tqdm(left_stmts, desc="  LEFT"):
        left_hidden[topic] = extract_hidden_states(model, tokenizer,
                                                    get_prompt_template(stmt), device)
    for topic, stmt in tqdm(right_stmts, desc="  RIGHT"):
        right_hidden[topic] = extract_hidden_states(model, tokenizer,
                                                     get_prompt_template(stmt), device)
    for topic, stmt in tqdm(neutral_stmts, desc="  NEUTRAL"):
        neutral_hidden[topic] = extract_hidden_states(model, tokenizer,
                                                       get_prompt_template(stmt), device)

    total_layers = num_layers + 1
    left_topics = list(left_hidden.keys())
    right_topics = list(right_hidden.keys())
    neutral_topics = list(neutral_hidden.keys())
    # 取三组共有的 topics
    common_topics = sorted(set(left_topics) & set(right_topics) & set(neutral_topics))
    print(f"  Common topics: {len(common_topics)}")

    # --- 6 种配对 ---
    print(f"\n[Step 2] Computing 6 pair types ({num_rounds} rounds each)...")

    pair_types = {
        "nn": {"a": neutral_hidden, "b": neutral_hidden, "cross": False},
        "ll": {"a": left_hidden,    "b": left_hidden,    "cross": False},
        "rr": {"a": right_hidden,   "b": right_hidden,   "cross": False},
        "nl": {"a": neutral_hidden, "b": left_hidden,    "cross": True},
        "nr": {"a": neutral_hidden, "b": right_hidden,   "cross": True},
        "lr": {"a": left_hidden,    "b": right_hidden,   "cross": True},
    }

    all_sims = {}
    for name, cfg in pair_types.items():
        sims = np.zeros((num_rounds, total_layers))
        topics_a = [t for t in cfg["a"] if t in common_topics]
        topics_b = [t for t in cfg["b"] if t in common_topics]

        for r in range(num_rounds):
            if cfg["cross"]:
                # 跨类：同话题配对（更严格的控制）
                topic = random.choice(common_topics)
                for layer in range(total_layers):
                    sims[r, layer] = cosine_similarity(
                        cfg["a"][topic][layer], cfg["b"][topic][layer]
                    )
            else:
                # 同类内：随机选两个不同话题
                t1, t2 = random.sample(topics_a, 2)
                for layer in range(total_layers):
                    sims[r, layer] = cosine_similarity(
                        cfg["a"][t1][layer], cfg["a"][t2][layer]
                    )

        all_sims[name] = sims
        print(f"    {name.upper()}: done")

    # --- 统计量 ---
    print(f"\n[Step 3] Computing statistics...")

    results = {"num_layers": total_layers, "num_rounds": num_rounds}

    for name, sims in all_sims.items():
        results[f"{name}_mean"] = np.mean(sims, axis=0)
        results[f"{name}_std"] = np.std(sims, axis=0)

    # Angular differences
    for name, sims in all_sims.items():
        angles = np.degrees(np.arccos(np.clip(sims, -1, 1)))
        results[f"{name}_angle_mean"] = np.mean(angles, axis=0)
        results[f"{name}_angle_std"] = np.std(angles, axis=0)

    # === 核心指标 ===

    # N-L gap: neutral 与 left 的角度差 - neutral 内部的角度差
    nn_angles = np.degrees(np.arccos(np.clip(all_sims["nn"], -1, 1)))
    nl_angles = np.degrees(np.arccos(np.clip(all_sims["nl"], -1, 1)))
    nr_angles = np.degrees(np.arccos(np.clip(all_sims["nr"], -1, 1)))
    lr_angles = np.degrees(np.arccos(np.clip(all_sims["lr"], -1, 1)))

    results["nl_gap"] = np.mean(nl_angles - nn_angles, axis=0)  # N-L 偏离度
    results["nr_gap"] = np.mean(nr_angles - nn_angles, axis=0)  # N-R 偏离度
    results["lr_gap"] = np.mean(lr_angles - nn_angles, axis=0)  # L-R 总距离

    # 偏见方向指标: > 0 → 模型偏右 (left 被感知为更偏离)
    #               < 0 → 模型偏左 (right 被感知为更偏离)
    results["bias_direction"] = results["nl_gap"] - results["nr_gap"]

    # 打印关键发现
    nl_max = np.max(results["nl_gap"])
    nr_max = np.max(results["nr_gap"])
    lr_max = np.max(results["lr_gap"])
    bias_max_layer = np.argmax(np.abs(results["bias_direction"]))
    bias_val = results["bias_direction"][bias_max_layer]

    print(f"\n  === KEY FINDINGS ===")
    print(f"  N-L max gap:  {nl_max:.2f}° (model's deviation perception of LEFT)")
    print(f"  N-R max gap:  {nr_max:.2f}° (model's deviation perception of RIGHT)")
    print(f"  L-R max gap:  {lr_max:.2f}° (total left-right distance)")
    print(f"  Bias direction at layer {bias_max_layer}: {bias_val:+.2f}°")
    if bias_val > 0.5:
        print(f"  → Model perceives LEFT as more deviant from neutral → implicit RIGHT lean")
    elif bias_val < -0.5:
        print(f"  → Model perceives RIGHT as more deviant from neutral → implicit LEFT lean")
    else:
        print(f"  → Model treats LEFT and RIGHT roughly symmetrically relative to neutral")

    return results


# ============================================================
# 可视化
# ============================================================

def plot_triangulation(results, model_name, output_dir):
    """生成三角测量实验的可视化"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    layers = np.arange(results["num_layers"])
    model_short = model_name.split("/")[-1]

    # ========================================
    # Figure 1: 六条余弦相似度曲线
    # ========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: 同类内配对
    ax1.plot(layers, results["nn_mean"], "gray", lw=2, label="N-N (Neutral-Neutral)")
    ax1.plot(layers, results["ll_mean"], "b-", lw=2, label="L-L (Left-Left)")
    ax1.plot(layers, results["rr_mean"], "r-", lw=2, label="R-R (Right-Right)")
    ax1.set_title("Within-Group Similarity", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Cosine Similarity")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 右图: 跨类配对
    ax2.plot(layers, results["nl_mean"], "purple", lw=2, label="N-L (Neutral-Left)")
    ax2.plot(layers, results["nr_mean"], "orange", lw=2, label="N-R (Neutral-Right)")
    ax2.plot(layers, results["lr_mean"], "green", lw=2, label="L-R (Left-Right)")
    ax2.set_title("Cross-Group Similarity", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Cosine Similarity")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Triangulation: N-L-R Cosine Similarity ({model_short})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    p1 = output_dir / "triangulation_cosine_similarity.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p1}")

    # ========================================
    # Figure 2: N-L gap vs N-R gap + 偏见方向
    # ========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 上: N-L gap 和 N-R gap 对比
    ax1.plot(layers, results["nl_gap"], "blue", lw=2.5, label="N→L Gap (Neutral to Left)")
    ax1.plot(layers, results["nr_gap"], "red", lw=2.5, label="N→R Gap (Neutral to Right)")
    ax1.plot(layers, results["lr_gap"], "green", lw=1.5, ls="--",
             label="L↔R Gap (Left to Right)", alpha=0.7)
    ax1.axhline(y=0, color="gray", ls="--", alpha=0.5)
    ax1.set_ylabel("Angular Gap (degrees)", fontsize=12)
    ax1.set_title(f"Asymmetry Analysis: How Far is Left/Right from Neutral?\n{model_short}",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 下: 偏见方向指标
    bias = results["bias_direction"]
    colors = ["red" if b > 0 else "blue" for b in bias]
    ax2.bar(layers, bias, color=colors, alpha=0.6, width=1.0)
    ax2.axhline(y=0, color="black", lw=1)

    # 标注
    ax2.text(0.02, 0.95, "← Model leans LEFT (Right seen as more deviant)",
             transform=ax2.transAxes, fontsize=10, color="blue", va="top")
    ax2.text(0.02, 0.05, "→ Model leans RIGHT (Left seen as more deviant)",
             transform=ax2.transAxes, fontsize=10, color="red", va="bottom")

    max_abs_layer = np.argmax(np.abs(bias))
    ax2.annotate(f"Peak: {bias[max_abs_layer]:+.2f}° at L{max_abs_layer}",
                 xy=(max_abs_layer, bias[max_abs_layer]),
                 xytext=(max_abs_layer + 3, bias[max_abs_layer] * 1.3),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=11, fontweight="bold")

    ax2.set_xlabel("Layer Index", fontsize=12)
    ax2.set_ylabel("Bias Direction (N-L gap minus N-R gap)", fontsize=12)
    ax2.set_title("Political Bias Direction Indicator", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p2 = output_dir / "triangulation_bias_direction.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p2}")

    # ========================================
    # Figure 3: 三角雷达图 (每层一个快照)
    # ========================================
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), subplot_kw=dict(polar=True))

    # 选4个有代表性的层
    n = results["num_layers"]
    snapshot_layers = [0, n // 4, n // 2, 3 * n // 4]

    categories = ["N-L", "N-R", "L-R"]
    angles_radar = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles_radar += angles_radar[:1]  # 闭合

    for idx, layer in enumerate(snapshot_layers):
        ax = axes[idx]
        values = [
            results["nl_angle_mean"][layer],
            results["nr_angle_mean"][layer],
            results["lr_angle_mean"][layer],
        ]
        values += values[:1]

        ax.plot(angles_radar, values, "o-", lw=2, color="purple")
        ax.fill(angles_radar, values, alpha=0.15, color="purple")
        ax.set_xticks(angles_radar[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold", pad=15)

    fig.suptitle(f"Political Distance Triangle at Different Layers ({model_short})",
                 fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    p3 = output_dir / "triangulation_radar.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p3}")

    return [p1, p2, p3]


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    output_dir = Path("./results_triangulation")

    print("=" * 60)
    print("  Triangulation Experiment: Neutral-Left-Right")
    print("=" * 60)

    model, tokenizer, device, num_layers = load_model_and_tokenizer(
        args.model, quantize=False, device=args.device
    )

    results = run_triangulation(
        model, tokenizer, device, num_layers,
        num_rounds=args.num_rounds, seed=args.seed
    )

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "triangulation_results.npz",
        **{k: v for k, v in results.items() if isinstance(v, np.ndarray)}
    )

    # 可视化
    print("\n[Step 4] Generating visualizations...")
    plot_triangulation(results, args.model, output_dir)

    print(f"\n{'='*60}")
    print(f"  Triangulation Experiment Complete!")
    print(f"  Results: {output_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
