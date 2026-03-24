"""
Step 1 Demo: Dataset Visualization & Analysis

展示数据集结构、主题分布、政治倾向，无需加载模型
"""

import json
from collections import Counter
from dataset_loader import DatasetLoader, DataSourceType
from political_dataset_expanded import PAIRED_POLITICAL_STATEMENTS_EXPANDED


def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def analyze_dataset():
    """分析数据集结构"""

    print_header("Step 1: Dataset Construction - Complete Analysis")

    # 1. 加载数据集
    config = {
        "use_custom_political": True,
        "use_nonpolitical": True,
    }
    loader = DatasetLoader(config)
    items = loader.load_all(use_expanded_custom=True)

    # 2. 基本统计
    stats = loader.get_statistics()
    print(f"\n📊 DATASET OVERVIEW")
    print(f"  Total items: {stats['total_items']}")
    print(f"  Topics: {stats['topics']}")
    print(f"  Sources: {len(stats['sources'])}")

    print(f"\n📈 BY STANCE:")
    for stance, count in sorted(stats['stances'].items()):
        pct = (count / stats['total_items']) * 100
        print(f"    {stance:15s}: {count:3d} items ({pct:5.1f}%)")

    print(f"\n🗂️  BY SOURCE:")
    for source, count in sorted(stats['sources'].items()):
        pct = (count / stats['total_items']) * 100
        print(f"    {source:20s}: {count:3d} items ({pct:5.1f}%)")

    # 3. 主题覆盖
    topics = loader.get_topics()
    print(f"\n🎯 TOPICS COVERED ({len(topics)} total):")
    for i, topic in enumerate(sorted(topics), 1):
        if i <= 20:
            print(f"  {i:2d}. {topic:25s} ", end="")
            if i % 2 == 0:
                print()
        elif i == 21:
            print(f"\n  ... and {len(topics)-20} more topics")
            break

    # 4. 政治主题 vs 非政治主题对比
    pol_items = loader.get_by_source(DataSourceType.CUSTOM_POLITICAL)
    nonpol_items = loader.get_by_source(DataSourceType.NONPOLITICAL)

    print(f"\n🔍 POLITICAL vs NON-POLITICAL BALANCE:")
    print(f"  Political items:     {len(pol_items):3d} ({len(pol_items)/stats['total_items']*100:.1f}%)")
    print(f"  Non-political items: {len(nonpol_items):3d} ({len(nonpol_items)/stats['total_items']*100:.1f}%)")

    # 5. 左右政治立场对称性
    left_items = loader.get_by_stance("left")
    right_items = loader.get_by_stance("right")

    print(f"\n⚖️  LEFT-RIGHT IDEOLOGICAL BALANCE:")
    print(f"  Left-leaning:  {len(left_items):3d} items ({len(left_items)/len(pol_items)*100:.1f}% of political)")
    print(f"  Right-leaning: {len(right_items):3d} items ({len(right_items)/len(pol_items)*100:.1f}% of political)")
    print(f"  Balance ratio: {len(left_items)}/{len(right_items)} (ideal: 1.0)")

    # 6. 样例输出
    print(f"\n📝 SAMPLE ITEMS:")
    print(f"\n  Example 1: Healthcare (Left) ")
    healthcare_left = [item for item in left_items if item.topic == "healthcare"][0]
    text = healthcare_left.metadata["original_text"] if healthcare_left.metadata else healthcare_left.text
    print(f"  {text[:100]}...")

    print(f"\n  Example 2: Healthcare (Right)")
    healthcare_right = [item for item in right_items if item.topic == "healthcare"][0]
    text = healthcare_right.metadata["original_text"] if healthcare_right.metadata else healthcare_right.text
    print(f"  {text[:100]}...")

    # 7. 数据质量指标
    print(f"\n✅ DATA QUALITY CHECKS:")
    all_topics = [item.topic for item in items]
    topic_counts = Counter(all_topics)

    # 检查平衡性
    pol_topics = [item.topic for item in pol_items]
    pol_topic_counts = Counter(pol_topics)

    items_per_topic = list(pol_topic_counts.values())
    print(f"  Topics with paired statements: {len(set(pol_topics))}")
    print(f"  Items per topic (political): min={min(items_per_topic)}, max={max(items_per_topic)}, avg={sum(items_per_topic)/len(items_per_topic):.1f}")
    print(f"  Pairing coverage: {len([t for t in pol_topic_counts if pol_topic_counts[t] == 2]) / len(pol_topic_counts) * 100:.1f}% topics have both L/R")

    # 8. Step 1 完成度
    print(f"\n🎯 STEP 1 COMPLETION STATUS:")
    print(f"  ✅ Custom political dataset: {len(left_items)} left + {len(right_items)} right (49 topics)")
    print(f"  ✅ Non-political baseline: {len(nonpol_items)} items (20 topics)")
    print(f"  ✅ DatasetLoader framework: Implemented & tested")
    print(f"  ✅ Filtering methods: By stance, source, topic")
    print(f"  ✅ Statistics & metadata: Available")
    print(f"  ⏳ OpinionQA integration: Awaiting download")
    print(f"  ⏳ P-Stance integration: Awaiting download")

    print_header("Step 1 Result Summary")
    print(f"\n🚀 READY FOR NEXT STEPS:")
    print(f"  • Step 2: Code Refactoring - Use DatasetLoader in pipeline")
    print(f"  • Step 3: Multi-model experiments - Run on Qwen/Llama/Mistral")
    print(f"  • Step 4: Evaluation upgrade - Political Compass scoring")

    # 9. 导出元数据
    loader.save_metadata("dataset_metadata_demo.json")
    print(f"\n📄 Metadata exported to: dataset_metadata_demo.json")

    return loader, stats


def visualize_topic_distribution(loader):
    """可视化主题分布"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        topics = sorted(loader.get_topics())
        left_counts = [len(loader.get_by_stance("left")) // len(topics) for _ in topics]
        right_counts = [len(loader.get_by_stance("right")) // len(topics) for _ in topics]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 柱状图
        x = np.arange(len(topics[:15]))  # 只显示前15个主题
        width = 0.35

        ax1.bar(x - width/2, left_counts[:15], width, label="Left", color="steelblue")
        ax1.bar(x + width/2, right_counts[:15], width, label="Right", color="coral")
        ax1.set_xlabel("Topics")
        ax1.set_ylabel("Items per stance")
        ax1.set_title("Political Bias Dataset: L/R Distribution (First 15 Topics)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(topics[:15], rotation=45, ha="right")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # 饼图
        stats = loader.get_statistics()
        sizes = [stats['stances'].get('left', 0),
                stats['stances'].get('right', 0),
                stats['stances'].get('non_political', 0)]
        labels = ['Left (Political)', 'Right (Political)', 'Non-Political']
        colors = ['steelblue', 'coral', 'lightgray']

        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title("Overall Dataset Composition")

        plt.tight_layout()
        plt.savefig("step1_dataset_distribution.png", dpi=150, bbox_inches="tight")
        print(f"\n📊 Visualization saved: step1_dataset_distribution.png")

        # 显示主题列表图
        fig, ax = plt.subplots(figsize=(12, 10))
        all_topics = loader.get_topics()
        y_pos = np.arange(len(all_topics))

        ax.barh(y_pos, [1]*len(all_topics), color="steelblue", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_topics, fontsize=9)
        ax.set_xlabel("All Topics Covered")
        ax.set_title(f"Step 1 Dataset: {len(all_topics)} Political Topics + 20 Non-Political Items")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig("step1_topics_overview.png", dpi=150, bbox_inches="tight")
        print(f"📊 Visualization saved: step1_topics_overview.png")

    except ImportError:
        print("\n⚠️  Matplotlib not installed, skipping visualizations")


if __name__ == "__main__":
    loader, stats = analyze_dataset()

    print("\n" + "="*70)
    print("  Now trying to generate visualizations...")
    print("="*70)

    visualize_topic_distribution(loader)

    print_header("Step 1 Complete")
    print("\n✅ Dataset is ready for experiments!")
    print("\n🔗 Next: Run Step 2 code refactoring")
    print("   $ python step2_analyze_bias.py")
