"""
Quick Demo: 使用小模型快速验证实验流程
=======================================
如果你想先快速验证代码能跑通，可以用这个脚本。
它使用 Qwen2.5-1.5B-Instruct (约 3GB) 而不是 7B，几分钟内就能出结果。

使用方法:
    python run_quick_demo.py
"""

import sys
import os

# 将当前目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiment import (
    load_model_and_tokenizer,
    run_experiment,
    plot_results,
    OUTPUT_DIR,
)
import numpy as np
from pathlib import Path

DEMO_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # 小模型，快速验证
DEMO_ROUNDS = 200  # 减少轮数加速

def main():
    print("=" * 60)
    print("  QUICK DEMO - Using smaller model for fast validation")
    print(f"  Model: {DEMO_MODEL}")
    print(f"  Rounds: {DEMO_ROUNDS}")
    print("=" * 60)

    # 加载小模型
    model, tokenizer, device, num_layers = load_model_and_tokenizer(
        DEMO_MODEL, quantize=False, device="auto"
    )

    # 运行实验
    results = run_experiment(
        model, tokenizer, device, num_layers,
        num_rounds=DEMO_ROUNDS,
        seed=42
    )

    # 输出
    output_dir = Path("./results_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating visualizations...")
    plot_results(results, DEMO_MODEL, output_dir)

    # 简要统计
    gap = results["angular_gap_mean"]
    max_layer = np.argmax(gap)
    print(f"\n{'='*60}")
    print(f"  DEMO RESULTS SUMMARY")
    print(f"  Model: {DEMO_MODEL}")
    print(f"  Total layers: {results['num_layers']}")
    print(f"  Max angular gap: {gap[max_layer]:.2f}° at layer {max_layer}")
    print(f"  Gap positive in {np.sum(gap > 0)}/{len(gap)} layers")
    print(f"{'='*60}")
    print(f"\n  Charts saved to: {output_dir.absolute()}")
    print(f"  To run full experiment with 7B model:")
    print(f"    python run_experiment.py --model Qwen/Qwen2.5-7B-Instruct")


if __name__ == "__main__":
    main()
