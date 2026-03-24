# Political Bias Cosine Similarity Gap Experiment

基于 Safety Layers (ICLR 2025) 的余弦相似度鸿沟法，检测 LLM 内部是否存在政治倾向的线性表征。

## 实验原理

Safety Layers 论文发现，aligned LLM 在中间层会对 normal vs malicious 查询产生可测量的余弦相似度分离。
我们将此方法迁移到政治领域：用 Left vs Right 的政治观点替代 Normal vs Malicious，观察模型内部是否也存在类似的"鸿沟"。

## 三种配对

1. **Left-Left (L-L)**: 两个不同左翼观点的隐藏状态相似度
2. **Right-Right (R-R)**: 两个不同右翼观点的隐藏状态相似度
3. **Left-Right (L-R)**: 一个左翼和一个右翼观点的隐藏状态相似度

## 运行方法

```bash
# 1. 安装依赖
pip install torch transformers matplotlib numpy tqdm --break-system-packages

# 2. 运行实验
python run_experiment.py

# 3. 查看结果
# 生成的图表会保存在当前目录
```

## 硬件要求

- Mac Studio M1 (或任何 Apple Silicon Mac)
- 至少 16GB 内存
- 模型: Qwen/Qwen2.5-7B-Instruct (4-bit 量化约需 ~5GB)
