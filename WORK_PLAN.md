# Work Plan: Political Bias Representation Engineering

> **状态**: 🔬 Pre-Paper Research Phase
> **目标**: 建立扎实的实验基础，为未来论文做准备
> **当前阶段**: Phase 0 已完成（Pilot/Proof-of-Concept），进入 Phase 1

---

## 📋 总体路线图

```
Phase 0 (✅ 已完成)     Phase 1 (← 当前)      Phase 2              Phase 3
Pilot / PoC            数据集 & 模型扩展      系统性实验            论文撰写
──────────────        ──────────────        ──────────────       ──────────
• 自建20题数据集       • 公开数据集接入        • 全模型×全数据集      • 整理结果
• Qwen-7B 单模型      • 多模型适配           • 统计显著性检验       • 写论文
• 4步Pipeline验证     • 代码重构/鲁棒性      • Ablation studies   • 提交
• 基本可行性确认       • 基线方法对比          • 可视化完善

预计: 2周              预计: 4-6周            预计: 4-6周          预计: 4-8周
```

---

## 🔍 Phase 0 回顾：什么是 Pilot，什么需要改进

### ✅ Phase 0 已验证的事情

| 验证项 | 结论 | 置信度 |
|--------|------|--------|
| Cosine gap 方法可迁移到政治领域 | ✅ 可行 | 中 (小数据集) |
| 弱分类器可区分政治/非政治内容 | ✅ 可行 | 中 |
| 三角测量(N-L-R)可检测偏见方向 | ✅ 可行 | 中 |
| Steering hook 可在推理时修改行为 | ✅ 可行 | 低 (未充分评估) |

### ⚠️ Phase 0 的局限性（需要在 Phase 1 解决）

| 问题 | 严重程度 | Phase 1 解决方案 |
|------|----------|-----------------|
| **数据集太小** (20 topics, 自建) | 🔴 高 | 接入公开数据集 + 扩充自建数据 |
| **只测了 1 个模型** (Qwen-7B) | 🔴 高 | 扩展到 8-12 个模型 |
| **没有 baseline 方法对比** | 🔴 高 | 实现 Political Compass / OpinionQA 基线 |
| **统计显著性不足** | 🟡 中 | 增加 bootstrap CI, 多种子实验 |
| **Steering 评估太简化** | 🟡 中 | 增加 MMLU, TruthfulQA 等标准benchmark |
| **没有 human evaluation** | 🟡 中 | 设计标注方案 |
| **代码可复现性** | 🟢 低 | 添加 config 系统, 实验记录 |

---

## 📊 Phase 1: 数据集策略

### 核心原则

> **公开数据集 = 主实验（可信度 + 可复现性）**
> **自建数据集 = 补充验证（灵活性 + 针对性）**

### 1.1 公开数据集（Primary）

#### 📌 Dataset A: OpinionQA (Santurkar et al., ICML 2023)

```
来源:    PEW American Trends Panel
规模:    1,498 个问题, 覆盖政治/科学/社会等多主题
标签:    每题有美国公众按人口学分组的回答分布
优势:    ✅ 顶会论文使用, ✅ 有 ground truth 分布, ✅ 覆盖面广
获取:    https://github.com/tatsu-lab/opinions_qa

用途（在我们的框架中）:
  - Step 1: 用政治类 vs 非政治类问题做 political layer 定位
  - Step 2: 用 PEW 的 liberal/conservative 分组做 L/R 三角测量
  - Step 4: 用 steering 后的回答分布 vs PEW 数据做定量评估

适配方式:
  1. 筛选 OpinionQA 中明确的政治议题问题 (~200-400 题)
  2. 利用 PEW 的 demographic 分组:
     - "Liberal Democrat" 回答 → Left 表征
     - "Conservative Republican" 回答 → Right 表征
     - 全人群平均 → Neutral 参照
  3. 非政治类问题（科学、生活方式）→ Non-Political 对照组

难点: OpinionQA 是选择题格式, 需要转换为 prompt + 生成格式
```

#### 📌 Dataset B: Political Compass / 8values / ISideWith

```
来源:    政治取向测试问卷 (公开)
规模:    62 道题 (Political Compass) / ~70 道题 (8values)
标签:    经济左右轴 + 社会自由/威权轴
优势:    ✅ 被广泛用于 LLM 政治偏见研究, ✅ 已有多篇论文 baseline
获取:    https://www.politicalcompass.org/test (需手动整理)

用途:
  - 作为 behavioral evaluation 的标准工具
  - Step 4 evaluation: steering 前后模型在 Political Compass 上的得分变化
  - 与已有论文结果可直接对比 (e.g., OpenAI 2024, Röttger et al. ACL 2024)

适配方式:
  1. 将62道 Agree/Disagree 题目转为 prompt
  2. 提取 hidden states (不需要修改现有框架)
  3. 自动评分: 统计模型回答在4象限中的位置
```

#### 📌 Dataset C: P-Stance (Li et al., ACL Findings 2021)

```
来源:    Twitter (2020年美选)
规模:    21,574 条推文, 3 个 target (Trump, Biden, Sanders)
标签:    Favor / Against / None
优势:    ✅ 大规模, ✅ 真实社交媒体文本, ✅ 学术数据集
获取:    https://github.com/chuchun8/PStance

用途:
  - 提供真实世界政治文本 (不是合成的)
  - 测试模型对真实政治语言的内部表征
  - Step 3 topic analysis: 对比合成数据 vs 真实数据的 gap 模式

适配方式:
  1. 按 target + stance 分组: Pro-Trump → Right, Pro-Biden → Left
  2. 选择长度和质量较好的子集 (~500-1000 条)
  3. 作为 run_experiment.py 的外部数据源
```

#### 📌 Dataset D: AllSides / Media Bias (验证用)

```
来源:    AllSides.com 媒体偏见评级
规模:    800+ 媒体来源, 5级偏见标签 (Left / Lean Left / Center / Lean Right / Right)
优势:    ✅ 第三方权威评级, ✅ 与 ACL 2024 论文方法一致 (Pearson r=0.80)
获取:    https://www.allsides.com/media-bias/ratings

用途:
  - 外部验证: 我们的 bias_direction 指标是否与 AllSides 评级相关?
  - 选择不同偏见程度的新闻文本作为额外数据源
  - Step 3: 按媒体偏见程度分组, 验证 topic-level 分析

适配方式:
  1. 从 Left / Right 媒体各抓取同一话题的报道 (~20-50 对)
  2. 或使用已有的 BiasLab 数据集 (300 articles, dual-axis annotations)
```

### 1.2 自建数据集扩充（Supplementary）

```
当前: 20 个话题, 每话题 1 对 L/R + 1 个 Neutral + 1 个 Non-Political
目标: 50 个话题, 每话题 3 对 L/R + 3 个 Neutral + 3 个 Non-Political

扩充方向:
  ✅ 保留原有 20 个核心 US 政治话题
  🆕 增加 10 个细分话题:
     - 具体政策: 学生贷款减免, 大麻合法化, 死刑, 持枪权具体案例
     - 新兴议题: AI监管, 加密货币监管, 社交媒体审查
  🆕 增加 10 个国际政治话题:
     - 中美关系, 俄乌冲突, 巴以问题, 北约扩张, 气候协议
  🆕 增加 10 个边界模糊话题:
     - 传统上不分左右的议题 (太空探索, 基础设施投资)
     - 用于测试方法的 specificity (应该检测不到偏见)

每个话题的语句数量:
  - L/R 各 3 条 (不同表述方式), 而不是 1 条
  - Neutral 3 条 (不同角度的中立描述)
  - Non-Political 3 条 (不同角度的事实陈述)

总计: 50 × (6 + 3 + 3) = 600 条语句 (vs 当前 80 条)
```

### 1.3 数据集使用矩阵

| 实验步骤 | 主数据集 | 补充数据集 | 目的 |
|----------|----------|-----------|------|
| **Step 1**: Political Layer 定位 | OpinionQA (pol vs non-pol) | 自建 50 题 | 可复现性 + 灵活性 |
| **Step 2**: L/R/N 三角测量 | OpinionQA (分组) + P-Stance | 自建 L/R/N | 大规模 + 多样性 |
| **Step 3**: Topic 级分析 | 自建 50 题 (更可控) | P-Stance (真实文本) | 细粒度 + 真实性 |
| **Step 4**: Steering 评估 | Political Compass (行为) + OpinionQA (表征) | MMLU/TruthfulQA (能力) | 标准化 + 可比性 |
| **Validation**: 外部验证 | AllSides 评级 | BiasLab | 与权威评级对比 |

---

## 🤖 Phase 1: 模型扩展策略

### 2.1 模型选择矩阵

#### Dimension 1: 模型规模对比 (Qwen 系列)

> **研究问题**: 模型规模是否影响政治偏见的 localization 和 magnitude?

| 模型 | 参数量 | 层数 | Hidden Dim | 硬件需求 | 优先级 |
|------|--------|------|------------|----------|--------|
| Qwen2.5-1.5B-Instruct | 1.5B | 28 | 1536 | CPU/MPS | ✅ P0 (已有) |
| Qwen2.5-7B-Instruct | 7B | 32 | 3584 | MPS/T4 | ✅ P0 (已有) |
| Qwen2.5-14B-Instruct | 14B | 40 | 5120 | T4/A100 | 🔶 P1 |
| Qwen2.5-32B-Instruct | 32B | 64 | 5120 | A100 | 🔷 P2 (if possible) |

**预期假设**: 更大的模型 → political layers 更集中 / 偏见更微妙

#### Dimension 2: 跨家族对比

> **研究问题**: 不同训练数据/方法是否导致不同的偏见模式?

| 模型 | 家族 | 训练数据特点 | 优先级 |
|------|------|-------------|--------|
| Qwen2.5-7B-Instruct | Qwen (阿里) | 中英双语, 中国公司 | ✅ P0 |
| Llama-3.1-8B-Instruct | Llama (Meta) | 英语为主, 美国公司 | 🔶 P1 |
| Mistral-7B-Instruct-v0.3 | Mistral (法国) | 欧洲视角 | 🔶 P1 |
| Phi-3.5-mini-instruct | Phi (微软) | 合成数据为主 | 🔷 P2 |
| Gemma-2-9B-it | Gemma (Google) | 网页数据为主 | 🔷 P2 |

**预期假设**:
- 中国公司训练的模型 vs 美国公司 → 不同的偏见方向?
- 合成数据训练 (Phi) → 偏见更弱?
- 欧洲公司 (Mistral) → 不同的政治轴?

#### Dimension 3: Instruct vs Base 对比

> **研究问题**: Alignment (RLHF/DPO) 是否引入或放大了政治偏见?

| 对比组 | Base 模型 | Instruct 模型 | 优先级 |
|--------|-----------|--------------|--------|
| Qwen-7B | Qwen2.5-7B | Qwen2.5-7B-Instruct | ✅ P0 |
| Llama-8B | Llama-3.1-8B | Llama-3.1-8B-Instruct | 🔶 P1 |
| Mistral-7B | Mistral-7B-v0.3 | Mistral-7B-Instruct-v0.3 | 🔷 P2 |

**预期假设**:
- Base → 偏见来自预训练数据
- Instruct → alignment 可能放大某些偏见
- 对比 = 隔离 alignment 的贡献

### 2.2 模型实验优先级

```
Priority 0 (Week 1-2):   已有 + 最小改动
  ✅ Qwen2.5-7B-Instruct (已完成)
  ✅ Qwen2.5-1.5B-Instruct (已完成 demo)
  → Qwen2.5-7B (Base, 对照)

Priority 1 (Week 3-4):   核心扩展
  → Llama-3.1-8B-Instruct
  → Llama-3.1-8B (Base)
  → Mistral-7B-Instruct-v0.3
  → Qwen2.5-14B-Instruct

Priority 2 (Week 5-6):   如果资源允许
  → Phi-3.5-mini-instruct
  → Gemma-2-9B-it
  → Qwen2.5-32B-Instruct (需 A100)
  → Mistral-7B-v0.3 (Base)
```

### 2.3 计算资源规划

| 设备 | 可跑的模型 | 每模型 Pipeline 时间 | 总时间 |
|------|-----------|---------------------|--------|
| Mac Studio M1 (64GB) | ≤14B | ~3-4 小时 | ~20 小时 (5 models) |
| Google Colab T4 (免费) | ≤14B (4-bit) | ~2-3 小时 | ~15 小时 (5 models) |
| Google Colab A100 (付费) | ≤32B | ~1-2 小时 | ~6 小时 (3 models) |

**建议**:
- P0 + P1 模型 → Mac Studio + Colab T4 免费资源即可
- P2 模型中 32B → 需要 Colab Pro (A100) 或学校 GPU 集群

---

## 🔧 Phase 1: 代码重构需求

### 3.1 配置系统

```python
# configs/experiment_config.yaml (新增)
experiment:
  name: "full_pipeline_v1"
  seed: 42
  num_rounds: 500

dataset:
  primary: "opinionqa"         # opinionqa | pstance | custom
  supplementary: "custom_50"   # custom_50 | none

model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  device: "auto"
  quantize: false

evaluation:
  political_compass: true
  mmlu_subset: true
  truthfulqa: false            # Phase 2
```

### 3.2 数据加载抽象层

```python
# datasets/loader.py (新增)
class DatasetLoader:
    """统一的数据加载接口, 支持多数据源"""

    def load(self, name: str) -> PoliticalDataset:
        if name == "opinionqa":
            return OpinionQADataset(...)
        elif name == "pstance":
            return PStanceDataset(...)
        elif name == "custom":
            return CustomDataset(...)
        elif name == "political_compass":
            return PoliticalCompassDataset(...)
```

### 3.3 实验记录系统

```python
# utils/experiment_logger.py (新增)
class ExperimentLogger:
    """记录每次实验的配置、结果、元数据"""

    def log_run(self, config, results, metadata):
        # 保存到 experiments/YYYY-MM-DD_HH-MM_model_dataset/
        # 包含: config.yaml, results.npz, plots/, metrics.json
```

### 3.4 模型适配层

```python
# models/adapter.py (新增)
class ModelAdapter:
    """统一不同模型架构的接口"""

    def get_layer_module(self, model, layer_idx):
        # Qwen: model.model.layers[i]
        # Llama: model.model.layers[i]
        # Mistral: model.model.layers[i]
        # Phi: model.model.layers[i]
        # Gemma: model.model.layers[i]
        # 大部分模型结构一致, 但需要处理异常
```

---

## 📅 Phase 1: 详细时间线

### Week 1-2: 数据集接入

```
Week 1:
  ☐ Day 1-2: 下载 + 处理 OpinionQA 数据集
     - 筛选政治类问题 (~200-400)
     - 按 PEW demographic 分组 (Liberal/Conservative/Center)
     - 转换为 prompt 格式
  ☐ Day 3-4: 下载 + 处理 P-Stance 数据集
     - 清洗推文 (去 URL, @mention, #hashtag)
     - 按 target + stance 分组
     - 选择高质量子集
  ☐ Day 5-7: 整理 Political Compass 题目
     - 62 道题转为 prompt
     - 实现自动评分脚本
     - 在 Qwen-7B 上验证

Week 2:
  ☐ Day 1-3: 扩充自建数据集 (20 → 50 topics)
     - 增加 30 个新话题 (细分 + 国际 + 边界模糊)
     - 每话题 3 对 L/R + 3 Neutral + 3 Non-Political
  ☐ Day 4-5: 实现 DatasetLoader 统一接口
     - 编写 OpinionQA adapter
     - 编写 P-Stance adapter
     - 测试: 所有数据集可通过同一接口加载
  ☐ Day 6-7: 在 Qwen-7B 上跑完整 pipeline 验证
     - 用 OpinionQA 跑 Step 1-4
     - 对比 OpinionQA vs 自建数据集的结果
     - 记录差异, 分析原因
```

### Week 3-4: 模型扩展

```
Week 3:
  ☐ Day 1-2: 代码重构
     - 实现 ModelAdapter
     - 实现 ExperimentLogger
     - 实现 config 系统
  ☐ Day 3-4: Qwen2.5-7B Base (对照实验)
     - 跑完整 pipeline
     - 对比 Base vs Instruct 结果
  ☐ Day 5-7: Llama-3.1-8B-Instruct
     - 适配模型加载 (HF token)
     - 跑完整 pipeline
     - 对比 Llama vs Qwen 结果

Week 4:
  ☐ Day 1-2: Llama-3.1-8B Base
     - 跑 pipeline, 对比 Base vs Instruct
  ☐ Day 3-4: Mistral-7B-Instruct-v0.3
     - 适配 + pipeline
  ☐ Day 5-6: Qwen2.5-14B-Instruct
     - 需要 Colab T4 或 Mac Studio
     - 跑 pipeline, 对比 7B vs 14B
  ☐ Day 7: 汇总 Week 3-4 结果
     - 生成对比表格
     - 识别模式和异常
```

### Week 5-6: 巩固 + P2 模型 (如果资源允许)

```
Week 5:
  ☐ 统计检验: bootstrap CI for all results
  ☐ 多种子实验: seed={42, 123, 456, 789, 2024}
  ☐ P2 模型: Phi-3.5, Gemma-2 (Colab)
  ☐ P2 模型: Qwen-32B (如果有 A100)

Week 6:
  ☐ 整理所有结果到统一格式
  ☐ 生成跨模型对比可视化
  ☐ 撰写 Phase 1 总结报告
  ☐ 决定 Phase 2 的方向
```

---

## 📊 Phase 1: 评估体系升级

### 4.1 表征层评估 (Representation)

| 指标 | Phase 0 | Phase 1 目标 |
|------|---------|-------------|
| Cosine gap | ✅ 有 | 增加 bootstrap 95% CI |
| Angular gap onset | ✅ 有 | 增加多种子稳定性检验 |
| Bias direction | ✅ 有 | 增加 per-topic 方差分析 |
| Classifier accuracy | ✅ 有 | 增加 ROC-AUC, F1 |

### 4.2 行为层评估 (Behavioral)

| 指标 | Phase 0 | Phase 1 目标 |
|------|---------|-------------|
| 生成文本定性检查 | ✅ 有 | 保留 |
| Political Compass 得分 | ❌ 无 | 🆕 自动化评分脚本 |
| OpinionQA 分布对齐 | ❌ 无 | 🆕 KL divergence with PEW data |
| Stance score (自动化) | ❌ 无 | 🆕 用 RoBERTa-stance 模型自动评分 |

### 4.3 能力层评估 (Capability)

| 指标 | Phase 0 | Phase 1 目标 |
|------|---------|-------------|
| 5 道 QA | ✅ 有 (太简化) | 🆕 MMLU subset (200 题) |
| TruthfulQA | ❌ 无 | 🔶 Phase 2 |
| Perplexity change | ❌ 无 | 🆕 在 WikiText 上测量 |

---

## 🔬 Phase 2 预览: 系统性实验 (Phase 1 完成后)

### 将要回答的研究问题

```
RQ1: 政治偏见在 LLM 的哪些层编码?
     → Step 1 跨 8+ 模型的一致性分析

RQ2: 不同模型家族的政治偏见方向是否不同?
     → Step 2 跨家族对比 (中国 vs 美国 vs 欧洲训练)

RQ3: Alignment (RLHF/DPO) 是否放大了政治偏见?
     → Base vs Instruct 对比

RQ4: 模型规模如何影响政治偏见?
     → 1.5B / 7B / 14B / 32B scaling analysis

RQ5: 表征层面的偏见是否导致行为层面的偏见?
     → 表征指标 vs 行为指标的相关性分析

RQ6: 推理时 steering 能否有效降低政治偏见?
     → Steering 效果跨模型泛化性
```

### Phase 2 的 Ablation Studies

```
☐ Ablation 1: Steering 层范围 — 只 steer 一半的 political layers vs 全部
☐ Ablation 2: 方向向量来源 — 用 OpinionQA 训练 vs 自建数据集训练
☐ Ablation 3: Alpha sweep — 更细粒度的 alpha 搜索 (0.1, 0.2, ..., 5.0)
☐ Ablation 4: Per-topic steering — 不同话题用不同方向向量
☐ Ablation 5: 数据集敏感性 — 同一模型在不同数据集上的 gap 一致性
```

---

## 📂 Phase 1 目标文件结构

```
political_bias_cosine_gap/
│
├── configs/                          # 🆕 实验配置
│   ├── default.yaml
│   ├── model_qwen7b.yaml
│   ├── model_llama8b.yaml
│   └── ...
│
├── datasets/                         # 🆕 统一数据接口
│   ├── __init__.py
│   ├── loader.py                     # DatasetLoader
│   ├── opinionqa_adapter.py          # OpinionQA 接入
│   ├── pstance_adapter.py            # P-Stance 接入
│   ├── political_compass.py          # Political Compass 题目
│   ├── custom_50_topics.py           # 扩充后的自建数据 (50 topics)
│   └── allsides_validation.py        # AllSides 外部验证
│
├── models/                           # 🆕 模型适配
│   ├── __init__.py
│   ├── adapter.py                    # ModelAdapter
│   └── steering_hook.py              # PoliticalSteeringHook (从 step4 提取)
│
├── evaluation/                       # 🆕 评估工具
│   ├── __init__.py
│   ├── political_compass_eval.py     # Political Compass 自动评分
│   ├── opinionqa_eval.py             # OpinionQA KL divergence
│   ├── mmlu_eval.py                  # MMLU subset
│   └── stance_scorer.py             # 基于 RoBERTa 的 stance 评分
│
├── utils/                            # 🆕 工具
│   ├── experiment_logger.py
│   ├── statistics.py                 # Bootstrap CI, 多种子聚合
│   └── visualization.py              # 统一可视化函数
│
├── experiments/                      # 🆕 实验记录 (自动生成)
│   ├── 2025-03-25_qwen7b_opinionqa/
│   ├── 2025-03-26_llama8b_opinionqa/
│   └── ...
│
├── step1_locate_political_layers.py  # 更新: 支持新数据接口
├── step2_analyze_bias.py             # 更新: 支持新数据接口
├── step3_topic_analysis.py           # 更新: 支持新数据接口
├── step4_steering.py                 # 更新: 支持新评估
│
├── run_full_pipeline.py              # 🆕 一键运行全 pipeline
├── run_cross_model_comparison.py     # 🆕 跨模型对比
├── generate_paper_figures.py         # 🆕 论文级可视化 (Phase 2)
│
└── WORK_PLAN.md                      # 本文档
```

---

## ✅ Phase 1 完成标准 (Definition of Done)

### 数据集
- [ ] OpinionQA 接入完成, 可通过统一接口加载
- [ ] P-Stance 接入完成, 可通过统一接口加载
- [ ] Political Compass 62 题自动评分脚本完成
- [ ] 自建数据集从 20 → 50 topics 扩充完成
- [ ] 所有数据集在 Qwen-7B 上验证通过

### 模型
- [ ] P0 模型全部完成: Qwen-1.5B, Qwen-7B, Qwen-7B-Base (3 个)
- [ ] P1 模型全部完成: Llama-8B, Llama-8B-Base, Mistral-7B, Qwen-14B (4 个)
- [ ] P2 模型至少完成 1 个: Phi-3.5 或 Gemma-2

### 评估
- [ ] Political Compass 评分在所有模型上运行
- [ ] MMLU subset (200 题) 能力评估在所有模型上运行
- [ ] Bootstrap 95% CI 用于所有关键指标

### 代码
- [ ] 配置系统可用 (YAML)
- [ ] DatasetLoader 统一接口可用
- [ ] ModelAdapter 支持所有目标模型
- [ ] ExperimentLogger 自动记录每次实验

### 文档
- [ ] 所有新代码有 docstring
- [ ] Phase 1 结果总结报告

---

## 📚 关键参考文献

| 论文 | 年份/会议 | 与我们的关系 |
|------|----------|-------------|
| Safety Layers | ICLR 2025 | 核心方法论来源 |
| Röttger et al. "Measuring Political Bias in LLMs" | ACL 2024 | 直接 baseline 对比 |
| Santurkar et al. "Whose Opinions Do LMs Reflect?" | ICML 2023 | OpinionQA 数据集 |
| Bang et al. Topic-specific stance | ACL 2024 | Content/Style 分解 |
| Zhou et al. Weak-to-Strong Explanation | EMNLP 2024 | 弱分类器探针方法 |
| Li et al. "P-Stance" | ACL Findings 2021 | P-Stance 数据集 |
| OpenAI "Defining Political Bias in LLMs" | 2024 Blog | 行业基准参考 |
| Zou et al. "Representation Engineering" | 2023 | RepE/CAA 方法 |

---

## 💡 最终目标 (Phase 3 — 论文)

> **暂时不急**, 但 Phase 1-2 的所有工作都应该为论文服务

**潜在论文标题**:

> *"Political Layers: A Hierarchical Approach to Locating, Analyzing, and Steering Political Biases in Large Language Models"*

**潜在投稿方向**:
- ACL / EMNLP / NAACL (NLP 顶会)
- AAAI / NeurIPS (AI 顶会)
- FAccT / AIES (AI 伦理/公平性)

**论文需要的最低实验量**:
- 至少 3 个模型家族 × 2 个规模 × (Base + Instruct) = 12 个模型
- 至少 2 个公开数据集 + 1 个自建数据集
- 至少 3 种 baseline 方法对比
- 统计显著性检验 (p < 0.05)

---

<div align="center">

**Phase 0 ✅ → Phase 1 🔬 → Phase 2 📊 → Phase 3 📝**

*当前: Phase 1 — 数据集扩展 + 模型扩展*

</div>
