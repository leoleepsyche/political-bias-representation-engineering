# Work Plan: Political Bias Representation Engineering

> **状态**: 🔬 Pre-Paper Research Phase
> **目标**: 建立扎实的实验基础，为未来论文做准备
> **执行方式**: 按 Step 顺序逐步推进，每个 Step 完成后再进入下一个

---

## 📋 总览：从 Pilot 到正式实验

```
Pilot (✅ 已完成)              正式实验 (← 当前)
────────────────              ────────────────────────────────────────
• 自建 20 题数据集            Step 1: 数据集建设 (公开 + 扩充自建)
• Qwen-7B 单模型              Step 2: 代码重构 (统一接口)
• 4 步 Pipeline 验证           Step 3: 多模型实验 (规模 × 家族 × Base/Instruct)
• 基本可行性确认               Step 4: 评估体系升级 (标准 benchmark)
                              Step 5: 统计检验 + 结果整理
                              Step 6: 论文准备 (不急，积累够了再写)
```

---

## 🔍 Pilot 回顾：已验证 vs 需改进

### ✅ 已验证

| 验证项 | 结论 | 置信度 |
|--------|------|--------|
| Cosine gap 方法可迁移到政治领域 | ✅ 可行 | 中 (小数据集) |
| 弱分类器可区分政治/非政治内容 | ✅ 可行 | 中 |
| 三角测量 (N-L-R) 可检测偏见方向 | ✅ 可行 | 中 |
| Steering hook 可在推理时修改行为 | ✅ 可行 | 低 (未充分评估) |

### ⚠️ 需改进

| 问题 | 严重程度 | 对应 Step |
|------|----------|----------|
| **数据集太小** (20 topics, 全自建) | 🔴 高 | Step 1 |
| **只测了 1 个模型** | 🔴 高 | Step 3 |
| **没有 baseline 方法对比** | 🔴 高 | Step 4 |
| **统计显著性不足** | 🟡 中 | Step 5 |
| **Steering 评估太简化** | 🟡 中 | Step 4 |
| **代码耦合度高** | 🟢 低 | Step 2 |

---

## Step 1: 数据集建设

> **目标**: 建立可信的、可复现的数据基础
> **原则**: 公开数据集 = 主实验（可信度），自建数据集 = 补充验证（灵活性）

### 1A. 公开数据集接入

#### 📌 OpinionQA (Santurkar et al., ICML 2023) — 主力数据集

```
来源:   PEW American Trends Panel
规模:   1,498 个问题
标签:   每题有美国公众按人口学分组的回答分布
获取:   https://github.com/tatsu-lab/opinions_qa
```

**在我们框架中的用法**:

| 我们的 Step | 怎么用 OpinionQA |
|------------|-----------------|
| 定位 Political Layers | 筛选政治类 vs 非政治类问题 (~200 pol + ~200 non-pol) |
| L/R/N 三角测量 | 用 PEW demographic 分组: Liberal Democrat → Left, Conservative Republican → Right, 全人群 → Neutral |
| Steering 评估 | 对比 steering 前后的回答分布 vs PEW 数据 (KL divergence) |

**适配工作**:
- [ ] 下载 OpinionQA 数据
- [ ] 筛选明确的政治议题问题 (~200-400 题)
- [ ] 按 demographic 分组转为 L/R/N 标签
- [ ] 将选择题格式转为 prompt + generation 格式
- [ ] 非政治类问题 (科学、生活方式) 作为 Non-Political 对照
- [ ] 在 Qwen-7B 上验证: 跑 Step 1-2，对比 Pilot 结果

#### 📌 P-Stance (Li et al., ACL Findings 2021) — 真实文本

```
来源:   Twitter (2020 美选)
规模:   21,574 条推文, 3 targets (Trump, Biden, Sanders)
标签:   Favor / Against / None
获取:   https://github.com/chuchun8/PStance
```

**用法**: 提供真实世界政治文本（不是合成的），测试模型对真实政治语言的内部表征

**适配工作**:
- [ ] 下载并清洗 (去 URL, @mention, #hashtag)
- [ ] 按 target + stance 分组: Pro-Trump → Right, Pro-Biden → Left
- [ ] 选择高质量子集 (~500-1000 条)
- [ ] 在 Qwen-7B 上验证

#### 📌 Political Compass — 行为评估工具

```
来源:   政治取向测试问卷 (公开)
规模:   62 道 Agree/Disagree 题
标签:   经济左右轴 + 社会自由/威权轴
```

**用法**: Step 4 评估 — steering 前后模型在 Political Compass 上的得分变化（与已有论文直接可比）

**适配工作**:
- [ ] 62 道题转为 prompt
- [ ] 实现自动评分脚本 (4 象限定位)
- [ ] 在 Qwen-7B 上跑一次 baseline

#### 📌 AllSides / BiasLab — 外部验证

```
AllSides:  800+ 媒体来源, 5 级偏见标签
BiasLab:   300 篇政治新闻, dual-axis 标注 (ACL 2024 验证 Pearson r=0.80)
```

**用法**: 外部验证 — 我们的 bias_direction 指标是否与权威评级一致？

**适配工作**:
- [ ] 从 Left / Right 媒体各抓取同一话题的报道 (~20-50 对)
- [ ] 或直接使用 BiasLab 数据集

### 1B. 自建数据集扩充 (20 → 50 topics)

```
保留:   原有 20 个核心 US 政治话题

新增 10 个细分话题:
  学生贷款减免, 大麻合法化, 死刑, 持枪权具体案例,
  AI 监管, 加密货币监管, 社交媒体审查, 堕胎具体限制,
  警察经费, 宗教自由 vs 反歧视

新增 10 个国际政治话题:
  中美关系, 俄乌冲突, 巴以问题, 北约扩张, 气候协议,
  WTO 改革, 联合国权力, 移民危机(欧洲), 台湾问题, 南海争议

新增 10 个边界模糊话题 (negative control):
  太空探索, 基础设施投资, 网络安全, 自然灾害应急,
  食品安全标准, 交通规划, 数学教育, 体育政策,
  公共图书馆, 时区改革
  → 应该检测不到政治偏见 → 验证方法 specificity

每个话题:
  - L/R 各 3 条 (不同表述)
  - Neutral 3 条
  - Non-Political 3 条

总计: 50 × 12 = 600 条语句 (vs Pilot 的 80 条)
```

**适配工作**:
- [ ] 编写 30 个新话题的 L/R/N/NP 语句
- [ ] 更新 `political_dataset.py` → `datasets/custom_50_topics.py`
- [ ] 在 Qwen-7B 上验证: 新话题的 cosine gap 是否合理

### 1C. 数据集使用矩阵

| 实验步骤 | 主数据集 | 补充数据集 |
|----------|----------|-----------|
| Political Layer 定位 | OpinionQA (pol vs non-pol) | 自建 50 题 |
| L/R/N 三角测量 | OpinionQA (分组) + P-Stance | 自建 L/R/N |
| Topic 级分析 | 自建 50 题 (更可控) | P-Stance |
| Steering 评估 (行为) | Political Compass 62 题 | OpinionQA |
| Steering 评估 (能力) | MMLU subset | — |
| 外部验证 | AllSides / BiasLab | — |

### ✅ Step 1 完成标准

- [ ] OpinionQA 可通过统一接口加载并跑通 pipeline
- [ ] P-Stance 可加载并用于表征提取
- [ ] Political Compass 自动评分脚本可用
- [ ] 自建数据集扩充到 50 topics, 600 条语句
- [ ] 所有数据集在 Qwen-7B 上验证通过

---

## Step 2: 代码重构

> **目标**: 让代码支持多数据集、多模型，方便后续大规模实验
> **前置**: Step 1 完成

### 2A. 新增文件结构

```
新增:
├── configs/
│   ├── default.yaml              # 默认配置
│   ├── model_qwen7b.yaml         # 模型特定配置
│   └── ...
├── datasets/
│   ├── __init__.py
│   ├── loader.py                 # DatasetLoader 统一接口
│   ├── opinionqa_adapter.py      # OpinionQA
│   ├── pstance_adapter.py        # P-Stance
│   ├── political_compass.py      # Political Compass
│   ├── custom_50_topics.py       # 扩充后的自建数据
│   └── allsides_validation.py    # AllSides 验证
├── models/
│   ├── adapter.py                # ModelAdapter (统一不同架构)
│   └── steering_hook.py          # 从 step4 提取出来
├── evaluation/
│   ├── political_compass_eval.py # PC 自动评分
│   ├── mmlu_eval.py              # MMLU subset
│   └── stance_scorer.py          # RoBERTa stance 评分
├── utils/
│   ├── experiment_logger.py      # 实验记录
│   └── statistics.py             # Bootstrap CI, 多种子聚合
├── run_full_pipeline.py          # 一键跑完 step1-4
└── run_cross_model_comparison.py # 跨模型对比
```

### 2B. 关键抽象

```python
# DatasetLoader: 统一数据加载
loader = DatasetLoader("opinionqa")  # 或 "pstance", "custom", "political_compass"
pol_stmts = loader.get_political()
nonpol_stmts = loader.get_nonpolitical()
left_stmts = loader.get_left()
right_stmts = loader.get_right()
neutral_stmts = loader.get_neutral()

# ModelAdapter: 统一模型接口
adapter = ModelAdapter("Qwen/Qwen2.5-7B-Instruct")
adapter.get_layer_module(layer_idx)  # 统一的层访问
adapter.get_num_layers()
adapter.get_hidden_dim()

# ExperimentLogger: 实验记录
logger = ExperimentLogger("qwen7b_opinionqa_step1")
logger.log_config(config)
logger.log_results(results)
logger.save()  # → experiments/2025-03-25_qwen7b_opinionqa_step1/
```

### 2C. 配置系统

```yaml
# configs/default.yaml
experiment:
  seed: 42
  num_rounds: 500

dataset:
  primary: "opinionqa"
  supplementary: "custom_50"

model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  device: "auto"
  quantize: false

evaluation:
  political_compass: true
  mmlu_subset: true
```

### ✅ Step 2 完成标准

- [ ] DatasetLoader 支持所有 Step 1 的数据集
- [ ] ModelAdapter 支持 Qwen, Llama, Mistral 架构
- [ ] ExperimentLogger 自动记录每次实验
- [ ] `run_full_pipeline.py` 可一键跑完 step1-4
- [ ] 原有 Pilot 代码仍然可以独立运行 (不破坏兼容性)

---

## Step 3: 多模型实验

> **目标**: 跨模型验证方法的泛化性，发现不同模型的偏见模式差异
> **前置**: Step 1 + Step 2 完成

### 3A. 模型矩阵 (三个维度)

#### Dimension 1: 规模对比 (Qwen 系列)

> **研究问题**: 更大的模型 → political layers 更集中? 偏见更微妙?

| 模型 | 参数量 | 层数 | 硬件需求 | 优先级 |
|------|--------|------|----------|--------|
| Qwen2.5-1.5B-Instruct | 1.5B | 28 | CPU/MPS | ✅ P0 (已有) |
| Qwen2.5-7B-Instruct | 7B | 32 | MPS/T4 | ✅ P0 (已有) |
| Qwen2.5-14B-Instruct | 14B | 40 | T4/A100 | 🔶 P1 |
| Qwen2.5-32B-Instruct | 32B | 64 | A100 | 🔷 P2 (if possible) |

#### Dimension 2: 跨家族对比

> **研究问题**: 中国公司 vs 美国公司 vs 欧洲公司 → 偏见方向不同?

| 模型 | 家族 | 训练背景 | 优先级 |
|------|------|---------|--------|
| Qwen2.5-7B-Instruct | Qwen (阿里) | 中英双语, 中国 | ✅ P0 |
| Llama-3.1-8B-Instruct | Llama (Meta) | 英语为主, 美国 | 🔶 P1 |
| Mistral-7B-Instruct-v0.3 | Mistral (法国) | 欧洲视角 | 🔶 P1 |
| Phi-3.5-mini-instruct | Phi (微软) | 合成数据为主 | 🔷 P2 |
| Gemma-2-9B-it | Gemma (Google) | 网页数据为主 | 🔷 P2 |

#### Dimension 3: Base vs Instruct

> **研究问题**: Alignment (RLHF/DPO) 是引入还是放大了政治偏见?

| Base | Instruct | 优先级 |
|------|---------|--------|
| Qwen2.5-7B | Qwen2.5-7B-Instruct | ✅ P0 |
| Llama-3.1-8B | Llama-3.1-8B-Instruct | 🔶 P1 |
| Mistral-7B-v0.3 | Mistral-7B-Instruct-v0.3 | 🔷 P2 |

### 3B. 执行顺序

```
P0 (首先做):
  ☐ Qwen2.5-7B-Base (对照, 隔离 alignment 效果)
  → 已有 Qwen-7B-Instruct 和 Qwen-1.5B-Instruct

P1 (核心扩展):
  ☐ Llama-3.1-8B-Instruct
  ☐ Llama-3.1-8B (Base)
  ☐ Mistral-7B-Instruct-v0.3
  ☐ Qwen2.5-14B-Instruct

P2 (资源允许时):
  ☐ Phi-3.5-mini-instruct
  ☐ Gemma-2-9B-it
  ☐ Qwen2.5-32B-Instruct (需 A100)
  ☐ Mistral-7B-v0.3 (Base)
```

### 3C. 计算资源

| 设备 | 可跑模型 | 每模型时间 |
|------|---------|-----------|
| Mac Studio M1 (64GB) | ≤14B | ~3-4 小时 |
| Colab T4 (免费) | ≤14B (4-bit) | ~2-3 小时 |
| Colab A100 (付费) | ≤32B | ~1-2 小时 |

**P0 + P1** → Mac Studio + Colab T4 免费资源即可
**P2 的 32B** → 需要 Colab Pro 或学校 GPU

### 3D. 每个模型跑什么

```
对于每个模型, 跑:
  1. step1_locate_political_layers.py (主数据集: OpinionQA)
  2. step2_analyze_bias.py (主数据集: OpinionQA)
  3. step3_topic_analysis.py (自建 50 topics)
  4. step4_steering.py (alpha sweep: 0.5, 1.0, 2.0, 3.0, 5.0)

然后额外跑:
  5. Political Compass 评分
  6. MMLU subset (200 题)
```

### ✅ Step 3 完成标准

- [ ] P0 模型全部完成 (3 个: Qwen-1.5B, Qwen-7B, Qwen-7B-Base)
- [ ] P1 模型全部完成 (4 个: Llama-8B, Llama-8B-Base, Mistral-7B, Qwen-14B)
- [ ] 每个模型有完整的 step1-4 结果
- [ ] 跨模型对比表格生成

---

## Step 4: 评估体系升级

> **目标**: 用标准 benchmark 替代 Pilot 中的简化评估
> **前置**: Step 1-3 完成（至少 P0 模型）

### 4A. 表征层评估

| 指标 | Pilot 状态 | 升级目标 |
|------|-----------|---------|
| Cosine gap | ✅ 有 | + bootstrap 95% CI |
| Bias direction | ✅ 有 | + per-topic 方差分析 |
| Classifier accuracy | ✅ 有 | + ROC-AUC, F1, confusion matrix |

### 4B. 行为层评估

| 指标 | Pilot 状态 | 升级目标 |
|------|-----------|---------|
| 生成文本定性检查 | ✅ 有 | 保留 |
| Political Compass 得分 | ❌ 无 | 🆕 自动化评分, 4 象限定位 |
| OpinionQA 分布对齐 | ❌ 无 | 🆕 KL divergence with PEW data |
| Stance 自动评分 | ❌ 无 | 🆕 RoBERTa-stance 模型 |

### 4C. 能力层评估

| 指标 | Pilot 状态 | 升级目标 |
|------|-----------|---------|
| 5 道简单 QA | ✅ 有 (太简化) | 🆕 MMLU subset (200 题) |
| Perplexity | ❌ 无 | 🆕 WikiText 上测量 steering 前后 |

### 4D. Baseline 方法对比

我们的方法需要和现有方法对比:

| Baseline 方法 | 论文来源 | 实现方式 |
|--------------|---------|---------|
| Political Compass 直接测试 | 多篇论文 | 62 题问答 + 评分 |
| Prompt-based stance 检测 | Röttger et al. ACL 2024 | 直接问模型立场 |
| Linear probe (全连接层) | 多篇论文 | 在 hidden states 上训练分类器 |
| 我们: Hierarchical RepE | — | 4-step pipeline |

### ✅ Step 4 完成标准

- [ ] Political Compass 自动评分在所有模型上运行
- [ ] MMLU subset 能力评估在所有模型上运行
- [ ] Bootstrap 95% CI 用于所有关键指标
- [ ] 至少 2 种 baseline 方法实现并对比
- [ ] Steering 前后的 Political Compass 分数变化记录

---

## Step 5: 统计检验 + 结果整理

> **目标**: 确保所有结论有统计支撑
> **前置**: Step 1-4 完成

### 5A. 统计检验

```
对每个关键结论, 需要:

1. Bootstrap 95% 置信区间
   - 对 cosine gap, bias_direction 等指标
   - 1000 次 bootstrap 重采样

2. 多种子实验
   - seed = {42, 123, 456, 789, 2024}
   - 报告均值 ± 标准差

3. 效应量 (Effect Size)
   - Cohen's d for 组间差异
   - 不只报告 p 值

4. 跨数据集一致性
   - 同一模型在 OpinionQA vs 自建数据集上的结果相关性
   - Pearson r > 0.7 → 方法稳健
```

### 5B. 跨模型对比可视化

```
需要生成的对比图:

1. Political Layer Heatmap: model × layer (哪些层是 political)
   → 所有模型并排对比

2. Bias Direction 对比: model × topic
   → 不同模型在不同话题上的偏见方向

3. Scaling Plot: model_size × bias_magnitude
   → 偏见是否随模型规模变化

4. Base vs Instruct 对比: 同一家族的 alignment 效果

5. Steering Effectiveness: model × alpha × bias_reduction
   → 哪些模型更容易 steer

6. Capability-Bias Pareto: 所有模型的 tradeoff 曲线
```

### 5C. 结果整理

```
输出:
├── results_summary/
│   ├── cross_model_comparison.csv      # 所有模型的关键指标汇总
│   ├── statistical_tests.csv           # 所有统计检验结果
│   ├── figures/                        # 论文级可视化
│   │   ├── fig1_political_layers_comparison.pdf
│   │   ├── fig2_bias_direction_heatmap.pdf
│   │   ├── fig3_scaling_analysis.pdf
│   │   ├── fig4_base_vs_instruct.pdf
│   │   ├── fig5_steering_effectiveness.pdf
│   │   └── fig6_capability_bias_tradeoff.pdf
│   └── tables/
│       ├── table1_model_overview.tex
│       ├── table2_political_layer_boundaries.tex
│       └── table3_steering_results.tex
```

### ✅ Step 5 完成标准

- [ ] 所有关键指标有 95% CI
- [ ] 多种子实验 (≥3 seeds) 完成
- [ ] 跨模型对比表格和可视化生成
- [ ] Phase 1 总结报告撰写

---

## Step 6: 论文准备 (不急)

> **目标**: 在 Step 1-5 积累足够后，开始论文撰写
> **前置**: Step 5 完成，且结果有明确的 story

### 6A. 论文需要的最低实验量

```
- 至少 3 个模型家族 × 2 个规模 = 6+ 个模型
- 至少 2 个公开数据集 + 1 个自建数据集
- 至少 2 种 baseline 方法对比
- 统计显著性检验 (bootstrap CI)
- Ablation studies (层范围, alpha, 数据集)
```

### 6B. Ablation Studies (Step 5 之后做)

```
☐ Ablation 1: Steering 层范围 — 只 steer 一半 vs 全部 political layers
☐ Ablation 2: 方向向量来源 — OpinionQA 训练 vs 自建数据集训练
☐ Ablation 3: Alpha 细粒度 sweep — (0.1, 0.2, ..., 5.0)
☐ Ablation 4: Per-topic steering — 不同话题用不同方向向量
☐ Ablation 5: 数据集敏感性 — 同一模型在不同数据集上的一致性
```

### 6C. 潜在研究问题

```
RQ1: 政治偏见在 LLM 的哪些层编码?
RQ2: 不同模型家族的政治偏见方向是否不同?
RQ3: Alignment 是否放大了政治偏见?
RQ4: 模型规模如何影响政治偏见?
RQ5: 表征偏见是否导致行为偏见? (两者相关性)
RQ6: 推理时 steering 能否有效降低政治偏见?
```

### 6D. 潜在投稿方向

```
NLP 顶会: ACL / EMNLP / NAACL
AI 顶会: AAAI / NeurIPS / ICML
AI 伦理: FAccT / AIES
```

---

## 📂 最终文件结构目标

```
political_bias_cosine_gap/
│
├── configs/                          # 实验配置
├── datasets/                         # 统一数据接口
│   ├── loader.py
│   ├── opinionqa_adapter.py
│   ├── pstance_adapter.py
│   ├── political_compass.py
│   ├── custom_50_topics.py
│   └── allsides_validation.py
│
├── models/                           # 模型适配
│   ├── adapter.py
│   └── steering_hook.py
│
├── evaluation/                       # 评估工具
│   ├── political_compass_eval.py
│   ├── mmlu_eval.py
│   └── stance_scorer.py
│
├── utils/                            # 工具
│   ├── experiment_logger.py
│   └── statistics.py
│
├── experiments/                      # 实验记录 (自动生成)
│
├── step1_locate_political_layers.py  # 核心 Pipeline (更新后)
├── step2_analyze_bias.py
├── step3_topic_analysis.py
├── step4_steering.py
│
├── run_full_pipeline.py              # 一键运行
├── run_cross_model_comparison.py     # 跨模型对比
│
├── pilot/                            # Pilot 阶段代码 (归档)
│   ├── run_experiment.py
│   ├── run_triangulation.py
│   ├── run_enhanced.py
│   ├── run_control_experiment.py
│   ├── political_dataset.py          # 原 20 topics
│   └── ...
│
└── WORK_PLAN.md                      # 本文档
```

---

## 📚 关键参考文献

| 论文 | 会议 | 与我们的关系 |
|------|------|-------------|
| Safety Layers | ICLR 2025 | 核心方法论来源 |
| Röttger et al. | ACL 2024 | 直接 baseline 对比 |
| Santurkar et al. | ICML 2023 | OpinionQA 数据集 |
| Bang et al. | ACL 2024 | Content/Style 分解 |
| Zhou et al. | EMNLP 2024 | 弱分类器探针 |
| Li et al. | ACL Findings 2021 | P-Stance 数据集 |
| OpenAI | 2024 Blog | 行业基准 |
| Zou et al. | 2023 | RepE/CAA 方法 |

---

## ✅ 整体进度追踪

| Step | 内容 | 状态 | 依赖 |
|------|------|------|------|
| Pilot | 自建 20 题 + Qwen-7B + 4 步 Pipeline | ✅ 完成 | — |
| Step 1 | 数据集建设 (公开 + 扩充自建) | ☐ 待开始 | — |
| Step 2 | 代码重构 (统一接口) | ☐ 待开始 | Step 1 |
| Step 3 | 多模型实验 (规模 × 家族 × Base/Instruct) | ☐ 待开始 | Step 1 + 2 |
| Step 4 | 评估体系升级 (标准 benchmark) | ☐ 待开始 | Step 1 + 2 |
| Step 5 | 统计检验 + 结果整理 | ☐ 待开始 | Step 3 + 4 |
| Step 6 | 论文准备 | ☐ 不急 | Step 5 |

---

<div align="center">

**Pilot ✅ → Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6**

*每个 Step 完成后再进入下一个，不赶时间，做扎实*

</div>
