# 🚀 GitHub Setup & Upload Instructions

完整的步骤将项目上传到 GitHub 并启用多人协作。

---

## 📋 前置条件

1. **GitHub 账号** — 注册在 https://github.com
2. **Git 已安装** — 检查：`git --version`
3. **GitHub 认证** — 使用 SSH 或 HTTPS token

### 检查 Git 配置

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## 🔧 Step 1: 在 GitHub 创建仓库（网页）

1. 访问 https://github.com/new
2. 填写信息：
   ```
   Repository name: political-bias-representation-engineering
   Description: A hierarchical approach to locating and steering political biases in LLMs using Representation Engineering
   Visibility: Public  (如果你想开源)
   ```
3. **不要勾选** "Initialize this repository with a README"
4. 点击 "Create repository"

**完成后你会看到：**
```
https://github.com/YOUR_USERNAME/political-bias-representation-engineering
```

---

## ⬆️ Step 2: 本地 Push 到 GitHub

### 方法 A: SSH（推荐）

```bash
cd /sessions/quirky-gifted-hypatia/mnt/outputs/political_bias_cosine_gap

# 1. 添加远程仓库
git remote add origin git@github.com:YOUR_USERNAME/political-bias-representation-engineering.git

# 2. 重命名分支（可选）
git branch -M main

# 3. 推送到 GitHub
git push -u origin main
```

### 方法 B: HTTPS（如果没有 SSH）

```bash
cd /sessions/quirky-gifted-hypatia/mnt/outputs/political_bias_cosine_gap

# 1. 添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/political-bias-representation-engineering.git

# 2. 重命名分支（可选）
git branch -M main

# 3. 推送到 GitHub（会要求输入用户名和token）
git push -u origin main
```

**获取 HTTPS Token**:
- https://github.com/settings/tokens
- 点击 "Generate new token (classic)"
- 勾选 `repo` 权限
- 复制 token，在 push 时粘贴

**获取 SSH Key**（推荐）:
```bash
# 生成 SSH 密钥（如果还没有）
ssh-keygen -t ed25519 -C "your.email@example.com"

# 复制公钥
cat ~/.ssh/id_ed25519.pub

# 粘贴到 https://github.com/settings/keys
```

---

## ✅ Step 3: 验证推送成功

```bash
# 检查远程仓库
git remote -v
# 应该输出：
# origin  https://github.com/YOUR_USERNAME/political-bias-representation-engineering.git (fetch)
# origin  https://github.com/YOUR_USERNAME/political-bias-representation-engineering.git (push)

# 查看 git 日志
git log --oneline
# 应该显示：
# 21af4a3 (HEAD -> main, origin/main) Initial commit: Political bias...
```

访问你的 GitHub 仓库：
```
https://github.com/YOUR_USERNAME/political-bias-representation-engineering
```

你应该能看到所有的文件和文件夹！

---

## 👥 Step 4: 邀请协作者

### 添加团队成员

1. 进入仓库 → **Settings** → **Collaborators**
2. 点击 "Add people"
3. 输入协作者的 GitHub 用户名
4. 选择权限：
   - **Maintain**: 可以修改设置、合并 PR
   - **Write**: 可以创建分支、推送代码
   - **Read**: 只能读取（不推荐用于开发者）

### 推荐权限设置

```
Role              Who
───────────────────────────────────────
Owner            你（创建者）
Maintain         导师/顾问
Write            其他开发者
Read             外部评审者
```

---

## 🔀 Step 5: 配置分支保护规则（可选）

1. Settings → Branches
2. "Add rule" → Branch name pattern: `main`
3. 勾选：
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging

**效果**：防止直接推送到 main，必须通过 PR

---

## 🤝 Step 6: 协作工作流

### 所有开发者都遵循这个流程：

```bash
# 1. Clone 仓库（第一次）
git clone https://github.com/YOUR_USERNAME/political-bias-representation-engineering.git
cd political-bias-representation-engineering

# 2. 创建特性分支
git checkout -b feature/your-feature-name
# 例如：
# git checkout -b feature/new-model-llama2
# git checkout -b feature/chinese-topics
# git checkout -b bugfix/steering-cuda

# 3. 做出更改
# ... 编辑文件 ...

# 4. 提交（小的、有意义的提交）
git add .
git commit -m "feat: Add description of change"
# 例如：
# git commit -m "feat: Add Llama-2 model support in step1"
# git commit -m "fix: Correct topic_analysis bug in step3"

# 5. 定期拉取最新代码
git pull origin main

# 6. 解决冲突（如果有）
# ... 编辑冲突文件 ...
git add .
git commit -m "merge: Resolve conflicts with main"

# 7. 推送到 GitHub
git push origin feature/your-feature-name

# 8. 在 GitHub 创建 Pull Request (PR)
# - 标题清晰
# - 描述: 做了什么、为什么、如何测试
# - 等待 reviews

# 9. 合并（owner 批准后）
# - "Squash and merge" 或 "Create a merge commit"
```

### 提交消息规范

```
feat: Add new feature
fix: Bug fix
docs: Documentation only
refactor: Code restructuring
test: Add/update tests
chore: Maintenance tasks

示例：
feat: Implement Step 3 topic-level analysis
fix: Correct cosine similarity computation in step1
docs: Update ARCHITECTURE.md with new design
test: Add unit tests for cosine_similarity function
```

---

## 📊 Step 7: 设置 GitHub Actions（CI/CD）（可选）

创建 `.github/workflows/tests.yml`：

```yaml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest tests/ -v
```

这样每次 push 或 PR 时会自动运行测试。

---

## 🎯 常见工作流

### 场景 1: 新增一个模型测试

```bash
# 创建分支
git checkout -b feature/test-mistral-7b

# 创建新文件或修改现有文件
# 例如修改 step1_locate_political_layers.py 以支持 Mistral

# 提交
git add step1_locate_political_layers.py
git commit -m "feat: Add Mistral-7B model support"

# 推送
git push origin feature/test-mistral-7b

# 在 GitHub 创建 PR，等待评审
# 通过后合并
```

### 场景 2: 修复 Bug

```bash
# 基于 main 创建分支
git checkout -b bugfix/steering-hook-cuda

# 修复 bug
# 例如修改 step4_steering.py

git add step4_steering.py
git commit -m "fix: Handle CUDA tensor dtype in steering hook"

git push origin bugfix/steering-hook-cuda

# 创建 PR，添加：
# - 问题描述
# - 如何重现
# - 修复方案
```

### 场景 3: 文档更新

```bash
git checkout -b docs/add-multilingual-guide

# 编辑文档
echo "## Multilingual Support Guide" >> README.md

git add README.md
git commit -m "docs: Add multilingual support documentation"

git push origin docs/add-multilingual-guide
```

---

## 🔍 代码审查流程

### PR 创建者的职责

1. 清晰的标题和描述
2. 链接相关 Issue（如果有）
3. 描述改动的原因和方式
4. 提供测试结果截图/日志
5. 标记为 `WIP` (Work In Progress) 如果还未完成

### 审查者的职责

1. 检查代码质量
2. 确认测试通过
3. 提出改进建议
4. 批准并合并

### 模板（在 PR 描述中使用）

```markdown
## 描述
这个 PR 做了什么？

## 改动原因
为什么需要这个改动？

## 测试方法
如何验证这个改动？

- [ ] 在 M1 Mac 上测试通过
- [ ] 在 CUDA 设备上测试通过
- [ ] 新增单元测试

## 相关 Issue
Closes #123
```

---

## 📱 推荐的 Git 客户端（可选）

| 工具 | 优点 | 缺点 |
|------|------|------|
| GitHub Desktop | 简单易用，可视化界面 | 功能有限 |
| VS Code | 集成开发环境，强大 | 需要学习 |
| GitKraken | 漂亮的 UI，功能全 | 部分功能收费 |
| 命令行 Git | 最强大，完全控制 | 学习曲线陡峭 |

---

## ⚠️ 常见问题

### Q: 我的改动被拒绝了怎么办？

A: 在同一分支上继续改动，然后 push。PR 会自动更新。

```bash
# 在你的分支上继续开发
git add .
git commit -m "Update: Address review comments"
git push origin feature/your-feature-name
```

### Q: 我的分支和 main 产生了冲突怎么办？

A: 从 main 拉取最新代码并解决冲突。

```bash
git fetch origin
git rebase origin/main
# 或
git merge origin/main

# 手动解决冲突文件中的 <<<<<<< >>>>>>
git add .
git commit -m "Merge: Resolve conflicts"
git push origin feature/your-feature-name
```

### Q: 如何删除已合并的分支？

A: 本地和远程都可以删除。

```bash
# 本地删除
git branch -d feature/your-feature-name

# 远程删除
git push origin --delete feature/your-feature-name
```

---

## ✨ 完成！

现在你的项目已经在 GitHub 上，可以多人协作了！

**下一步**：
- ⭐ 在 GitHub 上 Star 你自己的仓库（邀请他人 Star）
- 📢 分享给你的导师和合作者
- 🔔 在 GitHub Issues 中跟踪功能需求
- 📝 更新 README 中的贡献者列表

---

<div align="center">

**Happy collaborating! 🎉**

Have questions? Check GitHub Docs: https://docs.github.com

</div>
