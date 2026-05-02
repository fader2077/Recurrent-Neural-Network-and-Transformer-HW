# HW3: RAG for Science Question Answering - GitHub Push Instructions

**最后更新**: 2026年5月3日  
**项目所有者**: fader2077  
**目标仓库**: https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW.git

---

## 📋 目录
1. [项目概述](#项目概述)
2. [推送前准备](#推送前准备)
3. [完整推送步骤](#完整推送步骤)
4. [项目文件结构](#项目文件结构)
5. [常见问题与解决方案](#常见问题与解决方案)
6. [验证推送成功](#验证推送成功)

---

## 项目概述

### 内容说明
本项目实现了一个完整的 **RAG (Retrieval-Augmented Generation) Pipeline**，应用于 Kaggle LLM Science Exam 数据集，实现科学问答任务。

### 核心功能
- **Part 1**: 数据工程 - 比较两种分塊策略的效果
- **Part 2**: 检索与重排 - Dense Retrieval + Cross-Encoder Re-ranking
- **Part 3**: LLM 生成 - 本地 Ollama (Llama-3) 集成
- **Part 4**: 性能评估 - Hit Rate、延迟分析、准确率评估

### 最终成果
- 验证集准确率: **80.00%** (40/50 题)
- 平均延迟: **2.27 秒/查询**
- 生成文档: LaTeX 报告 + Jupyter Notebook

---

## 推送前准备

### ✅ 环境检查清单
```bash
# 1. 验证 Git 安装
git --version
# 应输出: git version 2.x.x 或更新

# 2. 验证 Git 配置
git config --global user.name "fader2077"
git config --global user.email "fader2077@github.com"

# 3. 进入项目目录
cd D:\course\rnnlstm\HW3
pwd  # 或 Get-Location (PowerShell)
```

### 📦 项目内容验证

本次提交包含以下主要内容：

#### 源代码文件
- `hw3.ipynb` - 主要 Notebook (完整实验)
- `hw3advance.ipynb` - 高级实验 Notebook
- `build_notebook.py` - Notebook 构建脚本
- `Chunking.py` - 文本分塊策略实现
- `Retrieval.py` - 检索系统核心
- `Generation.py` - LLM 生成模块
- `DB.py` - 数据库操作
- `official_eval.py` - 官方评估脚本

#### 数据文件
- `train.csv` - 训练数据集
- `kaggle-llm-science-exam/` - Kaggle 数据集
- `chroma_db_fixed/` - ChromaDB 向量索引 (固定分塊)
- `chroma_db_semantic/` - ChromaDB 向量索引 (语意分塊)
- `faiss_index_a/`, `faiss_index_b/` - FAISS 索引

#### 结果与评估
- `evaluation_results.csv` - 基础评估结果
- `official_evaluation_results.json` - 官方数据集评估
- `advanced_rag_results.csv` - 高级 RAG 结果
- `experiment_summary.json` - 实验总结
- `official_hit_rate_comparison.csv` - Hit Rate 对比

#### 可视化与报告
- `chunk_length_distribution.png` - 分塊长度分布图
- `hit_rate_comparison.png` - Hit Rate 比较图
- `latency_analysis.png` - 延迟分析图
- `advanced_accuracy_comparison.png` - 准确率对比图
- `report.pdf` - LaTeX 编译后的报告 (8页)
- `report.tex` - LaTeX 源文件
- `report.md` - Markdown 版本报告

#### 论文与文档
- `yi/main.tex` - 论文 LaTeX 源文件 (带中文)
- `yi/sample.bib` - 参考文献库
- `yi/main.pdf` - 编译后的论文 PDF
- `yi/*.png` - 论文中使用的图片

#### 中间与辅助文件
- 各种 JSON/CSV 结果文件 (候选、重排、示例)
- `Homework 3.txt` - 原始作业说明
- `Homework 3.pdf` - 作业 PDF

---

## 完整推送步骤

### 方案 A: 使用 HTTPS (Token 认证) - **推荐用于首次推送**

> ⚠️ **前置条件**: 需要 GitHub Personal Access Token  
> 获取方法: GitHub → Settings → Developer settings → Personal access tokens → Generate new token

**Step 1: 配置本地 Git 用户信息**
```bash
cd D:\course\rnnlstm\HW3
git config user.name "fader2077"
git config user.email "fader2077@github.com"
```

**Step 2: 添加或验证远程仓库**
```bash
# 检查现有远程
git remote -v

# 如果未配置，添加远程
git remote add origin https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW.git

# 如果已存在，更新 URL
git remote set-url origin https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW.git
```

**Step 3: 检查变更状态**
```bash
# 查看所有未追踪或已修改的文件
git status
```

**Step 4: 推荐分阶段提交**

```bash
# 阶段 1: 添加源代码
git add *.py hw3.ipynb hw3advance.ipynb build_notebook.py official_eval.py
git commit -m "feat: Add HW3 source code - RAG pipeline implementation"

# 阶段 2: 添加数据
git add train.csv kaggle-llm-science-exam/
git commit -m "data: Add training data and Kaggle dataset"

# 阶段 3: 添加向量索引
git add chroma_db_fixed/ chroma_db_semantic/ faiss_index_a/ faiss_index_b/
git commit -m "data: Add vector indices (ChromaDB and FAISS)"

# 阶段 4: 添加评估结果
git add *.csv *.json
git commit -m "results: Add evaluation metrics and experiment results"

# 阶段 5: 添加可视化和报告
git add *.png *.pdf *.tex *.md "Homework 3.txt"
git commit -m "docs: Add visualizations, LaTeX report, and documentation"

# 阶段 6: 添加论文文件
git add yi/
git commit -m "docs: Add thesis paper (main.tex, main.pdf) with figures"
```

**Step 5: 一次性全量提交 (所有文件)**
```bash
# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: Complete HW3 RAG pipeline project

- Implemented two-stage retrieval system (Dense + Cross-Encoder Re-ranking)
- Compared chunking strategies: Fixed-size vs Semantic
- Strategy A: 500-char chunks (120,206 chunks, 76% Hit Rate)
- Strategy B: 1000-char chunks (57,970 chunks, 80% Hit Rate)
- Integrated Ollama (Llama-3) for local generation
- Achieved 80% accuracy on 50-question validation set
- Average latency: 2.27s per query (LLM generation dominates at 98.7%)
- Included complete LaTeX report, Jupyter notebooks, and evaluation results
- Vector indices: ChromaDB and FAISS
- Re-ranking model: cross-encoder/ms-marco-MiniLM-L-6-v2"
```

**Step 6: 推送到 GitHub**
```bash
# 推送到 hw3 分支 (新分支，推荐)
git branch -M hw3
git push -u origin hw3

# 或推送到 main 分支
git push -u origin main
```

---

### 方案 B: 使用 SSH 密钥 (更安全) - **适合长期开发**

前置条件：已配置 GitHub SSH 密钥

```bash
# 验证 SSH 连接
ssh -T git@github.com

# 添加 SSH 远程
git remote set-url origin git@github.com:fader2077/Recurrent-Neural-Network-and-Transformer-HW.git

# 提交和推送
git add .
git commit -m "Initial commit: HW3 RAG pipeline"
git push -u origin hw3
```

---

### 方案 C: 完整快速推送 (复制粘贴)

```bash
# 1. 进入目录
cd D:\course\rnnlstm\HW3

# 2. 配置 Git
git config user.name "fader2077"
git config user.email "fader2077@github.com"

# 3. 添加远程 (如果需要)
git remote set-url origin https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW.git 2>/dev/null || git remote add origin https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW.git

# 4. 添加所有文件
git add .

# 5. 提交
git commit -m "feat: HW3 RAG pipeline - Initial commit"

# 6. 推送
git branch -M hw3
git push -u origin hw3

# 7. 验证
git log --oneline -1
git remote -v
```

---

## 项目文件结构

推送后的 GitHub 仓库结构：

```
Recurrent-Neural-Network-and-Transformer-HW/
├── hw3/                              # ← HW3 项目文件夹
│   ├── 源代码/
│   │   ├── hw3.ipynb                 # 主 Notebook
│   │   ├── hw3advance.ipynb          # 高级 Notebook
│   │   ├── Chunking.py               # 分塊模块
│   │   ├── Retrieval.py              # 检索模块
│   │   ├── Generation.py             # 生成模块
│   │   ├── DB.py                     # 数据库模块
│   │   ├── official_eval.py          # 评估脚本
│   │   └── build_notebook.py         # 构建脚本
│   │
│   ├── 数据/
│   │   ├── train.csv                 # 训练数据
│   │   ├── kaggle-llm-science-exam/  # Kaggle 数据集
│   │   ├── chroma_db_fixed/          # ChromaDB 索引 A
│   │   ├── chroma_db_semantic/       # ChromaDB 索引 B
│   │   ├── faiss_index_a/            # FAISS 索引 A
│   │   └── faiss_index_b/            # FAISS 索引 B
│   │
│   ├── 结果/
│   │   ├── *.csv                     # 评估结果
│   │   ├── *.json                    # 实验摘要
│   │   └── *.json (候选/重排示例)    # 详细结果
│   │
│   ├── 可视化/
│   │   ├── chunk_length_distribution.png
│   │   ├── hit_rate_comparison.png
│   │   ├── latency_analysis.png
│   │   └── advanced_accuracy_comparison.png
│   │
│   ├── 报告/
│   │   ├── report.pdf                # 编译后的 PDF
│   │   ├── report.tex                # LaTeX 源文件
│   │   ├── report.md                 # Markdown 版本
│   │   ├── Homework 3.pdf            # 作业 PDF
│   │   └── Homework 3.txt            # 作业文本
│   │
│   ├── 论文 (yi/)/
│   │   ├── main.tex                  # 论文源文件 (中文)
│   │   ├── main.pdf                  # 编译后论文
│   │   ├── sample.bib                # 参考文献
│   │   └── *.png (论文图片)          # 图表文件
│   │
│   └── PUSH_INSTRUCTIONS.md          # 本推送指南

└── [其他作业文件夹: hw1, hw2 等]
```

---

## 常见问题与解决方案

### Q1: 推送时提示 "fatal: 'origin' does not appear to be a 'git' repository"

**解决**:
```bash
git remote add origin https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW.git
git remote -v
```

---

### Q2: 认证失败 "401 Unauthorized"

**HTTPS 解决**:
```bash
# 清除并重新输入
git credential reject host=github.com
git push  # 会要求重新认证
```

**SSH 解决**:
```bash
ssh -vT git@github.com
# 如果失败，重新生成 SSH 密钥
ssh-keygen -t rsa -b 4096 -C "fader2077@github.com"
```

---

### Q3: 文件太大，超过 100MB 限制

**解决 (Git LFS)**:
```bash
git lfs install
git lfs track "*.csv" "*.json" "*.pdf" "*.pth" "*.bin"
git add .gitattributes
git commit -m "Configure Git LFS"
git push
```

---

### Q4: 推送后文件结构不对

**验证**:
```bash
# 查看远程文件
git ls-tree -r --name-only origin/hw3 | head -20

# 查看本地文件
git ls-files | head -20
```

---

## 验证推送成功

### ✅ 本地验证

```bash
# 查看提交历史
git log --oneline -5

# 查看远程分支
git branch -r

# 验证状态
git status
# 应输出: Your branch is up to date with 'origin/hw3'
```

### ✅ GitHub 网页验证

1. 登录 GitHub → https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW
2. 切换分支 → 选择 `hw3` 分支
3. 检查项目：
   - ✓ `hw3/` 文件夹存在
   - ✓ 所有 Python 文件可见
   - ✓ `*.ipynb` 笔记本可见
   - ✓ 数据文件夹存在
   - ✓ 报告和图表可见

### ✅ 克隆验证 (最严格)

```bash
git clone https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW.git test_clone
cd test_clone
git checkout hw3
ls hw3/  # 应显示所有文件
find hw3 -type f | wc -l  # 统计文件数
```

---

## 最后检查清单

- [ ] Git 已安装 (`git --version`)
- [ ] 用户配置完成 (`git config user.name/email`)
- [ ] 远程仓库已配置 (`git remote -v`)
- [ ] 所有文件已添加 (`git add .`)
- [ ] 提交消息清晰 (`git commit`)
- [ ] 无未提交更改 (`git status`)
- [ ] 分支正确 (`git branch`)
- [ ] 推送完成 (`git push`)
- [ ] GitHub 网页可见
- [ ] 文件结构正确

---

**文档版本**: 2.0  
**创建日期**: 2026年5月3日  
**作者**: HW3 RAG Pipeline Team
