# Git 推送教學 — HW2 上傳至 GitHub

## 前置需求

1. **安裝 Git**: [https://git-scm.com/downloads](https://git-scm.com/downloads)
2. **GitHub 帳號**: 已建立且有 repo 權限
3. **認證設定**: 建議使用 GitHub CLI 或 Personal Access Token（PAT）

---

## 方法一：從零開始推送（首次上傳）

### Step 1：Clone 遠端 Repository

```bash
git clone https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW.git
cd Recurrent-Neural-Network-and-Transformer-HW
```

### Step 2：將作業檔案放入 hw2 資料夾

將以下檔案複製到 `hw2/` 中：

```
hw2/
├── .gitignore                  # 排除大型檔案
├── HW2.ipynb                   # 主要 Notebook（包含 Part 1-4）
├── Baseline.py                 # Part 1 Python 腳本
├── BERT.py                     # Part 2 Python 腳本
├── LocalLLM.py                 # Part 3 Python 腳本
├── hw2.txt                     # 作業說明
├── Homework 2.pdf              # 作業 PDF
├── results_baseline/           # Part 1 結果（圖表）
├── results_bert-base-cased/    # Part 2 BERT-base 結果
├── results_bert-large-cased/   # Part 2 BERT-large 結果
└── results_adversarial/        # Part 3 對抗攻擊結果
```

> **注意**：不要上傳以下大型檔案：
> - `saved_model_*/`（模型權重，數 GB）
> - `DAIGT V2 Train Dataset/`（訓練資料集）
> - `llm-detect-ai-generated-text/`（競賽資料集）
> - `data_split.pkl`（~96MB，超過 GitHub 限制）

### Step 3：建立 .gitignore

在 `hw2/` 下建立 `.gitignore`：

```gitignore
# Large model files
saved_model_*/
checkpoint-*/

# Dataset
DAIGT V2 Train Dataset/
llm-detect-ai-generated-text/

# Large data files
*.pkl

# Python cache
__pycache__/
*.pyc

# Training logs
*.log
bert_*_log*.txt
```

### Step 4：加入暫存區並提交

```bash
git add hw2/
git commit -m "HW2: Detect AI Generated Text - complete submission"
```

### Step 5：推送到 GitHub

```bash
git push origin main
```

---

## 方法二：在現有本機 Repo 更新

如果之前已 clone 過該 repo：

```bash
cd Recurrent-Neural-Network-and-Transformer-HW

# 拉取最新版本
git pull origin main

# 複製/更新 hw2 資料夾的內容
# (手動將檔案放入 hw2/)

# 加入暫存區
git add hw2/

# 檢查狀態
git status

# 提交
git commit -m "Update HW2 submission"

# 推送
git push origin main
```

---

## 常用 Git 指令速查

| 指令 | 說明 |
|------|------|
| `git clone <url>` | 複製遠端 repo 到本機 |
| `git status` | 查看目前變更狀態 |
| `git add <path>` | 將檔案加入暫存區 |
| `git commit -m "msg"` | 提交變更 |
| `git push origin main` | 推送到遠端 main 分支 |
| `git pull origin main` | 從遠端拉取最新內容 |
| `git log --oneline -5` | 查看近 5 筆 commit 記錄 |
| `git diff` | 查看未暫存的變更 |

---

## 常見問題排解

### Q1：推送時出現認證錯誤
```bash
# 使用 Personal Access Token
# GitHub → Settings → Developer settings → Personal access tokens → Generate new token
# 推送時輸入 token 作為密碼

# 或安裝 GitHub CLI
winget install GitHub.cli
gh auth login
```

### Q2：檔案太大無法推送（超過 100MB）
```bash
# 確認 .gitignore 有排除大檔案
# 如果已經 commit 了大檔案：
git rm --cached <large_file>
git commit -m "Remove large file"
git push
```

### Q3：PowerShell 顯示紅色錯誤但其實成功了
Git 會將進度訊息輸出到 stderr，PowerShell 會將其顯示為紅色。  
只要看到類似 `main -> main` 的訊息，表示推送成功。

### Q4：push 被拒絕（non-fast-forward）
```bash
# 先拉取遠端變更再推送
git pull origin main --rebase
git push origin main
```

---

## 實際推送記錄

### 第一次推送（初始提交）
```
[main 359f409] HW2: Detect AI Generated Text - complete submission
 16 files changed, 2922 insertions(+)
 → b6972ac..359f409  main -> main
```

### 第二次推送（增強實驗）
```
[main 4853a55] HW2: Enhanced experiments - statistical EDA, multiple baselines,
 BERT ablation studies, iterative adversarial attacks, comprehensive analysis
 8 files changed, 1289 insertions(+), 46 deletions(-)
 → 359f409..4853a55  main -> main
```

新增/更新檔案：
- `HW2.ipynb` — 擴充至 62 cells，新增統計檢定、多模型 baseline、BERT 消融實驗、迭代對抗攻擊
- `results_baseline/enhanced_eda.png` — 增強 EDA（Mann-Whitney U + Cohen's d）
- `results_baseline/baseline_confusion_matrices.png` — LR/SVM/NB/RF 混淆矩陣
- `results_bert-base-cased/bert_confusion_confidence.png` — BERT 混淆矩陣 + 信心分佈
- `results_bert-base-cased/epoch_ablation.png` — 3 vs 5 epoch 消融比較
- `results_adversarial/enhanced_adversarial_analysis.png` — 對抗特徵分析視覺化
- `results_adversarial/adversarial_results.json` — 新增迭代攻擊結果
- `results_bert-base-cased/results_summary.json` — 更新含消融實驗數據

**推送成功！** 可至以下網址確認：  
https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW/tree/main/hw2
