# GitHub Push Tutorial  HW1 Submission

This guide explains how to push your HW1 files to the course GitHub repository.

## Repository Information

| Field | Value |
|---|---|
| Repository URL | `https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW.git` |
| Target Branch | `main` |
| Target Folder | `hw1/` |
| GitHub Username | `fader2077` |

---

## Files to Submit

| File | Description |
|---|---|
| `Stock_predict_v7 copy 11.ipynb` | Main notebook (all phases, full outputs) |
| `best_model_live_final_v7-11.pth` | C2 final model checkpoint (trained on 100% data) |
| `best_model_v7-11_strict.pth` | Best train/val model checkpoint |
| `scaler_live_final_v7-11.pkl` | MinMaxScaler fitted on full data |
| `live_log_v7-11.csv` | 10-day live competition trading log |
| `live_state_v7-11.json` | Final competition state snapshot |
| `HW1_Report.md` | Written report (convert to PDF for final submission) |

---

## Step-by-Step Push Guide

### Prerequisites

- Git installed (`git --version` to verify)
- A GitHub Personal Access Token (PAT) with `repo` scope
  - Create at: **GitHub -> Settings -> Developer Settings -> Personal Access Tokens (classic)**
  - Required scope: `repo` (Full control of private repositories)

---

### Method 1: Using the Pre-configured Local Repo (Recommended)

A local clone already exists at `d:\course\rnnlstm\hw1\_push_temp\`.
The PAT is already embedded in the remote URL.

**Step 1 -- Copy updated files into the push folder:**

```powershell
$src = "d:\course\rnnlstm\hw1\v10"
$dst = "d:\course\rnnlstm\hw1\_push_temp\hw1"

Copy-Item "$src\Stock_predict_v7 copy 112.ipynb" "$dst\Stock_predict_v7 copy 11.ipynb" -Force
Copy-Item "$src\best_model_live_final_v7-11.pth" $dst -Force
Copy-Item "$src\best_model_v7-11_strict.pth" $dst -Force
Copy-Item "$src\scaler_live_final_v7-11.pkl" $dst -Force
Copy-Item "$src\live_log_v7-11.csv" $dst -Force
Copy-Item "$src\live_state_v7-11.json" $dst -Force
Copy-Item "$src\HW1_Report_final.md" "$dst\HW1_Report.md" -Force
```

**Step 2 -- Stage all changes:**

```powershell
Set-Location "d:\course\rnnlstm\hw1\_push_temp"
git add hw1/
git status
```

**Step 3 -- Commit:**

```powershell
git commit -m "hw1: Update notebook with Phase 3 live-log override and final report"
```

**Step 4 -- Push:**

```powershell
git push origin main
```

---

### Method 2: Fresh Clone from Scratch

If the _push_temp folder is missing, create a fresh clone:

**Step 1 -- Clone with PAT embedded in URL:**

```powershell
$pat = "YOUR_GITHUB_PAT_HERE"
$url = "https://fader2077:$pat@github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW.git"
git clone $url "d:\course\rnnlstm\hw1\_push_temp2"
Set-Location "d:\course\rnnlstm\hw1\_push_temp2"
```

**Step 2 -- Create the hw1/ subfolder and copy files:**

```powershell
New-Item -ItemType Directory -Path hw1 -Force
$src = "d:\course\rnnlstm\hw1\v10"
Copy-Item "$src\Stock_predict_v7 copy 112.ipynb" "hw1\Stock_predict_v7 copy 11.ipynb"
Copy-Item "$src\best_model_live_final_v7-11.pth" hw1\
Copy-Item "$src\best_model_v7-11_strict.pth" hw1\
Copy-Item "$src\scaler_live_final_v7-11.pkl" hw1\
Copy-Item "$src\live_log_v7-11.csv" hw1\
Copy-Item "$src\live_state_v7-11.json" hw1\
Copy-Item "$src\HW1_Report_final.md" "hw1\HW1_Report.md"
```

**Step 3 -- Commit and push:**

```powershell
git add hw1/
git commit -m "hw1: LSTM stock prediction (2330.TW, 3/20-4/2 competition window)"
git push origin main
```

---

## Verifying the Push

After pushing, verify at:
https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW/tree/main/hw1

You should see all 7 files listed under hw1/.

---

## Converting HW1_Report.md to PDF

**Option A -- VS Code (Markdown PDF extension):**
1. Install extension: yzane.markdown-pdf
2. Open HW1_Report.md in VS Code
3. Right-click -> Markdown PDF: Export (pdf)

**Option B -- Pandoc (command line):**
```powershell
pandoc HW1_Report.md -o HW1_Report.pdf --pdf-engine=xelatex
```

**Option C -- Browser print:**
1. Open the .md file in VS Code with Markdown Preview (Ctrl+Shift+V)
2. Press Ctrl+P to print -> Save as PDF

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Authentication failed | Regenerate PAT at GitHub -> Settings -> Developer Settings |
| Updates were rejected | Run git pull origin main first, then push again |
| fatal: remote origin already exists | Use git remote set-url origin new-url |
| Large file warning | Normal -- model files are ~560KB, well under 100MB limit |
