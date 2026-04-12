# HW1 LaTeX 在 VS Code 編譯教學（嚴謹版）

本文件提供可重現、可稽核的 VS Code LaTeX 編譯流程，目標是穩定產出 `HW1_Report.pdf`，並避免 Overleaf timeout 問題。

## 1. 目標與檔案

- 來源檔：`HW1_Report.tex`
- 輸出檔：`HW1_Report.pdf`
- 建議編譯器：`XeLaTeX`（支援中文）
- 建議平台：Windows + MiKTeX + VS Code + LaTeX Workshop

## 2. 環境安裝

1. 安裝 VS Code 擴充套件
- `LaTeX Workshop`（作者 James Yu）

2. 安裝 TeX 發行版（擇一）
- MiKTeX（Windows 建議）
- TeX Live

3. 驗證可執行檔（PowerShell）
```powershell
Get-Command xelatex, latexmk -ErrorAction SilentlyContinue
```

若找不到 `xelatex`，可使用絕對路徑，例如：
```powershell
"C:\Users\user\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe"
```

## 3. VS Code 設定（建議）

在專案 `.vscode/settings.json` 加入：

```json
{
  "latex-workshop.latex.tools": [
    {
      "name": "xelatex-strict",
      "command": "xelatex",
      "args": [
        "-halt-on-error",
        "-file-line-error",
        "-interaction=nonstopmode",
        "-synctex=0",
        "%DOC%"
      ]
    }
  ],
  "latex-workshop.latex.recipes": [
    {
      "name": "XeLaTeX x2 (strict)",
      "tools": ["xelatex-strict", "xelatex-strict"]
    }
  ],
  "latex-workshop.latex.autoBuild.run": "never"
}
```

說明：
- `-halt-on-error`：遇到致命錯誤即停止。
- `-file-line-error`：輸出檔名與行號，便於精準修正。
- `-interaction=nonstopmode`：不中斷等待互動輸入。
- 兩次 XeLaTeX：確保目錄、引用、超連結穩定。

## 4. 編譯流程（UI）

1. 開啟 `HW1_Report.tex`
2. `Ctrl+Shift+P` -> `LaTeX Workshop: Build with recipe`
3. 選擇 `XeLaTeX x2 (strict)`
4. 檢查 OUTPUT 視窗：
- 不可有 `Fatal error`、`Emergency stop`、`Runaway argument`
- 可容忍 `Overfull \hbox`（版面警告，非致命）

## 5. 編譯流程（命令列，完全可重現）

```powershell
$xe = "C:\Users\user\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe"
& $xe -halt-on-error -file-line-error -interaction=nonstopmode -synctex=0 -output-directory="d:\course\rnnlstm\hw1\v10" "d:\course\rnnlstm\hw1\v10\HW1_Report.tex"
& $xe -halt-on-error -file-line-error -interaction=nonstopmode -synctex=0 -output-directory="d:\course\rnnlstm\hw1\v10" "d:\course\rnnlstm\hw1\v10\HW1_Report.tex"
```

## 6. 嚴謹驗證清單

1. 檔案存在
```powershell
Get-Item "d:\course\rnnlstm\hw1\v10\HW1_Report.pdf"
```

2. 日誌無致命錯誤
```powershell
rg -n "^!|Emergency stop|Fatal error|Illegal unit|Runaway argument" "d:\course\rnnlstm\hw1\v10\HW1_Report.log"
```

3. PDF 時間戳更新（代表最新輸出）
```powershell
Get-Item "d:\course\rnnlstm\hw1\v10\HW1_Report.pdf" | Select-Object Length, LastWriteTime
```

## 7. 常見問題排除

1. 中文字型找不到（例如 SimHei）
- 改用明確字型設定：
```latex
\usepackage{fontspec}
\usepackage{xeCJK}
\setmainfont{Times New Roman}
\setCJKmainfont{Microsoft JhengHei UI}
```

2. `Illegal unit of measure (pt inserted)`
- 常見於表格列開頭誤被當成選項參數。
- 若列內容以 `[` 開頭，改成 `{}[` 開頭。

3. Overleaf timeout
- 改在本機 VS Code 編譯（無免費版 1 分鐘限制）。
- 若仍需 Overleaf，務必切換 Compiler 為 XeLaTeX。

4. MiKTeX update 提示
- 非致命，但建議定期更新以避免套件相容問題。

## 8. 本專案建議提交內容

- `HW1_Report.tex`
- `HW1_Report.pdf`
- `HW1_LaTeX_VSCode_Compile_Guide.md`

以上三檔可完整重現報告與編譯流程。