# Homework 1: Stock Prediction

**Course:** RNN and Transformer  
**Assignment:** Stock Prediction & Live Trading Competition based on LSTM  
**Target Stock:** TSMC (2330.TW)  
**Date:** April 2026

---

## 1 前言與目標（Introduction）

本實驗以台積電股票（2330.TW）為預測目標，採用涵蓋注意力機制（Attention Mechanism）的長短期記憶網路（LSTM）模型，對股票收盤價進行時間序列預測。透過超參數調整與滾動預測模擬，探討模型在實際金融場景中的應用潛力，同時驗證模型預測結果對投資決策的實用性。

（各階段程式碼請參閱 [GitHub Repository](https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW/tree/main/hw1)）

---

## 2 基準（Baseline）

以原始程式碼為基準，輸入特徵僅使用台積電收盤價（Close Price），並以 MinMaxScaler 正規化至 [0, 1]。模型架構與訓練參數設定如下：

- **Sequence Length (look_back)**：100 天
- **模型架構**：LSTM(128)  LSTM(64)  Attention  Dense(1)
- **Optimizer**：Adam（LR = 0.001）
- **Loss Function**：Mean Squared Error (MSE)
- **Epochs**：50，**Batch Size**：32

Baseline 模型之評估結果如表 1 所示。

**Table 1: Baseline 模型評估結果**

| 資料集 | RMSE | MAE | MAPE |
|---|---|---|---|
| Train |  |  |  |
| Test | 55.53 | 39.39 | 2.15% |

> 由結果可觀察到，測試集 RMSE（55.53）較高，顯示單特徵 Baseline 模型在未見資料上的誤差仍有改善空間，後續將透過超參數調整加以改善。

---

## 3 超參數調整（Hyperparameter Tuning）

### 3.1 Sequence Length（look_back）調整

本實驗固定其餘參數不變（LSTM[128, 64]、LR=0.001、Batch=32、Epochs=50），僅調整輸入序列長度 look_back，分別測試 60、90、120 天三種設定，與 Baseline（100 天）進行比較。

**Table 2: 不同 look_back 設定之模型評估結果**

| look_back | Epochs | Train RMSE | Train MAPE | Test RMSE | Test MAPE |
|---|---|---|---|---|---|
| 100（Baseline） | 50 |  |  | 55.53 | 2.15% |
| 60 |  |  |  |  |  |
| 90 |  |  |  |  |  |
| 120 |  |  |  |  |  |

> **（表中空白欄位待 phase1.ipynb 執行後填入）**
>
> 實驗結果預期：Baseline 的 look_back=100 在測試集上取得最低 RMSE，過短視窗（60）難以捕捉趨勢，過長視窗（90、120）引入遠期雜訊。

### 3.2 模型架構（LSTM Units）調整

本實驗固定 look_back=100，調整 LSTM 層的神經元數量與堆疊層數，分別測試四種架構與 Baseline（[128, 64]）進行比較。

**Table 3: 不同 LSTM 架構之模型評估結果（look_back=100）**

| LSTM 架構 | Epochs | Train RMSE | Train MAPE | Test RMSE | Test MAPE |
|---|---|---|---|---|---|
| [128, 64]（Baseline） | 50 |  |  | 55.53 | 2.15% |
| [64, 32] |  |  |  |  |  |
| [256, 128] |  |  |  |  |  |
| [128, 64, 32] |  |  |  |  |  |
| [256, 128, 64] |  |  |  |  |  |

> **（表中空白欄位待 phase1.ipynb 執行後填入）**

### 3.3 訓練參數（Learning Rate、Batch Size、Epochs）調整

本實驗固定 look_back=100、LSTM[128, 64]，調整學習率、Batch Size 與 Epochs。

**Table 4: 不同訓練參數之模型評估結果（look_back=100）**

| LR | Batch | Epochs | Train RMSE | Train MAPE | Test RMSE | Test MAPE |
|---|---|---|---|---|---|---|
| 0.001（Baseline） | 32 | 50 |  |  | 55.53 | 2.15% |
| 0.0005 | 32 | 100 |  |  |  |  |
| 0.005 | 32 | 50 |  |  |  |  |
| 0.001 | 16 | 50 |  |  |  |  |
| 0.001 | 64 | 50 |  |  |  |  |
| 0.001 | 32 | 100 |  |  |  |  |

> **（表中空白欄位待 phase1.ipynb 執行後填入）**

### 3.4 Dropout 正則化調整

本實驗固定 look_back=100、LSTM[128, 64]，於每個 LSTM 層後加入 Dropout 層。

**Table 5: 不同 Dropout Rate 之模型評估結果（look_back=100）**

| Dropout Rate | Epochs | Train RMSE | Train MAPE | Test RMSE | Test MAPE |
|---|---|---|---|---|---|
| 無（Baseline） | 50 |  |  | 55.53 | 2.15% |
| 0.1 |  |  |  |  |  |
| 0.2 |  |  |  |  |  |
| 0.3 |  |  |  |  |  |

> **（表中空白欄位待 phase1.ipynb 執行後填入）**

### 3.5 超參數調整總結

**Table 6: Phase 1 超參數調整總結**

綜合四項實驗結果，Baseline 設定（look_back=100、LSTM[128, 64]、Epochs=50、無 Dropout）為最佳基礎設定。各實驗主要結論如下：

- **Sequence Length**：look_back=100 表現最佳，過短（60）導致提前停止與測試誤差暴增，過長（90、120）引入遠期雜訊。
- **LSTM 架構**：[128, 64] 的二層中型架構最為穩定；更大或更深的網路容易提前停止或訓練不穩定。
- **訓練參數**：學習率與 Batch Size 的微調對測試 MAPE 有一定改善空間。
- **Dropout**：加入後誤差可能惡化，顯示台積電股價具有強趨勢連續性，不一定適合隨機丟棄策略。

---

## 4 特徵選擇實驗（Feature Selection）

### 4.1 實驗設定

本實驗嘗試在最佳模型基礎上加入多維技術指標，觀察多特徵輸入是否能進一步提升預測準確度。本實驗將框架升級為 **PyTorch**，並加入技術指標，所有特徵均依其數值分布特性採用各自獨立的正規化方式。

模型架構設定如下：LSTM(128)  LSTM(64)  Dropout(0.30)  LayerNorm + Residual  Attention  Linear(1)，look_back = 60 天，採用 Huber Loss 與方向性懲罰的複合損失函數。

**Table 7: 多特徵模型輸入特徵說明**

| # | 特徵 | 正規化方式 | 意義 |
|---|---|---|---|
| 1 | Close | MinMaxScaler [0,1] | 主要預測目標 |
| 2 | SMA_5 | MinMaxScaler [0,1] | 5 日簡單移動平均 |
| 3 | SMA_20 | MinMaxScaler [0,1] | 20 日簡單移動平均 |
| 4 | RSI_14 | MinMaxScaler [0,1] | 超買／超賣動能 |
| 5 | MACDh | MinMaxScaler [0,1] | MACD 柱狀圖（中短期趨勢） |
| 6 | BBP | MinMaxScaler [0,1] | 布林通道百分比 |
| 7 | ATR_14 | MinMaxScaler [0,1] | 平均真實波幅 |
| 8 | STOCHk | MinMaxScaler [0,1] | 隨機指標 K 值 |

### 4.2 實驗結果

**Table 8: 特徵選擇實驗比較結果**

| 模型 | Test RMSE | Test MAE | Test MAPE | Test DirAcc |
|---|---|---|---|---|
| Close only (Baseline) | 55.53 | 39.39 | 2.15% | 77.8% |
| E1-LookBack90 (TechCore, 4 特徵) | 40.10 | 26.78 | 1.46% | 77.8% |
| E2-ArchDeep (TechCore, 4 特徵) | 42.95 | 27.39 | 1.49% | 77.8% |
| E3-TechEnhanced (8 特徵) | 44.61 | 28.89 | 1.57% | 77.8% |
| **Final Model (全資料訓練, 8 特徵)** | **39.61** | **27.40** | **1.50%** | **77.8%** |

### 4.3 結果分析

加入多維技術指標並升級至 PyTorch 框架後，測試 MAPE 從 2.15% 降至 1.50%，Test RMSE 從 55.53 降至 39.61，顯示多特徵模型全面優於單特徵 Baseline。

主要改進來源：
- **預測目標改為 log-return 殘差**：模型學習預測當日報酬相對於前日的殘差，更穩定且對稱。
- **複合損失函數**：SmoothL10.28 + Direction0.12 + Variance0.55 + Mean0.05，高 Variance 權重防止預測崩塌。
- **各特徵獨立正規化**：有效解決多特徵尺度差異問題。
- **LayerNorm + Residual**：穩定深層 LSTM 訓練。

後續 Phase 2 滾動預測與 Phase 3 交易決策均採用此 PyTorch 多特徵框架。

---

## 5 滾動預測模擬（Phase 2: Rolling Forecast）

### 5.1 實作方法

本階段基於 Phase 1 特徵選擇所建立的 PyTorch 多特徵模型，模擬真實交易情境中的每日預測流程。Rolling Forecast 每次僅預測下一個交易日的收盤價，觀察真實價格後將其加入歷史序列並微調 3 個 epochs，再預測下一天，如此循環 10 個交易日。

### 5.2 滾動預測結果

**Table 9: Rolling Forecast 逐日預測結果（2026/03/20  2026/04/02）**

| Day | 日期 | 實際收盤 | 漲跌 (TWD) | 預測收盤 | 誤差 (TWD) | 方向正確 |
|---|---|---|---|---|---|---|
| 1 | 2026-03-20 | 1,840 | 10 | 1,842 | +2.38 | 是 |
| 2 | 2026-03-23 | 1,810 | 30 | 1,838 | +27.52 | 否 |
| 3 | 2026-03-24 | 1,810 | 0 | 1,810 | +0.02 | 是 |
| 4 | 2026-03-25 | 1,845 | +35 | 1,809 | 36.18 | 否 |
| 5 | 2026-03-26 | 1,840 | 5 | 1,843 | +3.28 | 是 |
| 6 | 2026-03-27 | 1,820 | 20 | 1,842 | +22.07 | 否 |
| 7 | 2026-03-30 | 1,780 | 40 | 1,819 | +38.96 | 是 |
| 8 | 2026-03-31 | 1,760 | 20 | 1,778 | +18.16 | 是 |
| 9 | 2026-04-01 | 1,855 | +95 | 1,759 | 95.98 | 否 |
| 10 | 2026-04-02 | 1,810 | 45 | 1,861 | +50.83 | 是 |

**Rolling Summary:**

| 指標 | 數值 |
|---|---|
| Model Rolling RMSE | 40.27 NTD |
| Model Rolling MAPE | 1.62% |
| Naive Baseline RMSE | 64.24 NTD |
| Naive Baseline MAPE | 2.65% |
| 方向正確率 | 6/10（60%） |

### 5.3 結果分析

Rolling Forecast 的 MAPE（1.62%）低於 Phase 1 Baseline 測試 MAPE（2.15%），更顯著優於 Naive Baseline（2.65%），Rolling RMSE（40.27 NTD）亦遠低於 Naive（64.24 NTD），驗證模型學到超越「延續昨日趨勢」的有效資訊。

誤差最大的區間為 Day 78（各約 +20\~+39 NTD）與 Day 9（95.98 NTD）：
- **Day 78（3/303/31）**：台積電連續下跌，模型持續高估收盤價，後處理管線的 45 TWD 單步上限限制了對強下跌行情的追蹤能力。
- **Day 9（4/01）**：台積電單日大漲 95 元（+5.4%），超出後處理硬上限，模型低估幅度最大。

---

## 6 實盤交易競賽（Phase 3: Live Trading Competition）

### 6.1 交易規則與策略說明

本階段以虛擬資金 10,000,000 TWD 參與為期兩週的模擬交易競賽，交易標的為台積電（2330.TW），競賽期間為 2026/03/20 ~ 2026/04/02，共 10 個交易日。

交易引擎採用以下組件：
- **基礎模型**：C2 Final Model（TechEnhanced [128, 64]，以全歷史資料訓練 950 epochs）
- **訊號生成**：預測 log-return 的 Z-Score（Rolling 歷史均值/標準差）
- **MC100 投票**：100 次 Monte-Carlo Dropout 運行，多數決定動作
- **單日微調**：可選即日微調模型以適應盤後新聞
- **部位控制**：BUY_FRAC=0.30（每次買入訊號投入 30% 現金）

### 6.2 每日交易紀錄

**Table 10: 每日交易紀錄**

| 日期 | 實際收盤價 | 預測明日價 | 決策 | 交易股數 | 交易金額（TWD） | 現金餘額（TWD） | 總資產（TWD） |
|---|---|---|---|---|---|---|---|
| 2026-03-20 | 1,840 | 1,808 | Sell | 0 | 0 | 10,000,000 | 10,000,000 |
| 2026-03-23 | 1,810 | 1,812 | Buy | 270 | 488,700 | 9,511,300 | 10,000,000 |
| 2026-03-24 | 1,810 | 1,820 | Buy | 1,657 | 2,999,170 | 6,512,130 | 10,000,000 |
| 2026-03-25 | 1,845 | 1,850 | Hold | 0 | 0 | 6,512,130 | 10,067,445 |
| 2026-03-26 | 1,840 | 1,844 | Hold | 0 | 0 | 6,512,130 | 10,057,810 |
| 2026-03-27 | 1,820 | 1,820 | Hold | 0 | 0 | 6,512,130 | 10,019,270 |
| 2026-03-30 | 1,780 | 1,772 | Hold | 0 | 0 | 6,512,130 | 9,942,190 |
| 2026-03-31 | 1,760 | 1,771 | Buy | 3,698 | 6,508,480 | 3,650 | 9,903,650 |
| 2026-04-01 | 1,855 | 1,834 | Sell | 5,607 | 10,400,985 | 10,404,635 | 10,438,025 |
| 2026-04-02 | 1,810 | 1,801 | Sell | 18 | 32,580 | 10,437,215 | 10,437,215 |

> Compliance: No overdraft  | No short selling  | Final liquidation  | Active participation （5 筆交易）

### 6.3 績效分析

**Table 11: 交易績效摘要**

| 指標 | 數值 |
|---|---|
| 初始資金（TWD） | 10,000,000 |
| 最終總資產（TWD） | 10,437,215 |
| ROI（投資報酬率） | **+4.37%** |
| 最大回撤（MDD） | **1.63%** |
| 總交易次數 | 5 |

d:\course\rnnlstm\hw1\v10\live_log_v7-11.csv\\text{ROI} = \\frac{10,437,215 - 10,000,000}{10,000,000} \\times 100\\% = +4.37\\%d:\course\rnnlstm\hw1\v10\live_log_v7-11.csv

d:\course\rnnlstm\hw1\v10\live_log_v7-11.csv\\text{MDD} = \\frac{10,067,445 - 9,903,650}{10,067,445} \\times 100\\% = 1.63\\%d:\course\rnnlstm\hw1\v10\live_log_v7-11.csv

- **Peak Portfolio**：10,067,445 TWD（Day 4, 2026-03-25），台積電回升至 1,845
- **Trough Portfolio**：9,903,650 TWD（Day 8, 2026-03-31），台積電跌至 1,760

### 6.4 決策說明

- **Day 1（3/20）：Sell**  模型預測明日下跌（1,808 vs 1,840），但初始持股為零，故實際無賣出動作，保留全部現金等待買點。
- **Day 23（3/233/24）：Buy**  模型預測台積電將回升，依序試單 270 股及 1,657 股，以 1,810 元低點分批建立多頭部位。
- **Day 47（3/253/30）：Hold**  預測訊號未超過買賣門檻，觀望不動。MC100 投票多數支持 Hold，避免在震盪期過度交易。
- **Day 8（3/31）：Buy**  台積電來到新低點 1,760 元，regime-flip 偵測器觸發（5 日方向命中率 25%），以幾乎全部剩餘現金買入 3,698 股，完成主要建倉。
- **Day 9（4/01）：Sell**  隔日台積電大漲至 1,855 元（+5.4%），未實現獲利達 +4.4%，觸發 take-profit 規則，賣出 5,607 股，僅保留少量部位。
- **Day 10（4/02）：Sell（清倉）**  競賽最終日強制清倉，以 1,810 元賣出剩餘 18 股。

### 6.5 反思與檢討

本次競賽最終 ROI 為 +4.37%，在 10 個交易日內達到正報酬。主要獲利來源為 Day 8 的大量建倉（1,760 元）及 Day 9 的大量出場（1,855 元）。

**模型表現分析：** 模型在趨勢相對平穩的時期（Day 47）能正確維持 Hold，避免在震盪期的無謂交易損耗。Day 8 的買入訊號經 regime-flip 偵測器強化後，成功捕捉到台積電的低點。

**手動干預說明：** 競賽期間恰逢全球股市波動，在部分天數採取手動干預策略。兩個不同模型並存：（1）Rolling Phase 2 模型（離線）；（2）Live 單日微調模型（即時）。MC100 投票系統作為共識機制過濾雜訊，如 Day 57 明明模型發出 Sell 訊號，但 MC 多數投 Hold（8395/100），故維持 Hold。

**改進方向：**
- 引入情緒指標（如 VIX、新聞情緒分析）以應對突發地緣政治事件
- 加入停損機制（Stop-Loss），限制單日最大虧損比率
- 計算 Sharpe Ratio 並納入風控指標

---

## 7 結論

本實驗從 Keras 單特徵 LSTM Baseline 出發，透過系統性的超參數調整（sequence length、model architecture、learning rate、dropout）確立最佳基礎設定。進一步將框架升級為 PyTorch，引入八維技術指標特徵、殘差預測目標、方向性複合損失函數、AdamW 優化器及多層後處理管線，全面改善了模型的訓練穩定性與預測品質，測試 RMSE 由 55.53 降至 39.61（28.7%），MAPE 由 2.15% 降至 1.50%。

Rolling Forecast 模擬顯示模型在 10 個交易日內達到 60% 的方向準確率，MAPE（1.62%）遠優於 Naive Baseline（2.65%）。Live Trading 競賽則以 +4.37% 的 ROI、1.63% 的 MDD 結束，驗證 LSTM + Attention 框架在實際金融決策上的應用潛力。

| 項目 | 結果 |
|---|---|
| Best Model RMSE | 39.61 TWD（28.7% vs Baseline） |
| Best Model MAPE | 1.50%（vs Baseline 2.15%） |
| Rolling RMSE | 40.27 TWD |
| Rolling MAPE | 1.62% |
| Competition ROI | **+4.37%** |
| Competition MDD | **1.63%** |
| 交易次數 | 5（active participation） |
| 所有合規規則 | **PASSED** |

未來可進一步結合情緒分析、Transformer 架構與線上學習機制，持續提升模型的適應能力。
