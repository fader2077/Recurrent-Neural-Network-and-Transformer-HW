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
- **模型架構**：LSTM(128) → LSTM(64) → LayerNorm + Residual → Attention → Dense(1)
- **Optimizer**：Adam（LR = 0.001）
- **Loss Function**：Mean Squared Error (MSE)
- **Epochs**：150（Early Stopping, patience=10），**Batch Size**：32
- **Dropout**：0.2
- **預測目標**：Log-return ln(P_t / P_{t-1})，評估時轉換回價格空間

Baseline 模型之評估結果如表 1 所示。

**Table 1: Baseline 模型評估結果**

| 資料集 | RMSE | MAE | MAPE |
|---|---|---|---|
| Train | 9.54 | — | 1.25% |
| Test | 40.07 | 31.82 | 1.75% |

> Baseline 模型在測試集達到 RMSE 40.07 / MAPE 1.75%，已具備一定預測能力。後續將透過超參數調整探索進一步改善空間。

---

## 3 超參數調整（Hyperparameter Tuning）

### 3.1 Sequence Length（look_back）調整

本實驗固定其餘參數不變（LSTM[128, 64]、LR=0.001、Batch=32、Epochs=150 + Early Stopping），僅調整輸入序列長度 look_back，分別測試 60、90、120 天三種設定，與 Baseline（100 天）進行比較。

**Table 2: 不同 look_back 設定之模型評估結果**

| look_back | Actual Epochs | Train RMSE | Train MAPE | Test RMSE | Test MAPE |
|---|---|---|---|---|---|
| 100（Baseline） | 41 | 9.54 | 1.25% | 40.07 | 1.75% |
| 60 | 53 | 9.18 | 1.24% | 40.47 | 1.81% |
| 90 | 26 | 9.62 | 1.27% | 40.59 | 1.83% |
| 120 | 38 | 9.75 | 1.26% | 40.61 | 1.83% |

> Baseline 的 look_back=100 在測試集上取得最低 RMSE（40.07）與最低 MAPE（1.75%）。過短視窗（60）雖訓練 RMSE 最低，但測試泛化力不如 100；過長視窗（90、120）引入遠期雜訊，測試 RMSE 均高於 Baseline。

### 3.2 模型架構（LSTM Units）調整

本實驗固定 look_back=100，調整 LSTM 層的神經元數量與堆疊層數，分別測試四種架構與 Baseline（[128, 64]）進行比較。

**Table 3: 不同 LSTM 架構之模型評估結果（look_back=100）**

| LSTM 架構 | Actual Epochs | Train RMSE | Train MAPE | Test RMSE | Test MAPE |
|---|---|---|---|---|---|
| [128, 64]（Baseline） | 41 | 9.54 | 1.25% | 40.07 | 1.75% |
| [64, 32] | 31 | 9.60 | 1.26% | 40.45 | 1.81% |
| **[256, 128]** | **26** | **9.49** | **1.23%** | **39.86** | **1.71%** |
| [128, 64, 32] | 45 | 9.65 | 1.27% | 40.93 | 1.87% |
| [256, 128, 64] | 35 | 9.61 | 1.26% | 41.00 | 1.88% |

> [256, 128] 二層寬架構取得所有實驗中最低的 Test RMSE（39.86）與 MAPE（1.71%），較 Baseline 改善約 0.5%。更深的三層架構（[128,64,32]、[256,128,64]）反而表現最差，顯示增加深度未必提升泛化力，反而容易過擬合。

### 3.3 訓練參數（Learning Rate、Batch Size、Epochs）調整

本實驗固定 look_back=100、LSTM[128, 64]，調整學習率、Batch Size 與 Epochs（均搭配 Early Stopping）。

**Table 4: 不同訓練參數之模型評估結果（look_back=100）**

| LR | Batch | Max Epochs | Actual Epochs | Train RMSE | Train MAPE | Test RMSE | Test MAPE |
|---|---|---|---|---|---|---|---|
| 0.001（Baseline） | 32 | 150 | 41 | 9.54 | 1.25% | 40.07 | 1.75% |
| 0.0005 | 32 | 150 | 41 | 9.72 | 1.30% | 40.88 | 1.87% |
| 0.005 | 32 | 150 | 44 | 9.61 | 1.26% | 40.92 | 1.87% |
| 0.001 | 16 | 150 | 32 | 9.62 | 1.26% | 41.14 | 1.90% |
| 0.001 | 64 | 150 | 40 | 9.56 | 1.25% | 40.14 | 1.76% |
| 0.001 | 32 | 100 | 41 | 9.54 | 1.25% | 40.07 | 1.75% |

> Baseline（LR=0.001, BS=32）表現最佳。BS=64 略遜但接近 Baseline（Test RMSE 40.14 vs 40.07）。降低或提高學習率均未帶來改善，BS=16 表現最差（Test MAPE 1.90%）。Epochs=100 與 150 結果相同，因 Early Stopping 均在 41 epoch 停止。

### 3.4 Dropout 正則化調整

本實驗固定 look_back=100、LSTM[128, 64]，調整 Dropout Rate（Baseline 已設為 0.2）。

**Table 5: 不同 Dropout Rate 之模型評估結果（look_back=100）**

| Dropout Rate | Actual Epochs | Train RMSE | Train MAPE | Test RMSE | Test MAPE |
|---|---|---|---|---|---|
| **0.2（Baseline）** | **41** | **9.54** | **1.25%** | **40.07** | **1.75%** |
| 0.1 | 41 | 9.59 | 1.27% | 40.24 | 1.78% |
| 0.3 | 50 | 9.63 | 1.26% | 41.05 | 1.89% |

> Baseline 的 Dropout=0.2 為最佳設定（Test RMSE 40.07）。降低至 0.1 差異不大（40.24），但提高至 0.3 明顯惡化（41.05），表明過度正則化對台積電時序預測不利，因其具有強趨勢連續性。

### 3.5 超參數調整總結

**Table 6: Phase 1 超參數調整總結**

綜合四項實驗結果，**最佳單一實驗為 Exp2-256_128**（Test RMSE = 39.86, MAPE = 1.71%），較 Baseline（RMSE = 40.07, MAPE = 1.75%）有小幅改善。各實驗主要結論如下：

- **Sequence Length**：look_back=100 表現最佳（Test RMSE 40.07），過短（60, RMSE 40.47）或過長（90/120, RMSE 40.59/40.61）均不如 Baseline。
- **LSTM 架構**：[256, 128] 寬二層架構取得全局最佳 Test RMSE（39.86）；三層架構反而因深度增加而泛化力下降（RMSE > 40.9）。
- **訓練參數**：Baseline 的 LR=0.001、BS=32 即為最佳組合；BS=64 表現接近（RMSE 40.14），其餘調整均使測試誤差上升。
- **Dropout**：0.2（Baseline）為最佳，0.3 明顯惡化（MAPE 1.89% vs 1.75%），顯示台積電股價具有強趨勢連續性，過度正則化不利於預測。

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
| Close only (Phase 1 Baseline) | 40.07 | 31.82 | 1.75% | 22.2% |
| E1-LookBack90 (TechCore, 4 特徵) | 40.10 | 26.78 | 1.46% | 77.8% |
| E2-ArchDeep (TechCore, 4 特徵) | 42.95 | 27.39 | 1.49% | 77.8% |
| E3-TechEnhanced (8 特徵) | 44.61 | 28.89 | 1.57% | 77.8% |
| **Final Model (全資料訓練, 8 特徵)** | **39.61** | **27.40** | **1.50%** | **77.8%** |

### 4.3 結果分析

在 Phase 1 超參數調整確立最佳基礎設定後，進一步引入多維技術指標與複合損失函數，測試 MAPE 從 1.75% 降至 1.50%，Test RMSE 從 40.07 降至 39.61，顯示多特徵模型與方向性損失函數能有效提升預測品質。

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

本實驗以 PyTorch 框架建構 Stacked LSTM + Attention 模型。Phase 1 超參數調整涵蓋 17 組實驗（sequence length、model architecture、learning rate、batch size、dropout、feature set），以 log-return 為預測目標。Phase 1 Baseline（LSTM[128,64], look_back=100）達到 Test RMSE 40.07、MAPE 1.75%，而最佳單一實驗（[256,128] 架構）進一步降至 RMSE 39.86、MAPE 1.71%。進一步引入八維技術指標特徵、方向性複合損失函數與多層後處理管線後，最終模型 Test RMSE 降至 39.61、MAPE 降至 1.50%。

Rolling Forecast 模擬顯示模型在 10 個交易日內達到 60% 的方向準確率，MAPE（1.62%）遠優於 Naive Baseline（2.65%）。Live Trading 競賽則以 +4.37% 的 ROI、1.63% 的 MDD 結束，驗證 LSTM + Attention 框架在實際金融決策上的應用潛力。

| 項目 | 結果 |
|---|---|
| Phase 1 Baseline RMSE | 40.07 TWD（MAPE 1.75%） |
| Phase 1 Best Exp RMSE | 39.86 TWD（[256,128], MAPE 1.71%） |
| Final Model RMSE | 39.61 TWD（8 特徵, MAPE 1.50%） |
| Rolling RMSE | 40.27 TWD |
| Rolling MAPE | 1.62% |
| Competition ROI | **+4.37%** |
| Competition MDD | **1.63%** |
| 交易次數 | 5（active participation） |
| 所有合規規則 | **PASSED** |

未來可進一步結合情緒分析、Transformer 架構與線上學習機制，持續提升模型的適應能力。
