#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1 Experiment Runner (PyTorch)
===================================
Runs all Phase 1 experiments matching the reference report structure:
  - Baseline: Close only, LSTM[128,64], look_back=100, LR=0.001, Batch=32, Epochs=150, Dropout=0.2
  - Exp 1: Sequence Length (60, 90, 120 vs Baseline 100)
  - Exp 2: Architecture ([64,32], [256,128], [128,64,32], [256,128,64] vs Baseline [128,64])
  - Exp 3: Training Params (LR=0.0005/0.005, Batch=16/64, Epochs=100)
  - Exp 4: Dropout (0.1, 0.3 vs Baseline 0.2)

Uses PyTorch LSTM + Bahdanau Attention (same as production notebook).
Outputs: phase1_results.csv
"""
import os, sys, time, random, warnings
import numpy as np
import pandas as pd

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from copy import deepcopy

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch {torch.__version__} | Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    torch.backends.cudnn.benchmark = False

# ────────────────────────────────────────────────────────────
# Global Config
# ────────────────────────────────────────────────────────────
GLOBAL_SEED = 42
TICKER = '2330.TW'
TEST_DAYS = 10
VAL_FRAC = 0.10
GRAD_CLIP = 1.0
COMP_END_DATE = '2026-04-02'

# ────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.W = nn.Linear(hidden, hidden)
        self.u = nn.Parameter(torch.randn(hidden) * 0.05)
    def forward(self, x):
        score = torch.matmul(torch.tanh(self.W(x)), self.u)
        alpha = torch.softmax(score, dim=1).unsqueeze(-1)
        return (x * alpha).sum(dim=1)


class StockLSTM(nn.Module):
    def __init__(self, input_size, lstm_units, dropout_rate):
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.proj_layers = nn.ModuleList()
        in_sz = input_size
        for units in lstm_units:
            self.lstm_layers.append(nn.LSTM(in_sz, units, batch_first=True))
            self.norm_layers.append(nn.LayerNorm(units))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            self.proj_layers.append(nn.Linear(in_sz, units, bias=False) if in_sz != units else nn.Identity())
            in_sz = units
        self.attention = Attention(lstm_units[-1])
        self.fc = nn.Sequential(
            nn.Linear(lstm_units[-1], 64),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:    nn.init.xavier_uniform_(p.data)
            elif 'weight_hh' in name:  nn.init.orthogonal_(p.data)
            elif 'bias' in name:       nn.init.zeros_(p.data)
        nn.init.normal_(self.fc[-1].weight, std=1e-2)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        out = x
        for lstm, norm, drop, proj in zip(self.lstm_layers, self.norm_layers,
                                           self.dropout_layers, self.proj_layers):
            residual = proj(out)
            out, _ = lstm(out)
            out = norm(out)
            out = drop(out)
            out = out + residual
        context = self.attention(out)
        return self.fc(context).squeeze(-1)


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calc_mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    mask = np.abs(y_true) > 1e-6
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calc_dir_acc(y_true_ret, y_pred_ret):
    yt = np.asarray(y_true_ret); yp = np.asarray(y_pred_ret)
    mask = np.abs(yt) > 1e-6
    if mask.sum() == 0: return float('nan')
    return float(np.mean(np.sign(yt[mask]) == np.sign(yp[mask])) * 100)


# ────────────────────────────────────────────────────────────
# Data
# ────────────────────────────────────────────────────────────
print('\n📊 Loading data ...')
csv_path = '2330_stock_data.csv'
if not os.path.exists(csv_path):
    import yfinance as yf
    raw = yf.download(TICKER, period='10y', interval='1d', auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.to_csv(csv_path)

df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
for col in ['Open','High','Low','Close','Volume']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=[c for c in ['Open','High','Low','Close'] if c in df.columns]).copy()
df.index = pd.to_datetime(df.index, errors='coerce')
df = df[~df.index.isna()].copy()
if getattr(df.index, 'tz', None) is not None:
    df.index = df.index.tz_convert(None)

# Technical indicators
try:
    import pandas_ta as ta
except ModuleNotFoundError:
    import pandas_ta_classic as ta

df.ta.sma(length=5, append=True)
df.ta.sma(length=20, append=True)
df.ta.rsi(length=14, append=True)
df.ta.macd(fast=12, slow=26, signal=9, append=True)
df.ta.bbands(length=20, std=2.0, append=True)
df.ta.atr(length=14, append=True)
df.ta.stoch(k=14, d=3, smooth_k=3, append=True)

macdh_col  = next((c for c in df.columns if c.startswith('MACDh')), None)
bbp_col    = next((c for c in df.columns if c.startswith('BBP')), None)
atr_col    = next((c for c in df.columns if c.startswith('ATRr') or c.startswith('ATR_')), None)
stochk_col = next((c for c in df.columns if c.startswith('STOCHk')), None)

FEATURE_SETS = {
    'CloseOnly': ['Close'],
    'TechCore': [c for c in ['Close', 'SMA_5', 'SMA_20', 'RSI_14'] if c in df.columns],
    'TechEnhanced': [c for c in ['Close', 'SMA_5', 'SMA_20', 'RSI_14',
                                   macdh_col, bbp_col, atr_col, stochk_col]
                      if c is not None and c in df.columns],
}

# Align NaN from TA indicators
all_needed = sorted(set(sum(FEATURE_SETS.values(), [])))
df.dropna(subset=all_needed, inplace=True)

close_raw = df['Close'].values.astype(np.float64)
N = len(df)

# Test window
comp_end = pd.Timestamp(COMP_END_DATE).date()
matches = [i for i, d in enumerate(df.index) if d.date() == comp_end]
test_end_idx = matches[0] if matches else N - 1
test_start_idx = test_end_idx - TEST_DAYS + 1

print(f'  Rows: {N}')
print(f'  Test: {df.index[test_start_idx].date()} ~ {df.index[test_end_idx].date()}')
for name, cols in FEATURE_SETS.items():
    print(f'  {name}: {cols}')


def create_dataset(scaled, close, lb):
    X, y, tidx = [], [], []
    for i in range(len(scaled) - lb):
        X.append(scaled[i:i+lb])
        log_ret = np.log(close[i+lb] / (close[i+lb-1] + 1e-10))
        y.append(log_ret)
        tidx.append(i + lb)
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(tidx, dtype=np.int32))


# ────────────────────────────────────────────────────────────
# Experiment Runner
# ────────────────────────────────────────────────────────────
def run_experiment(name, feature_set='CloseOnly', look_back=100,
                   lstm_units=None, lr=0.001, batch_size=32,
                   epochs=150, dropout=0.2, patience=25, seed=None):
    if lstm_units is None:
        lstm_units = [128, 64]
    if seed is None:
        seed = GLOBAL_SEED

    feat_cols = FEATURE_SETS[feature_set]
    raw = df[feat_cols].values.astype(np.float64)

    sc = MinMaxScaler((0, 1))
    sc.fit(raw[:test_start_idx])
    scaled = sc.transform(raw)

    X_all, y_all, t_all = create_dataset(scaled, close_raw, look_back)
    tr_m = t_all < test_start_idx
    te_m = (t_all >= test_start_idx) & (t_all <= test_end_idx)

    X_tv, y_tv = X_all[tr_m], y_all[tr_m]
    X_te, y_te = X_all[te_m], y_all[te_m]

    if len(X_te) == 0:
        print(f'  ⚠️ {name}: no test samples with lb={look_back}, skipping')
        return None

    n_tv = len(X_tv)
    n_val = max(int(n_tv * VAL_FRAC), 20)
    n_tr = n_tv - n_val

    X_tr, y_tr = X_tv[:n_tr], y_tv[:n_tr]
    X_va, y_va = X_tv[n_tr:], y_tv[n_tr:]

    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_tr).to(DEVICE)
    X_va_t = torch.FloatTensor(X_va).to(DEVICE)
    y_va_t = torch.FloatTensor(y_va).to(DEVICE)
    X_te_t = torch.FloatTensor(X_te).to(DEVICE)

    tr_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=batch_size, shuffle=False)

    set_seed(seed)
    model = StockLSTM(len(feat_cols), lstm_units, dropout).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    best_val = float('inf')
    best_state = None
    bad = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_samples = 0
        for xb, yb in tr_loader:
            optimizer.zero_grad(set_to_none=True)
            pb = model(xb)
            loss = criterion(pb, yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item() * len(xb)
            n_samples += len(xb)
        train_loss /= max(n_samples, 1)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_val_s = 0
            for xb, yb in va_loader:
                vl = criterion(model(xb), yb).item() * len(xb)
                val_loss += vl
                n_val_s += len(xb)
            val_loss /= max(n_val_s, 1)

        scheduler.step(val_loss)

        if np.isfinite(val_loss) and val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    elapsed = time.time() - t0

    if best_state is None:
        print(f'  ⚠️ {name}: training failed (no valid checkpoint)')
        return None

    model.load_state_dict(best_state)
    model.eval()

    # Train metrics
    with torch.no_grad():
        pred_tr_lr = model(X_tr_t).cpu().numpy().flatten()
    idx_tr = t_all[tr_m][:n_tr]
    prev_tr = close_raw[idx_tr - 1]
    next_tr = close_raw[idx_tr]
    pred_px_tr = prev_tr * np.exp(pred_tr_lr)
    train_rmse = float(np.sqrt(mean_squared_error(next_tr, pred_px_tr)))
    train_mape = calc_mape(next_tr, pred_px_tr)

    # Test metrics
    with torch.no_grad():
        pred_lr = model(X_te_t).cpu().numpy().flatten()
    idx_te = t_all[te_m]
    prev_te = close_raw[idx_te - 1]
    next_te = close_raw[idx_te]
    pred_px = prev_te * np.exp(pred_lr)

    test_rmse = float(np.sqrt(mean_squared_error(next_te, pred_px)))
    test_mae = float(mean_absolute_error(next_te, pred_px))
    test_mape = calc_mape(next_te, pred_px)
    dir_acc = calc_dir_acc(y_te.flatten(), pred_lr)

    result = {
        'Experiment': name,
        'FeatureSet': feature_set,
        'LookBack': look_back,
        'Architecture': str(lstm_units),
        'LR': lr,
        'BatchSize': batch_size,
        'Epochs': epochs,
        'ActualEpochs': ep,
        'Dropout': dropout,
        'Train_RMSE': round(train_rmse, 4),
        'Train_MAPE%': round(train_mape, 2),
        'Test_RMSE': round(test_rmse, 4),
        'Test_MAE': round(test_mae, 4),
        'Test_MAPE%': round(test_mape, 2),
        'DirAcc%': round(dir_acc, 1),
        'Time_s': round(elapsed, 1),
    }

    print(f'  ✅ {name}: Train RMSE={train_rmse:.2f} MAPE={train_mape:.2f}% | '
          f'Test RMSE={test_rmse:.2f} MAPE={test_mape:.2f}% DirAcc={dir_acc:.1f}% ({elapsed:.1f}s)')
    return result


# ════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS
# ════════════════════════════════════════════════════════════
all_results = []

# ── Baseline ──────────────────────────────────────────────
print('\n' + '='*80)
print('📋 Baseline: Close, lb=100, [128,64], LR=0.001, BS=32, Epochs=150, Drop=0.2')
print('='*80)
r = run_experiment('Baseline', feature_set='CloseOnly', look_back=100,
                   lstm_units=[128,64], lr=0.001, batch_size=32,
                   epochs=150, dropout=0.2)
if r: all_results.append(r)

# ── Exp 1: Sequence Length ────────────────────────────────
print('\n' + '='*80)
print('🧪 Experiment 1: Sequence Length (60 / 90 / 120)')
print('='*80)
for lb in [60, 90, 120]:
    r = run_experiment(f'Exp1-LB{lb}', feature_set='CloseOnly', look_back=lb,
                       lstm_units=[128,64], lr=0.001, batch_size=32,
                       epochs=150, dropout=0.2)
    if r: all_results.append(r)

# ── Exp 2: Architecture ──────────────────────────────────
print('\n' + '='*80)
print('🧪 Experiment 2: Architecture ([64,32],[256,128],[128,64,32],[256,128,64])')
print('='*80)
for arch_name, units in [('64_32', [64,32]), ('256_128', [256,128]),
                          ('128_64_32', [128,64,32]), ('256_128_64', [256,128,64])]:
    r = run_experiment(f'Exp2-{arch_name}', feature_set='CloseOnly', look_back=100,
                       lstm_units=units, lr=0.001, batch_size=32,
                       epochs=150, dropout=0.2)
    if r: all_results.append(r)

# ── Exp 3: Training Params ───────────────────────────────
print('\n' + '='*80)
print('🧪 Experiment 3: Training Params (LR, Batch, Epochs)')
print('='*80)
training_cfgs = [
    ('LR0.0005', {'lr': 0.0005}),
    ('LR0.005',  {'lr': 0.005}),
    ('BS16',     {'batch_size': 16}),
    ('BS64',     {'batch_size': 64}),
    ('Ep100',    {'epochs': 100}),
]
for tag, overrides in training_cfgs:
    kw = dict(feature_set='CloseOnly', look_back=100, lstm_units=[128,64],
              lr=0.001, batch_size=32, epochs=150, dropout=0.2)
    kw.update(overrides)
    r = run_experiment(f'Exp3-{tag}', **kw)
    if r: all_results.append(r)

# ── Exp 4: Dropout ────────────────────────────────────────
print('\n' + '='*80)
print('🧪 Experiment 4: Dropout (0.1 / 0.3)')
print('='*80)
for drop in [0.1, 0.3]:
    r = run_experiment(f'Exp4-Drop{drop}', feature_set='CloseOnly', look_back=100,
                       lstm_units=[128,64], lr=0.001, batch_size=32,
                       epochs=150, dropout=drop)
    if r: all_results.append(r)

# ── Feature Selection (bonus) ─────────────────────────────
print('\n' + '='*80)
print('🧪 Feature Selection: TechCore / TechEnhanced')
print('='*80)
for fs in ['TechCore', 'TechEnhanced']:
    r = run_experiment(f'Feat-{fs}', feature_set=fs, look_back=100,
                       lstm_units=[128,64], lr=0.001, batch_size=32,
                       epochs=150, dropout=0.2)
    if r: all_results.append(r)

# ════════════════════════════════════════════════════════════
# Save Results
# ════════════════════════════════════════════════════════════
results_df = pd.DataFrame(all_results)
results_df.to_csv('phase1_results.csv', index=False)

print('\n' + '='*80)
print('📊 Phase 1 Results Summary')
print('='*80)
display_cols = ['Experiment', 'FeatureSet', 'LookBack', 'Architecture', 'LR', 'BatchSize',
                'Dropout', 'Train_RMSE', 'Train_MAPE%', 'Test_RMSE', 'Test_MAPE%', 'DirAcc%']
print(results_df[display_cols].to_string(index=False))
print(f'\n✅ Results saved to phase1_results.csv ({len(all_results)} experiments)')
