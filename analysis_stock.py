# -*- coding: utf-8 -*-
"""
AI Market Intelligence Pro+  ➜  Speed improved & bug fixed version
Uses Prophet built-in CV + better data caching + forecast caching
streamlit run analysis_stock.py
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
import json, hashlib, os
from datetime import timedelta
import logging

import pickle
from pathlib import Path

DATA_CACHE_FILE = Path("stock_data_cache.pkl")  # or .parquet for better performance

def load_cached_data():
    if DATA_CACHE_FILE.exists():
        with open(DATA_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cached_data(data_dict):
    with open(DATA_CACHE_FILE, 'wb') as f:
        pickle.dump(data_dict, f)


logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.ERROR)
st.set_page_config(page_title="AI Market Intelligence Pro+", layout="wide")

USER_FILE = "users.json"

# ── Auth helpers ─────────────────────────────────────────────────────────────
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE) as f:
            return json.load(f)
    default_users = {"banana": hash_password("140484")}
    save_users(default_users)
    return default_users

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

# ── Session init ─────────────────────────────────────────────────────────────
def init_auth():
    for k in ("authenticated", "username", "is_admin", "show_user_management"):
        if k not in st.session_state:
            st.session_state[k] = False if k != "username" else None

@st.cache_data(ttl=3600*24, show_spinner=False)
def fetch_market_only(market_tickers, period="5y"):
    """Download only market indices; never crash."""
    try:
        raw = yf.download(list(market_tickers), period=period, auto_adjust=True, progress=False)
        if raw.empty:
            return {}
        # Multi-index → dict
        return {t: raw.xs(t, axis=1, level=1).ffill().bfill()
                for t in market_tickers if t in raw.columns.get_level_values(1)}
    except Exception as e:
        st.warning(f"Market download failed: {e}")
        return {}

@st.cache_data(ttl=3600*24*7, show_spinner=False)
def fetch_single_ticker(ticker, period="5y"):
    """Never crash; return empty DF if Yahoo blocks us."""
    for attempt in range(3):
        try:
            data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if not data.empty:
                return data
        except Exception as e:
            if attempt == 2:          # last attempt
                st.warning(f"Yahoo block for {ticker}: {e}")
        time.sleep(2)                 # polite pause
    return pd.DataFrame()    # Return empty if all attempts fail

# ── Fast data caching & pre-alignment ────────────────────────────────────────
@st.cache_data(ttl=3600*24, show_spinner=False)
def prepare_aligned_data(stocks, market_tickers):
    # Streamlit Cloud doesn't persist files well, so we rely on st.cache_data
    # and download only what is missing from the current session's cache.
    all_tickers = list(set(stocks + market_tickers))
    
    # Download all at once - yfinance is faster with a list
    try:
        raw_data = yf.download(all_tickers, period="5y", auto_adjust=True, progress=False)
    except Exception as e:
        st.error(f"Global download failure: {e}")
        return {}    
    # Handle the MultiIndex columns if multiple tickers were downloaded
    aligned_dict = {}
    for ticker in all_tickers:
        try:
            # Extract data for this specific ticker from the bulk result
            if len(all_tickers) > 1:
                df = raw_data.xs(ticker, axis=1, level=1).dropna(how='all')
            else:
                df = raw_data.dropna(how='all')
            
            if not df.empty:
                # Clean up the data
                df = df.ffill().bfill()
                aligned_dict[ticker] = df
        except KeyError:
            continue
            
    return aligned_dict

# ── Fast CV using Prophet built-in ───────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_fast_cv_errors(_model, df, horizons=[1, 5, 10]):
    df_cv = cross_validation(
        _model,
        initial='730 days',
        period='180 days',
        horizon='10 days'
    )
    df_p = performance_metrics(df_cv)

    errors = {}
    for h in horizons:
        td = pd.Timedelta(days=h)
        row = df_p[df_p['horizon'] == td]
        errors[h] = row['mape'].iloc[0] if not row.empty else np.nan

    return errors

@st.cache_data(ttl=7200, show_spinner=False, hash_funcs={pd.DataFrame: lambda df: df.index[-1] if not df.empty else 0})
def forecast_multivariate_cached(
    symbol: str,
    market_tickers_tuple: tuple,
    forecast_days: int,
    cps: float,
    sps: float
):
    market_tickers = list(market_tickers_tuple)
    
    if symbol not in df_dict:
        raise KeyError(f"Symbol {symbol!r} not found in df_dict. Available: {list(df_dict.keys())}")
    
    df = df_dict[symbol]

    # ── Outlier clipping ─────────────────────────────────────────────────────
    close_values = df["Close"].to_numpy()          # 1D numpy array
    if len(close_values) == 0:
        raise ValueError(f"No valid Close data for {symbol}")

    mean_val = close_values.mean()
    std_val = close_values.std(ddof=0)

    y_clipped = np.clip(close_values,
                        mean_val - 3 * std_val,
                        mean_val + 3 * std_val)

    stock_df = pd.DataFrame({
        "ds": df.index,
        "y": y_clipped,
    }).dropna().reset_index(drop=True)

    stock_df["ds"] = pd.to_datetime(stock_df["ds"]).dt.tz_localize(None)

    # ── Add regressors to training data ──────────────────────────────────────
    for mkt in market_tickers:
        reg_name = mkt.replace("^", "")
        stock_df[reg_name] = df_dict[mkt]["Close"].reindex(stock_df["ds"]).ffill().bfill().values

    # ── Prophet model with logistic growth (realistic cap) ───────────────────
    model = Prophet(
        growth='logistic',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=cps * 0.3,          # lower = smoother, less overfit
        seasonality_prior_scale=sps,
        interval_width=0.85,                         # slightly wider confidence
        holidays_prior_scale=10.0                    # stronger holiday effect if needed
    )
    for reg in [m.replace("^", "") for m in market_tickers]:
        model.add_regressor(reg)

    # Cap/floor: 50% above historical max, 50% below min (prevents extreme forecasts)
    df_cap = df["Close"].max() * 1.3               # max +30%
    df_floor = df["Close"].min() * 0.7             # min -30%
    stock_df['cap'] = df_cap
    stock_df['floor'] = df_floor
    stock_df['y'] = stock_df['y'] * 0.97           # slight decay for recent prices

    model.fit(stock_df)

    # ── Future frame ─────────────────────────────────────────────────────────
    future_dates = model.make_future_dataframe(periods=forecast_days, freq="B")
    future_dates['cap'] = df_cap
    future_dates['floor'] = df_floor

    # ── Conservative regressor extrapolation (constant last value) ───────────
    for mkt in market_tickers:
        reg_name = mkt.replace("^", "")
        hist = df_dict[mkt]["Close"].dropna()
        last_val = hist.iloc[-1] if not hist.empty else 0.0
        future_dates[reg_name] = last_val  # No compounding drift

    fcst = model.predict(future_dates)
    # ── Ensemble: Add ETS + ARIMA for more robust & realistic forecast ──────
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    
    # Use the training y values (clipped)
    y_train = stock_df['y'].values
    
    # 1. Prophet forecast (already have)
    prophet_fcst = fcst['yhat'].values[-forecast_days:]
    
    # 2. Exponential Smoothing (ETS) - stable, mean-reverting

    try:
            ets_model = ExponentialSmoothing(
                y_train,
                trend='add',
                seasonal='add',
                seasonal_periods=5  # weekly-ish pattern
            ).fit()
            ets_fcst = ets_model.forecast(steps=forecast_days)
    except Exception as e:
            print(f"ETS failed: {e}")
            ets_fcst = prophet_fcst  # fallback
    # 3. ARIMA (short-term momentum)
    try:
        arima_model = ARIMA(y_train, order=(1,1,1)).fit()
        arima_fcst = arima_model.forecast(steps=forecast_days)
    except:
        arima_fcst = prophet_fcst  # fallback

    ensemble_fcst = np.median([prophet_fcst, ets_fcst, arima_fcst], axis=0)
    # ── Predict ──────────────────────────────────────────────────────────────
    fcst = model.predict(future_dates)
    #future_horizon = fcst.tail(forecast_days)

    future_horizon = fcst.tail(forecast_days).copy()
    future_horizon['yhat'] = ensemble_fcst

    future_horizon['yhat_lower'] = future_horizon['yhat'] - 1.5 * (future_horizon['yhat_upper'] - future_horizon['yhat'])
    future_horizon['yhat_upper'] = future_horizon['yhat'] + 1.5 * (future_horizon['yhat_upper'] - future_horizon['yhat'])

    errors = get_fast_cv_errors(model, stock_df, horizons=[1,5,10])

    return future_horizon, errors

# ── Technical indicators ─────────────────────────────────────────────────────
def clip_outliers(y, thresh=3):
    mu, sig = y.mean(), y.std()
    return np.clip(y, mu - thresh * sig, mu + thresh * sig)

def calculate_rsi(series, windows=None):
    if windows is None:
        windows = [3, 5, 7, 9, 14, 21, 30]
    
    res = {}
    
    # Force to numpy array right at the beginning
    close = np.asarray(series, dtype=float)
    
    if len(close) < 2:
        for w in windows:
            res[w] = np.nan
        return res
    
    delta = np.diff(close)
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    for w in windows:
        if len(delta) < w:
            res[w] = np.nan
            continue
        
        # Explicitly ensure inputs are 1D numpy arrays
        kernel = np.ones(w) / w
        
        try:
            avg_gain = np.convolve(gain, kernel, mode='valid')[-1]
            avg_loss = np.convolve(loss, kernel, mode='valid')[-1]
        except ValueError as e:
            # If any dimension issue, fallback to nan
            print(f"Convolve error for window {w}: {e}")
            res[w] = np.nan
            continue
        
        if np.isnan(avg_gain) or np.isnan(avg_loss):
            res[w] = np.nan
            continue
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        res[w] = float(rsi)
    
    return res

def get_rsi_label(val):
    if val >= 80: return "!! PARABOLIC !!"
    if val >= 70: return "OVERBOUGHT"
    if val <= 20: return "!! EXTREME OVERSOLD !!"
    if val <= 30: return "OVERSOLD"
    return "NEUTRAL"

def get_exit_strategy(price, series):
    # 只取最近 30 筆，轉成 numpy array 避免 pandas 陷阱
    recent_close = series.tail(30).to_numpy(dtype=float)
    
    if len(recent_close) == 0:
        std_value = 0.0
    else:
        # 使用 numpy std，保證是 scalar
        std_value = np.nanstd(recent_close)
        
        # 如果是 nan 或無效，設為 0
        if np.isnan(std_value):
            std_value = 0.0
    
    # 計算 tp / sl（現在 std_value 永遠是 float）
    tp = price + 2.5 * std_value
    sl = price - 1.5 * std_value
    
    return tp, sl

def calculate_macd(s_close, fast=12, slow=26, signal=9):
    # Force to numpy array to avoid any pandas residue
    close = np.asarray(s_close, dtype=float).flatten()
    
    if len(close) < slow:
        return np.nan, np.nan, np.nan  # not enough data
    
    # EMA using numpy (more reliable than pandas ewm in edge cases)
    def ema(arr, span):
        alpha = 2 / (span + 1)
        ema_values = np.zeros_like(arr)
        ema_values[0] = arr[0]
        for i in range(1, len(arr)):
            ema_values[i] = alpha * arr[i] + (1 - alpha) * ema_values[i-1]
        return ema_values
    
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    signal_line = ema(macd_line, signal)
    
    histogram = macd_line - signal_line
    
    # Take last values, force to float scalar
    macd_last = float(macd_line[-1]) if len(macd_line) > 0 else np.nan
    signal_last = float(signal_line[-1]) if len(signal_line) > 0 else np.nan
    hist_last = float(histogram[-1]) if len(histogram) > 0 else np.nan
    
    return macd_last, signal_last, hist_last

def calculate_cmf(df, window=20):
    # Force ALL columns to clean 1D numpy arrays
    high = np.asarray(df['High'], dtype=float).flatten()
    low = np.asarray(df['Low'], dtype=float).flatten()
    close = np.asarray(df['Close'], dtype=float).flatten()
    volume = np.asarray(df['Volume'], dtype=float).flatten()
    
    # Sync lengths (safety for misaligned data)
    min_len = min(len(high), len(low), len(close), len(volume))
    if min_len < window:
        return 0.0
    
    high = high[:min_len]
    low = low[:min_len]
    close = close[:min_len]
    volume = volume[:min_len]
    
    # Money Flow Multiplier (scalar array)
    mfm = ((close - low) - (high - close)) / (high - low + 1e-9)
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # Rolling sums using np.convolve (both inputs now guaranteed 1D numpy)
    window_kernel = np.ones(window)
    
    sum_mfv = np.convolve(mfv, window_kernel, mode='valid')
    sum_vol = np.convolve(volume, window_kernel, mode='valid')
    
    # CMF
    cmf_array = sum_mfv / (sum_vol + 1e-9)
    
    # Take last value, force scalar
    if len(cmf_array) > 0:
        cmf_val = float(cmf_array[-1])
    else:
        cmf_val = 0.0
    
    # Final nan safety
    if np.isnan(cmf_val):
        cmf_val = 0.0
    
    return cmf_val

def calculate_kd(s_high, s_low, s_close, n=9, m=3):
    high = np.asarray(s_high, dtype=float).flatten()
    low = np.asarray(s_low, dtype=float).flatten()
    close = np.asarray(s_close, dtype=float).flatten()
    
    if len(high) < n:
        return np.nan, np.nan
    
    # rolling min/max 使用 stride tricks
    shape = (len(high) - n + 1, n)
    strides = (high.strides[0], high.strides[0])
    
    low_n = np.lib.stride_tricks.as_strided(low, shape=shape, strides=strides).min(axis=1)
    high_n = np.lib.stride_tricks.as_strided(high, shape=shape, strides=strides).max(axis=1)
    
    denominator = high_n - low_n + 1e-9
    k_value = 100 * (close[n-1:] - low_n) / denominator
    
    # D 值 rolling mean
    d_value = np.convolve(k_value, np.ones(m)/m, mode='valid')[-1] if len(k_value) >= m else np.nan
    
    k_final = float(k_value[-1]) if len(k_value) > 0 and not np.isnan(k_value[-1]) else np.nan
    d_final = float(d_value) if not np.isnan(d_value) else np.nan
    
    return k_final, d_final

def calculate_vpt(s_close, s_vol):
    # 強制轉成 1D numpy array，並確保形狀正確
    close = np.asarray(s_close).flatten()
    vol = np.asarray(s_vol).flatten()
    
    # 確保長度相同且至少有 2 筆資料
    min_len = min(len(close), len(vol))
    if min_len < 2:
        return np.nan, np.nan
    
    # 裁切到相同長度
    close = close[:min_len]
    vol = vol[:min_len]
    
    # 計算 price_change
    price_diff = np.diff(close)
    price_prev = close[:-1]
    
    # 防除零
    price_change = np.divide(price_diff, price_prev, where=price_prev != 0, out=np.zeros_like(price_diff))
    
    # VPT = 累積 (volume * price_change)
    vpt = np.cumsum(vol[1:] * price_change)
    
    # 如果 vpt 空，設為 nan
    if len(vpt) == 0:
        return np.nan, np.nan
    
    # VPT 的 EMA（使用 pandas ewm 計算最後一筆）
    vpt_series = pd.Series(vpt)
    vpt_ema = vpt_series.ewm(span=20, adjust=False).mean().iloc[-1]
    
    # 強制轉成 float scalar
    vpt_final = float(vpt[-1]) if not np.isnan(vpt[-1]) else np.nan
    vpt_ema_final = float(vpt_ema) if not pd.isna(vpt_ema) else np.nan
    
    return vpt_final, vpt_ema_final

def calculate_obv(s_close, s_vol):
    # Force flatten to 1D numpy array — this fixes (N,0) shape issues
    close = np.asarray(s_close).flatten()
    vol = np.asarray(s_vol).flatten()
    
    # Ensure same length and at least 2 points
    min_len = min(len(close), len(vol))
    if min_len < 2:
        return np.nan, False
    
    close = close[:min_len]
    vol = vol[:min_len]
    
    # Compute direction safely
    price_diff = np.diff(close)
    price_prev = close[:-1]
    
    # Avoid division by zero and get sign
    direction = np.sign(price_diff / np.where(price_prev != 0, price_prev, 1))
    
    # Volume part must match length
    vol_for_cumsum = vol[1:][:len(price_diff)]
    
    # Safe cumsum
    obv = np.cumsum(direction * vol_for_cumsum)
    
    # Rising check — only if we have at least 2 points
    if len(obv) >= 2:
        obv_rising = obv[-1] > obv[-2]
    else:
        obv_rising = False
    
    # Final values — always scalar
    obv_final = float(obv[-1]) if len(obv) > 0 and not np.isnan(obv[-1]) else np.nan
    obv_rising = bool(obv_rising)
    
    return obv_final, obv_rising

def calculate_bollinger(s_close, window=20, stds=2):
    # 強制轉成 1D numpy array，避免任何 pandas 殘留
    close = np.asarray(s_close, dtype=float).flatten()
    
    if len(close) < window:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # 計算中線（SMA）
    sma = np.convolve(close, np.ones(window)/window, mode='valid')[-1]
    
    # 計算標準差
    # 取最後 window 筆計算 std
    recent = close[-window:]
    std = np.std(recent, ddof=0)
    
    upper = sma + std * stds
    lower = sma - std * stds
    
    # Bandwidth
    bandwidth = (upper - lower) / sma if sma != 0 else np.nan
    
    # %B
    percent_b = (close[-1] - lower) / (upper - lower + 1e-9) if (upper - lower) != 0 else np.nan
    
    # 強制全部轉成 float scalar
    return (
        float(sma) if not np.isnan(sma) else np.nan,
        float(upper) if not np.isnan(upper) else np.nan,
        float(lower) if not np.isnan(lower) else np.nan,
        float(bandwidth) if not np.isnan(bandwidth) else np.nan,
        float(percent_b) if not np.isnan(percent_b) else np.nan
    )

def find_support_resistance(price_series, lookback=365, min_distance=20, tolerance_pct=2.0):
    prices = price_series.tail(lookback).to_numpy(dtype=float)
    if len(prices) < 50:
        return {'support': [], 'resistance': []}
    
    levels = []
    for i in range(1, len(prices)-1):
        price = float(prices[i])  # 強制轉 scalar
        left = prices[max(0, i-min_distance):i]
        right = prices[i+1:min(len(prices), i+min_distance+1)]
        
        if len(left) == 0 or len(right) == 0:
            continue
        
        left_min = np.nanmin(left)
        right_min = np.nanmin(right)
        left_max = np.nanmax(left)
        right_max = np.nanmax(right)
        
        if np.isnan(left_min) or np.isnan(right_min):
            continue
        
        if (price <= left_min * (1 + tolerance_pct/100)) and (price <= right_min * (1 + tolerance_pct/100)):
            levels.append(('support', price))
        
        if (price >= left_max * (1 - tolerance_pct/100)) and (price >= right_max * (1 - tolerance_pct/100)):
            levels.append(('resistance', price))
    
    # 最後處理 - 全部轉 float 再 round
    supports = []
    resistances = []
    
    for typ, p in levels:
        p_float = float(p)  # 保證是 scalar
        rounded_p = round(p_float, 2)
        
        if typ == 'support':
            supports.append(rounded_p)
        elif typ == 'resistance':
            resistances.append(rounded_p)
    
    supports = sorted(set(supports), reverse=True)[:5]
    resistances = sorted(set(resistances), reverse=True)[:5]
    
    return {'support': supports, 'resistance': resistances}

def calculate_relative_strength(stock_close, market_close):
    # Force to numpy arrays to eliminate any pandas residue
    stock = np.asarray(stock_close, dtype=float).flatten()
    market = np.asarray(market_close, dtype=float).flatten()
    
    min_len = min(len(stock), len(market))
    if min_len < 2:
        return np.nan, np.nan
    
    stock = stock[:min_len]
    market = market[:min_len]
    
    # Cumulative returns (safe from zero division)
    stock_ret = np.cumprod(1 + np.diff(stock) / stock[:-1])
    market_ret = np.cumprod(1 + np.diff(market) / market[:-1])
    
    # Relative Strength ratio
    rs = stock_ret / market_ret if np.all(market_ret != 0) else np.full_like(stock_ret, np.nan)
    
    # Final RS value (last one)
    rs_last = float(rs[-1]) if len(rs) > 0 and not np.isnan(rs[-1]) else np.nan
    
    # RS Rating
    if np.isnan(rs_last) or np.all(np.isnan(rs)):
        rs_rating = np.nan
    else:
        rs_mean = np.nanmean(rs)
        rs_rating = (rs_last / rs_mean) * 100 if rs_mean != 0 else np.nan
    
    return rs_last, float(rs_rating) if not np.isnan(rs_rating) else np.nan

def detect_follow_through(df, window=20):
    recent = df.tail(window).copy()
    if len(recent) < 15:
        return {"status": "None", "msg": "Insufficient data for FTD analysis"}
    
    # 使用 numpy 計算最低價位置，避免 index 問題
    close_prices = recent['Close'].to_numpy(dtype=float)
    volume = recent['Volume'].to_numpy(dtype=float)
    
    if len(close_prices) == 0:
        return {"status": "None", "msg": "No price data"}
    
    lowest_pos = np.argmin(close_prices)
    
    if lowest_pos > len(recent) - 4:
        return {"status": "Waiting", "msg": "Rally attempt in progress (too early for FTD)"}
    
    search_start = lowest_pos + 3
    search_end = min(lowest_pos + 10, len(recent))
    
    if search_start >= search_end:
        return {"status": "None", "msg": "No valid search range for FTD"}
    
    for i in range(search_start, search_end):
        current_close = float(close_prices[i])
        prev_close = float(close_prices[i-1])
        
        # 強制轉成 float 計算，避免 numpy 類型殘留
        price_gain = (current_close / prev_close) - 1 if prev_close != 0 else 0.0
        
        vol_increase = volume[i] > volume[i-1]
        
        if price_gain >= 0.015 and vol_increase:
            # undercut 檢查也強制 scalar
            prior_low = float(np.min(close_prices[lowest_pos:i]))
            current_low = float(close_prices[lowest_pos])
            
            if prior_low >= current_low:
                date = recent.index[i]
                return {
                    "status": "VALID",
                    "date": date.date(),
                    "gain": f"{float(price_gain)*100:.2f}%",  # 這裡明確轉 float
                    "msg": f"Confirmed on {date.date()}"
                }
    
    return {"status": "None", "msg": "No valid FTD detected in this window."}

# ── Fast tuning ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def tune(symbol, market_tickers_tuple):
    candidates = [
        {"cps": 0.001, "sps": 10},
        {"cps": 0.005, "sps": 10},
        {"cps": 0.01, "sps": 5},
    ]
    best, best_mape = None, 1e9
    total = len(candidates)
    prog = st.progress(0)

    for i, hp in enumerate(candidates):
        _, errs = forecast_multivariate_cached(
            symbol,
            market_tickers_tuple,
            30,
            hp["cps"],
            hp["sps"]
        )
        m = errs.get(5, 1e9)
        if m < best_mape:
            best, best_mape = hp, m
        prog.progress((i + 1) / total)
    prog.empty()
    return best

# ── Login page (unchanged) ───────────────────────────────────────────────────
def login_page():
    st.title("🔐 Login to AI Market Intelligence Pro+")
    tab1, tab2 = st.tabs(["Login", "Admin: Manage Users"])
    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            users = load_users()
            if u in users and users[u] == hash_password(p):
                st.session_state.authenticated = True
                st.session_state.username = u
                st.session_state.is_admin = u == "banana"
                st.session_state.show_user_management = False
                st.success(f"Welcome {u}")
                st.rerun()
            else:
                st.error("Invalid credentials")
    with tab2:
        if st.session_state.get("is_admin", False):
            st.success("Admin access")
            new_u = st.text_input("New username")
            new_p = st.text_input("New password", type="password")
            if st.button("Add User"):
                if new_u and new_p:
                    users = load_users()
                    if new_u in users:
                        st.warning("User exists")
                    else:
                        users[new_u] = hash_password(new_p)
                        save_users(users)
                        st.success("Added")
                        st.rerun()
            st.write("### Current users")
            users = load_users()
            for usr in list(users):
                if usr != "banana":
                    c1, c2 = st.columns([3, 1])
                    c1.write(usr)
                    if c2.button("Delete", key=f"del_{usr}"):
                        del users[usr]
                        save_users(users)
                        st.rerun()
        else:
            st.warning("Admin only")

# ── Main application ─────────────────────────────────────────────────────────
def main_app():
    global df_dict

    st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout", key="logout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    if st.session_state.is_admin:
        if st.sidebar.button("👤 Manage Users", key="manage_users"):
            st.session_state.show_user_management = True
            st.rerun()
    if st.session_state.get("show_user_management", False):
        login_page()
        return

    st.sidebar.title("🚀 AI Market Intelligence Pro+")
    user_input = st.sidebar.text_input("Tickers (comma-separated)", "MSFT, NVDA").upper()
    forecast_days = st.sidebar.slider("Forecast Horizon (days)", 30, 180, 90)
    show_chart = st.sidebar.checkbox("Show Interactive Chart", True)
    show_sr = st.sidebar.checkbox("Detect Support/Resistance", True)

    market_indices = st.sidebar.multiselect(
        "Market Regressors",
        ["^GSPC", "^DJI", "^IXIC", "^VIX", "^TNX"],
        default=["^GSPC", "^DJI", "^IXIC"]
    )
    market_tickers = market_indices

    with st.spinner("Pre-loading market data …"):
        market_dict = fetch_market_only(market_tickers)

    if not market_dict:
        st.error("❌ Could not download market indices.  Check internet or ticker names.")
        st.stop()

    stocks = [s.strip() for s in user_input.split(",") if s.strip()]
    if not stocks:
        stocks = ["TSLA", "MSFT", "NVDA", "GOOG", "AAPL", "AMZN", "AVGO", "CRWD"]

    hp = {"cps": 0.05, "sps": 10}  # default
    run_tune = st.sidebar.checkbox("Auto-tune Prophet (faster now)", False, key="auto_tune")

    with st.spinner("Downloading & aligning market data... (may take 30–90 seconds on first run)"):
        # 1. Prepare data quietly in the background
        df_dict = prepare_aligned_data(stocks, market_tickers)

        # 2. Check if data was actually loaded to prevent crashes later
        if not df_dict:
            st.error("Failed to load market data. Please check your internet connection or ticker symbols.")
            st.stop()
        print("df_dict prepared with keys:", list(df_dict.keys()))
        progress_bar = st.progress(0)
        status_text = st.empty()

        all_tickers = set(stocks + market_tickers)
        total = len(all_tickers)
        
        #raw_data = {}
        #for i, t in enumerate(all_tickers):
        #    status_text.text(f"Downloading {t} ({i+1}/{total})...")
        #    try:
        #        raw_data[t] = fetch_single_ticker(t)
        #    except Exception as e:
        #        st.warning(f"Failed {t}: {e}")
        #    progress_bar.progress((i + 1) / total)
        
        #status_text.text("Aligning common dates...")

    hp = {"cps": 0.05, "sps": 10}
    run_tune = st.sidebar.checkbox("Auto-tune Prophet (faster now)", False)

    if st.sidebar.button("Refresh Market Data (slow)", key="refresh_data"):
        prepare_aligned_data.clear()
        fetch_single_ticker.clear()
        st.success("Data refreshed")

    if st.sidebar.button("🔥 Run Full Analysis", key="run_analysis"):

        for symbol in stocks:
            # Safe place for the print
            print(f"Processing {symbol} - checking df_dict keys: {list(df_dict.keys())}")

            with st.spinner(f"Analyzing {symbol} ..."):
                fcst, errs = forecast_multivariate_cached(
                    symbol,
                    tuple(market_tickers),
                    forecast_days,
                    hp["cps"],
                    hp["sps"]
                )

        print("data caches cleared - forcing fresh download")

        if run_tune and stocks:
            with st.spinner("Hyper-parameter tuning..."):
                hp = tune(stocks[0], tuple(market_tickers))
            st.sidebar.write("Best parameters:", hp)

        for symbol in stocks:
            print(f"Processing {symbol} - checking df_dict keys: {list(df_dict.keys())}")
            with st.spinner(f"Analyzing {symbol} ..."):
                fcst, errs = forecast_multivariate_cached(
                    symbol,
                    tuple(market_tickers),
                    forecast_days,
                    hp["cps"],
                    hp["sps"]
                )

            # 關鍵修正：強制轉成 float scalar
            price_series = df_dict[symbol]["Close"]
            price = float(price_series.iloc[-1]) if not price_series.empty else 0.0

            rsi = calculate_rsi(price_series)
            tp, sl = get_exit_strategy(price, price_series)

            df_sym = df_dict[symbol]
            macd_line, macd_signal, macd_hist = calculate_macd(df_sym["Close"])
            cmf_val = calculate_cmf(df_sym)
            
            # Force sma_200 to scalar
            sma_200_series = df_sym["Close"].rolling(200).mean()
            sma_200 = float(sma_200_series.iloc[-1]) if not sma_200_series.empty else np.nan

            k_val, d_val = calculate_kd(df_sym["High"], df_sym["Low"], df_sym["Close"])
            vpt, vpt_ema = calculate_vpt(df_sym["Close"], df_sym["Volume"])
            obv_val, obv_rising = calculate_obv(df_sym["Close"], df_sym["Volume"])
            bb_mid, bb_upper, bb_lower, bb_bw, bb_percent = calculate_bollinger(df_sym["Close"])
            sr_levels = find_support_resistance(df_sym["Close"]) if show_sr else {'support':[], 'resistance':[]}
            rs_ratio, rs_rating = calculate_relative_strength(df_sym["Close"], df_dict["^GSPC"]["Close"])
            ftd_data = detect_follow_through(df_sym)


            # 輸出部分 - 現在 price 是安全的 float
            st.header(f"📊 {symbol} Analysis ({df_sym.index[-1].date()})")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Recent Forecast Accuracy & Predicted Prices")
                st.caption("MAPE = average error in past 30 days. Predicted prices are from the latest model run.")

                today = pd.Timestamp.now().normalize()  # Use real current date (or fix to 2026-01-16 for testing)
                current_price = float(df_sym["Close"].iloc[-1])  # Make sure it's scalar

                mape_data = []
                for h, v in errs.items():
                    if pd.isna(v):
                        pred_price = "N/A"
                        pred_change = "N/A"
                        est_date = "N/A"
                        quality = "N/A"
                        color = "gray"
                    else:
                        # Calculate future trading date
                        future_date = today + pd.offsets.BDay(h)
                        est_date = future_date.strftime("%Y-%m-%d (%a)")

                        # Get predicted price from forecast (find closest matching date)
                        future_row = fcst[fcst['ds'].dt.date == future_date.date()]
                        if not future_row.empty:
                            pred_price = float(future_row['yhat'].iloc[0])
                            pred_change = (pred_price / current_price - 1) * 100
                            change_str = f"{pred_change:+.1f}%"
                        else:
                            pred_price = "Not in forecast"
                            change_str = "N/A"

                        # Quality rating
                        if v < 0.03:
                            quality = "Excellent (<3%)"
                            color = "green"
                        elif v < 0.07:
                            quality = "Good (3–7%)"
                            color = "orange"
                        else:
                            quality = "High Error (>7%)"
                            color = "red"

                    mape_data.append({
                        "Horizon": f"{h} trading days",
                        "Target date": est_date,
                        "MAPE": f"{v:.2%}",
                        "Predicted price": f"${pred_price:,.2f}" if isinstance(pred_price, (int, float)) else pred_price,
                        "% Change": change_str,
                        "Quality": quality
                    })

                df_mape = pd.DataFrame(mape_data)

                # Nice styled table
                st.dataframe(
                    df_mape.style.applymap(
                        lambda x: f"color: {x}" if x in ["green", "orange", "red", "gray"] else "",
                        subset=["Quality"]
                    ).format({
                        "Predicted price": "${:,.2f}" if isinstance(df_mape["Predicted price"].iloc[0], (int, float)) else "{}"
                    }).hide(axis="index"),
                    use_container_width=True
                )

                # Overall summary
                avg_mape = np.nanmean([v for v in errs.values() if not pd.isna(v)])
                if not np.isnan(avg_mape):
                    if avg_mape < 0.04:
                        st.success(f"Overall: **Very good accuracy** (avg MAPE {avg_mape:.1%}) → reliable for 1–10 day trades")
                    elif avg_mape < 0.08:
                        st.info(f"Overall: **Good accuracy** (avg MAPE {avg_mape:.1%}) → usable for short-term")
                    else:
                        st.warning(f"Overall: **Moderate accuracy** (avg MAPE {avg_mape:.1%}) → use with caution")

                rsi_rows = []
                for period in [3, 5, 7, 9, 14, 21]:
                    val = rsi.get(period, np.nan)
                    if not np.isnan(val):
                        label = get_rsi_label(val)
                        rsi_rows.append({"Period": f"RSI({period})", "Value": f"{val:.1f}", "Interpretation": label})

                if rsi_rows:
                    st.subheader("RSI Multi-Timeframe Overview")
                    st.dataframe(pd.DataFrame(rsi_rows), hide_index=True, use_container_width=True)

            with col2:
                st.metric("Current", f"${price:.2f}")
                st.success(f"Take-Profit: ${tp:.2f}")
                st.error(f"Stop-Loss: ${sl:.2f}")

                st.write(f"**RSI (14D):** {float(rsi[14]):.1f} – {get_rsi_label(rsi[14])}")
                st.write(f"**KD Index (K={float(k_val):.1f}):** " +
                    ("Top of range" if float(k_val) > 80 else "Bottom of range" if float(k_val) < 20 else "Stable"))
                if pd.isna(vpt) or pd.isna(vpt_ema):
                    st.write("VPT: Insufficient data")
                else:
                    st.write(f"VPT: {'Accumulating 🟢' if float(vpt) > float(vpt_ema) else 'Distributing 🔴'}")
                st.write(f"OBV: {'RISING 🟢' if bool(obv_rising) else 'FALLING 🔴'}")
                st.write(f"CMF(20): {float(cmf_val):+.3f} → {'Strong Buying 🟢' if cmf_val > 0.05 else 'Strong Selling 🔴' if cmf_val < -0.05 else 'Neutral'}")
                st.write(f"MACD Hist: {float(macd_hist):+.3f} → {'🟢 Bullish' if float(macd_hist) > 0 else '🔴 Bearish'}")
                st.write(f"%B (BB): {float(bb_percent):.2f} → {'Overbought' if float(bb_percent) > 0.8 else 'Oversold' if float(bb_percent) < 0.2 else 'Neutral'}")
                st.metric("Relative Strength", f"{float(rs_ratio):.2f}x", f"{float(rs_rating):.0f} Rating")

                ftd_status = ftd_data["status"]
                ftd_msg = ftd_data["msg"]
                ftd_color = "normal" if ftd_status == "VALID" else "inverse"

                st.metric("Follow-Through Status", ftd_status, ftd_msg, delta_color=ftd_color)


                st.metric("Price vs 200SMA", "Above" if price > sma_200 else "Below", f"{(price / sma_200 - 1)*100:+.1f}%")

            print(f"{symbol} columns: {df_sym.columns.tolist()}")
            if show_chart:
                if "Close" not in df_sym.columns:
                    st.error(f"No 'Close' column for {symbol} - chart skipped")
                else:
                    recent = df_sym.tail(400).dropna(subset=["Close"])
                    
                    if recent.empty or fcst.empty:
                        st.warning(f"No valid chart data for {symbol}")
                    else:
                        fig = go.Figure()
                        
                        # Historical Price
                        fig.add_trace(go.Scatter(
                            x=recent.index,
                            y=recent["Close"],
                            name="Price",
                            line=dict(width=3)
                        ))
                        
                        # Bollinger Bands (20-day, 2 std) - added back for historical data
                        bb_window = 20
                        bb_std = 2
                        sma_bb = recent["Close"].rolling(window=bb_window).mean()
                        std_bb = recent["Close"].rolling(window=bb_window).std()
                        upper_bb = sma_bb + bb_std * std_bb
                        lower_bb = sma_bb - bb_std * std_bb
                        
                        fig.add_trace(go.Scatter(
                            x=recent.index,
                            y=upper_bb,
                            name="BB Upper",
                            line=dict(color="gray", width=1, dash="dash")
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=recent.index,
                            y=lower_bb,
                            name="BB Lower",
                            line=dict(color="gray", width=1, dash="dash"),
                            fill="tonexty",  # Fill between upper and lower
                            fillcolor="rgba(200,200,200,0.15)",
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=recent.index,
                            y=sma_bb,
                            name="BB Middle (SMA 20)",
                            line=dict(color="orange", width=1, dash="dot")
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=fcst["ds"],
                            y=fcst["yhat"],
                            name="Forecast",
                            line=dict(color="lime", dash="dot")
                        ))
                        
                        # Confidence ribbon
                        fig.add_trace(
                            go.Scatter(
                                x=fcst["ds"].tolist() + fcst["ds"][::-1].tolist(),
                                y=fcst["yhat_upper"].tolist() + fcst["yhat_lower"][::-1].tolist(),
                                fill="toself",
                                fillcolor="rgba(0,176,246,0.15)",
                                line=dict(width=0),
                                name="80% Confidence"
                            )
                        )
                        
                        # Support/Resistance
                        if show_sr and sr_levels.get('support'):
                            fig.add_hline(y=sr_levels['support'][0], line_dash="dash", line_color="green")
                        if show_sr and sr_levels.get('resistance'):
                            fig.add_hline(y=sr_levels['resistance'][0], line_dash="dash", line_color="red")
                        
                        fig.update_layout(
                            title=f"{symbol} Price, Bollinger Bands & Forecast",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            hovermode="x unified",
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{symbol}")

            # Recommendation score
            score = 0
            if rsi[14] < 30: score += 4
            elif rsi[14] > 70: score -= 4
            if price > sma_200: score += 4
            else: score -= 4
            if ftd_data["status"] == "VALID": score += 6
            if cmf_val > 0: score += 2
            else: score -= 2

            if score >= 12:
                st.success(f"🔥 **STRONG BUY** (Score: {score})")
            elif score >= 6:
                st.success(f"✅ **BUY** (Score: {score})")
            elif score <= -6:
                st.error(f"⚠️ **STRONG SELL** (Score: {score})")
            elif score <= -2:
                st.error(f"🔻 **SELL / REDUCE** (Score: {score})")
            else:
                st.warning(f"⚖️ **HOLD / WAIT** (Score: {score})")

# ── Entry point ──────────────────────────────────────────────────────────────
init_auth()
if not st.session_state.authenticated:
    login_page()
else:
    main_app()
