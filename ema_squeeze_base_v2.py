"""
EMA Squeeze Base Scanner v2 — Production 5-10-20-40 Weekly EMA Absorption Engine
---------------------------------------------------------------------------------
Features:
  1. Full Stage 2 Bullish Alignment: 10W > 20W > 40W (sloping up, price above 40W).
  2. Structural Integrity: Zero 40W violations in past 12 weeks (rejects SBIN/Delhivery/HindZinc).
  3. Natural Pullback Flexibility: Evaluates 10W/20W touches without blocking fast EMA curl.
  4. Volume Signature Split: Dry-up Absorption (RVOL <= 0.95x) vs Ignition Breakout (RVOL >= 1.30x).
  5. Dual Rolling vwRS: Stock vs Nifty 500 & Stock vs Sector Index.
  6. Auto-Chunking Telegram Delivery: Prevents 4096-char truncation with plain-text fallback.
"""

import argparse
import csv
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator

# ---------------------------------------------------------------------------
# Strategy Parameters
# ---------------------------------------------------------------------------

NEAR_EMA_TOLERANCE_PCT = 0.030     # 3.0% proximity to 10W / 20W EMA
SUPPORT_CLOSE_TOLERANCE_PCT = 4.0  # Allow close within 4% of EMA during shallow tag
MAX_PCT_OFF_52W_HIGH = 32.0        # Minervini boundary
PULLBACK_RSI_MIN = 48.0            # Trend momentum floor
RS_LOOKBACK_WEEKS = 12             # Rolling vwRS lookback
TRAILING_VOL_WINDOW = 10           # Volume baseline window

# Squeeze & Extension Guards
MAX_EXTENSION_FROM_40W_PCT = 45.0  # Avoid extreme parabolic blow-offs
MAX_EMA_SPREAD_PCT = 6.5           # Max spread between 10W and 20W EMAs
MIN_VW_RS_MARKET = -1.0            # Non-negative RS threshold

# Volume Signatures
VOL_ABSORPTION_MAX_RVOL = 0.95     # Dry-up volume threshold on pullback
HIGH_VOL_IGNITION_RATIO = 1.30     # Ignition volume on breakout

MARKET_INDEX_TICKER = "^CRSLDX"    # Nifty 500
MARKET_INDEX_LABEL = "Nifty 500"
FALLBACK_SECTOR_INDEX_TICKER = "^CRSLDX"

DOWNLOAD_RETRIES = 3
DOWNLOAD_BACKOFF_SECONDS = [1, 3, 6]
DEFAULT_WORKERS = 8
DEFAULT_PER_REQUEST_DELAY = 0.25

MATCH_HISTORY_CSV = "match_history.csv"
MATCH_HISTORY_FIELDS = [
    "scan_run_date", "symbol", "sector", "week_date", "close_at_match",
    "ema10_at_match", "rs_vs_market_at_match", "rs_vs_sector_at_match",
    "monthly_confirmed", "buy_tag", "pullback_ema", "stop_loss_at_match", "risk_pct_at_match"
]

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# ---------------------------------------------------------------------------
# Universe & Sector Benchmark Mapping
# ---------------------------------------------------------------------------

SECTOR_BENCHMARK_MAP = {
    "Financial Services": "^CNXFIN",
    "Automobile and Auto Components": "^CNXAUTO",
    "Fast Moving Consumer Goods": "^CNXFMCG",
    "Healthcare": "^CNXPHARMA",
    "Information Technology": "^CNXIT",
    "Metals & Mining": "^CNXMETAL",
    "Oil Gas & Consumable Fuels": "^CNXENERGY",
    "Power": "^CNXENERGY",
    "Realty": "^CNXREALTY",
    "Capital Goods": "^CRSLDX",
    "Consumer Durables": "^CNXCONSUM",
}

SECTOR_ASSIGNMENTS = {
    "360ONE": "Financial Services", "ABB": "Capital Goods", "ACC": "Capital Goods",
    "APLAPOLLO": "Capital Goods", "AUBANK": "Financial Services", "ADANIENT": "Metals & Mining",
    "ADANIPORTS": "Services", "ADANIPOWER": "Power", "ABCAPITAL": "Financial Services",
    "AFFLE": "Information Technology", "AJANTPHARM": "Healthcare", "ALKEM": "Healthcare",
    "AMBER": "Consumer Durables", "AMBUJACEM": "Capital Goods", "ANGELONE": "Financial Services",
    "APARINDS": "Capital Goods", "APOLLOHOSP": "Healthcare", "APOLLOTYRE": "Automobile and Auto Components",
    "ASHOKLEY": "Automobile and Auto Components", "ASTRAL": "Capital Goods", "AUROPHARMA": "Healthcare",
    "DMART": "Fast Moving Consumer Goods", "AXISBANK": "Financial Services", "BEML": "Capital Goods",
    "BSE": "Financial Services", "BAJAJ-AUTO": "Automobile and Auto Components", "BAJFINANCE": "Financial Services",
    "BAJAJFINSV": "Financial Services", "BALKRISIND": "Automobile and Auto Components", "BANKBARODA": "Financial Services",
    "BEL": "Capital Goods", "BHARATFORG": "Automobile and Auto Components", "BHEL": "Capital Goods",
    "BPCL": "Oil Gas & Consumable Fuels", "BHARTIARTL": "Information Technology", "BIOCON": "Healthcare",
    "BSOFT": "Information Technology", "BLUESTARCO": "Consumer Durables", "BOSCHLTD": "Automobile and Auto Components",
    "BRITANNIA": "Fast Moving Consumer Goods", "CESC": "Power", "CGPOWER": "Capital Goods",
    "CDSL": "Financial Services", "CIPLA": "Healthcare", "COALINDIA": "Metals & Mining",
    "COCHINSHIP": "Capital Goods", "COFORGE": "Information Technology", "COLPAL": "Fast Moving Consumer Goods",
    "CAMS": "Financial Services", "CONCOR": "Services", "CUMMINSIND": "Capital Goods",
    "CYIENT": "Information Technology", "DLF": "Realty", "DABUR": "Fast Moving Consumer Goods",
    "DALBHARAT": "Capital Goods", "DATAPATTNS": "Capital Goods", "DEEPAKNTR": "Healthcare",
    "DELHIVERY": "Services", "DIVISLAB": "Healthcare", "DIXON": "Consumer Durables",
    "DRREDDY": "Healthcare", "EICHERMOT": "Automobile and Auto Components", "ELECON": "Capital Goods",
    "ELGIEQUIP": "Capital Goods", "EMAMILTD": "Fast Moving Consumer Goods", "EXIDEIND": "Automobile and Auto Components",
    "NYKAA": "Fast Moving Consumer Goods", "FEDERALBNK": "Financial Services", "GAIL": "Oil Gas & Consumable Fuels",
    "GRSE": "Capital Goods", "GLENMARK": "Healthcare", "GODREJCP": "Fast Moving Consumer Goods",
    "GODREJPROP": "Realty", "GRASIM": "Capital Goods", "GRAVITA": "Metals & Mining",
    "HCLTECH": "Information Technology", "HDFCAMC": "Financial Services", "HDFCBANK": "Financial Services",
    "HDFCLIFE": "Financial Services", "HAVELLS": "Consumer Durables", "HAL": "Capital Goods",
    "HINDALCO": "Metals & Mining", "HINDPETRO": "Oil Gas & Consumable Fuels", "HINDUNILVR": "Fast Moving Consumer Goods",
    "HUDCO": "Financial Services", "ICICIBANK": "Financial Services", "ICICIGI": "Financial Services",
    "ICICIPRULI": "Financial Services", "IDFCFIRSTB": "Financial Services", "IIFL": "Financial Services",
    "INDIANB": "Financial Services", "INDHOTEL": "Services", "IOC": "Oil Gas & Consumable Fuels",
    "IRCTC": "Services", "IRFC": "Financial Services", "IREDA": "Financial Services",
    "INDUSTOWER": "Information Technology", "INDUSINDBK": "Financial Services", "NAUKRI": "Information Technology",
    "INFY": "Information Technology", "INOXWIND": "Power", "INDIGO": "Services",
    "IPCALAB": "Healthcare", "JKCEMENT": "Capital Goods", "JSWENERGY": "Power",
    "JSWINFRA": "Services", "JSWSTEEL": "Metals & Mining", "JINDALSTEL": "Metals & Mining",
    "JIOFIN": "Financial Services", "JUBLFOOD": "Fast Moving Consumer Goods", "JWL": "Capital Goods",
    "KAYNES": "Capital Goods", "KEI": "Capital Goods", "KOTAKBANK": "Financial Services",
    "LTF": "Financial Services", "LTTS": "Information Technology", "LT": "Capital Goods",
    "LAURUSLABS": "Healthcare", "LICI": "Financial Services", "LODHA": "Realty",
    "LUPIN": "Healthcare", "M&MFIN": "Financial Services", "M&M": "Automobile and Auto Components",
    "MANKIND": "Healthcare", "MARICO": "Fast Moving Consumer Goods", "MARUTI": "Automobile and Auto Components",
    "MAXHEALTH": "Healthcare", "MAZDOCK": "Capital Goods", "MCX": "Financial Services",
    "MUTHOOTFIN": "Financial Services", "NATCOPHARM": "Healthcare", "NBCC": "Realty",
    "NCC": "Capital Goods", "NHPC": "Power", "NMDC": "Metals & Mining",
    "NTPC": "Power", "NATIONALUM": "Metals & Mining", "NAVINFLUOR": "Healthcare",
    "NESTLEIND": "Fast Moving Consumer Goods", "NETWEB": "Information Technology", "OBEROIRLTY": "Realty",
    "ONGC": "Oil Gas & Consumable Fuels", "OIL": "Oil Gas & Consumable Fuels", "PAYTM": "Financial Services",
    "OFSS": "Information Technology", "POLICYBZR": "Financial Services", "PCBL": "Chemicals",
    "PGEL": "Consumer Durables", "PIIND": "Healthcare", "PNBHOUSING": "Financial Services",
    "PERSISTENT": "Information Technology", "PETRONET": "Oil Gas & Consumable Fuels", "PHOENIXLTD": "Realty",
    "PIDILITIND": "Fast Moving Consumer Goods", "POLYCAB": "Capital Goods", "POONAWALLA": "Financial Services",
    "PFC": "Financial Services", "POWERGRID": "Power", "PRESTIGE": "Realty",
    "PNB": "Financial Services", "RVNL": "Capital Goods", "RAILTEL": "Information Technology",
    "RATNAMANI": "Capital Goods", "RELIANCE": "Oil Gas & Consumable Fuels", "SBICARD": "Financial Services",
    "SBILIFE": "Financial Services", "SJVN": "Power", "MOTHERSON": "Automobile and Auto Components",
    "SCHNEIDER": "Capital Goods", "SHREECEM": "Capital Goods", "SHRIRAMFIN": "Financial Services",
    "SIEMENS": "Capital Goods", "SOLARINDS": "Capital Goods", "SONACOMS": "Automobile and Auto Components",
    "SBIN": "Financial Services", "SAIL": "Metals & Mining", "SWSOLAR": "Power",
    "SUNPHARMA": "Healthcare", "SUNTV": "Services", "SUNDARMFIN": "Financial Services",
    "SUZLON": "Power", "TDPOWERSYS": "Capital Goods", "TVSMOTOR": "Automobile and Auto Components",
    "TCS": "Information Technology", "TATACONSUM": "Fast Moving Consumer Goods", "TATAPOWER": "Power",
    "TATASTEEL": "Metals & Mining", "TATATECH": "Information Technology", "TECHM": "Information Technology",
    "TITAGARH": "Capital Goods", "TITAN": "Consumer Durables", "TORNTPHARM": "Healthcare",
    "TORNTPOWER": "Power", "TRENT": "Fast Moving Consumer Goods", "TIINDIA": "Automobile and Auto Components",
    "ULTRACEMCO": "Capital Goods", "UNIONBANK": "Financial Services", "UNITDSPR": "Fast Moving Consumer Goods",
    "VBL": "Fast Moving Consumer Goods", "VEDL": "Metals & Mining", "VOLTAS": "Consumer Durables",
    "WIPRO": "Information Technology", "ZENTEC": "Capital Goods", "ZENSARTECH": "Information Technology",
    "ZYDUSLIFE": "Healthcare"
}

SYMBOLS = list(SECTOR_ASSIGNMENTS.keys())

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    symbol: str
    close: float
    ema5: float
    ema10: float
    ema20: float
    ema40: float
    rsi14: float
    adx14: float
    pdi14: float
    ndi14: float
    compression_pct: float
    ema_spread_pct: float
    week_date: str
    sector: str = ""
    rs_vs_market_pct: float = 0.0
    rs_vs_sector_pct: float = 0.0
    dist_from_52w_high_pct: float = 0.0
    breakout_vol_ratio: float = 0.0
    vol_contracting: bool = False
    tightness_label: str = "Tight"
    buy_tag: bool = True
    stop_loss: float = 0.0
    risk_pct: float = 0.0
    pullback_ema: str = "10W"
    setup_type: str = "pullback_absorption"

# ---------------------------------------------------------------------------
# Data Loading & Indicators
# ---------------------------------------------------------------------------

def _download_with_retry(ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            daily = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True, timeout=12)
            if daily is not None and not daily.empty:
                if isinstance(daily.columns, pd.MultiIndex):
                    daily.columns = daily.columns.get_level_values(0)
                daily.columns = [c.lower() for c in daily.columns]
                return daily
        except Exception:
            pass
        time.sleep(DOWNLOAD_BACKOFF_SECONDS[min(attempt, len(DOWNLOAD_BACKOFF_SECONDS) - 1)])
    return None

def build_weekly(daily: pd.DataFrame) -> Optional[pd.DataFrame]:
    weekly = daily.resample("W-FRI").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()

    today = pd.Timestamp.now().normalize()
    if len(weekly) > 0 and weekly.index[-1] > today:
        weekly = weekly.iloc[:-1]

    if len(weekly) < 45:
        return None

    weekly["ema5"] = weekly["close"].ewm(span=5, adjust=False).mean()
    weekly["ema10"] = weekly["close"].ewm(span=10, adjust=False).mean()
    weekly["ema20"] = weekly["close"].ewm(span=20, adjust=False).mean()
    weekly["ema40"] = weekly["close"].ewm(span=40, adjust=False).mean()

    weekly["rsi14"] = RSIIndicator(close=weekly["close"], window=14).rsi()
    adx_ind = ADXIndicator(high=weekly["high"], low=weekly["low"], close=weekly["close"], window=14)
    weekly["adx14"] = adx_ind.adx()
    weekly["pdi14"] = adx_ind.adx_pos()
    weekly["ndi14"] = adx_ind.adx_neg()

    weekly["high_52w"] = weekly["high"].rolling(window=52, min_periods=40).max()
    weekly["vol_sma10"] = weekly["volume"].rolling(window=TRAILING_VOL_WINDOW).mean()
    return weekly

def compute_volume_weighted_rs_series(stock_weekly: pd.DataFrame, benchmark_weekly: pd.DataFrame) -> pd.Series:
    if stock_weekly is None or benchmark_weekly is None:
        return pd.Series(dtype=float)

    idx_s = pd.to_datetime(stock_weekly.index).tz_localize(None)
    idx_b = pd.to_datetime(benchmark_weekly.index).tz_localize(None)
    
    sw = stock_weekly.copy()
    bw = benchmark_weekly.copy()
    sw.index = idx_s
    bw.index = idx_b

    combined = pd.DataFrame({
        "stock_close": sw["close"],
        "stock_vol": sw["volume"],
        "stock_vol_sma": sw["vol_sma10"],
        "bench_close": bw["close"],
    }).dropna()

    if len(combined) < RS_LOOKBACK_WEEKS + 1:
        return pd.Series(dtype=float)

    combined["ratio"] = combined["stock_close"] / combined["bench_close"]
    combined["ratio_return"] = combined["ratio"].pct_change()
    combined["vol_weight"] = combined["stock_vol"] / combined["stock_vol_sma"].replace(0, 1)
    combined["vw_rs_comp"] = combined["ratio_return"] * combined["vol_weight"]

    rs_series = (combined["vw_rs_comp"].rolling(window=RS_LOOKBACK_WEEKS).sum() * 100).round(2)
    return rs_series.dropna()

# ---------------------------------------------------------------------------
# Setup Evaluator: Robust 5-10-20-40 Weekly Absorption Engine
# ---------------------------------------------------------------------------

def evaluate_conditions(df: pd.DataFrame, idx: int, rs_mkt_series: pd.Series, rs_sec_series: pd.Series) -> Optional[dict]:
    if idx < 30 or idx >= len(df):
        return None

    row = df.iloc[idx]
    prev_row = df.iloc[idx - 1]

    # --- 1. CORE STAGE 2 TREND TEMPLATE ---
    if pd.isna(row.ema40) or pd.isna(row.ema20) or pd.isna(row.ema10):
        return None

    ema40_prev4 = df.iloc[idx - 4]["ema40"]
    ema10_prev4 = df.iloc[idx - 4]["ema10"]

    trend_rising = (row.ema40 >= ema40_prev4) and (row.ema10 >= ema10_prev4)
    stage2_stack = (row.ema10 > row.ema20) and (row.ema20 > row.ema40) and (row.close > row.ema40)

    if not (trend_rising and stage2_stack):
        return None

    # --- 2. ANTI-CHOP & DAMAGE FILTER (No 40W violations in past 12 weeks) ---
    recent_lows = df["low"].iloc[max(0, idx - 12): idx + 1]
    recent_ema40 = df["ema40"].iloc[max(0, idx - 12): idx + 1]
    if (recent_lows < recent_ema40 * 0.97).any():
        return None

    # --- 3. SQUEEZE COILING & EXTENSION GUARDS ---
    ema_spread_pct = abs(row.ema10 - row.ema20) / row.close * 100
    if ema_spread_pct > MAX_EMA_SPREAD_PCT:
        return None

    extension_from_40w = (row.close - row.ema40) / row.ema40 * 100
    if extension_from_40w > MAX_EXTENSION_FROM_40W_PCT:
        return None

    # --- 4. SHALLOW PULLBACK TEST (10W or 20W Pocket) ---
    touch_10w = (row.low <= row.ema10 * (1 + NEAR_EMA_TOLERANCE_PCT)) and (row.close >= row.ema10 * (1 - SUPPORT_CLOSE_TOLERANCE_PCT / 100))
    touch_20w = (row.low <= row.ema20 * (1 + NEAR_EMA_TOLERANCE_PCT)) and (row.close >= row.ema20 * (1 - SUPPORT_CLOSE_TOLERANCE_PCT / 100))

    if not (touch_10w or touch_20w):
        return None

    pullback_ema = "10W" if abs(row.close - row.ema10) <= abs(row.close - row.ema20) else "20W"

    # --- 5. CANDLE ANATOMY & TIGHT RESTING RANGE ---
    candle_range = row.high - row.low
    bar_range_pct = (candle_range / row.close) * 100 if row.close > 0 else 10.0
    upper_half_close = (row.close >= row.low + (candle_range * 0.38)) if candle_range > 0 else True
    is_tight_bar = bar_range_pct <= 6.5

    if not (upper_half_close or is_tight_bar):
        return None

    # --- 6. VOLUME SIGNATURE (Dry-up Absorption OR Ignition Breakout) ---
    vol_avg = row.get("vol_sma10", 0)
    if pd.isna(vol_avg) or vol_avg <= 0:
        return None

    rvol = round(float(row.volume / vol_avg), 2)
    is_quiet_absorption = (rvol <= VOL_ABSORPTION_MAX_RVOL)
    is_ignition_breakout = (rvol >= HIGH_VOL_IGNITION_RATIO) and (row.close >= prev_row.high * 0.99)

    if not (is_quiet_absorption or is_ignition_breakout):
        return None

    # --- 7. DUAL RELATIVE STRENGTH (vwRS) ---
    vw_rs_mkt = rs_mkt_series.get(row.name, rs_mkt_series.iloc[-1]) if not rs_mkt_series.empty else 0.0
    vw_rs_sec = rs_sec_series.get(row.name, rs_sec_series.iloc[-1]) if not rs_sec_series.empty else 0.0

    if vw_rs_mkt < MIN_VW_RS_MARKET:
        return None

    # RSI & 52W Distance Filters
    rsi_val = row.get("rsi14", 50.0)
    if pd.notna(rsi_val) and rsi_val < PULLBACK_RSI_MIN:
        return None

    high_52w = row.get("high_52w", float("nan"))
    dist_off_52w = round((high_52w - row.close) / high_52w * 100, 2) if pd.notna(high_52w) and high_52w > 0 else 0.0
    if dist_off_52w > MAX_PCT_OFF_52W_HIGH:
        return None

    stop_loss = round(min(row.low, prev_row.low) * 0.99, 2)
    risk_pct = round((row.close - stop_loss) / row.close * 100, 2)

    return {
        "row": row,
        "pullback_ema": pullback_ema,
        "dist_from_52w_high_pct": dist_off_52w,
        "stop_loss": stop_loss,
        "risk_pct": risk_pct,
        "rvol": rvol,
        "vw_rs_mkt": vw_rs_mkt,
        "vw_rs_sec": vw_rs_sec,
        "ema_spread_pct": round(ema_spread_pct, 2),
        "setup_type": "🚀 Breakout Ignition" if is_ignition_breakout else "🛡️ 10W/20W Absorption",
    }

def _build_scan_result(evald: dict) -> ScanResult:
    row = evald["row"]
    return ScanResult(
        symbol="",
        close=round(row.close, 2),
        ema5=round(row.ema5, 2),
        ema10=round(row.ema10, 2),
        ema20=round(row.ema20, 2),
        ema40=round(row.ema40, 2),
        rsi14=round(row.rsi14, 2) if pd.notna(row.get("rsi14")) else 50.0,
        adx14=round(row.adx14, 2) if pd.notna(row.get("adx14")) else 0.0,
        pdi14=round(row.pdi14, 2) if pd.notna(row.get("pdi14")) else 0.0,
        ndi14=round(row.ndi14, 2) if pd.notna(row.get("ndi14")) else 0.0,
        compression_pct=evald["ema_spread_pct"],
        ema_spread_pct=evald["ema_spread_pct"],
        week_date=str(row.name.date()),
        dist_from_52w_high_pct=evald["dist_from_52w_high_pct"],
        breakout_vol_ratio=evald["rvol"],
        vol_contracting=("Absorption" in evald["setup_type"]),
        tightness_label="Tight",
        buy_tag=True,
        stop_loss=evald["stop_loss"],
        risk_pct=evald["risk_pct"],
        pullback_ema=evald["pullback_ema"],
        setup_type=evald["setup_type"],
        rs_vs_market_pct=evald["vw_rs_mkt"],
        rs_vs_sector_pct=evald["vw_rs_sec"],
    )

# ---------------------------------------------------------------------------
# Parallel Execution Engine
# ---------------------------------------------------------------------------

def scan_symbol(symbol: str, market_weekly: pd.DataFrame, sector_cache: dict, backtest: bool, lookback_weeks: int, weekly_cache: dict = None):
    daily = _download_with_retry(f"{symbol}.NS", period="5y")
    if daily is None:
        return []

    weekly = build_weekly(daily)
    if weekly is None:
        return []

    if weekly_cache is not None:
        weekly_cache[symbol] = weekly

    sector = SECTOR_ASSIGNMENTS.get(symbol, "Other")
    sec_ticker = SECTOR_BENCHMARK_MAP.get(sector, FALLBACK_SECTOR_INDEX_TICKER)
    sector_weekly = sector_cache.get(sec_ticker, market_weekly)

    rs_mkt_series = compute_volume_weighted_rs_series(weekly, market_weekly)
    rs_sec_series = compute_volume_weighted_rs_series(weekly, sector_weekly)

    results = []
    if backtest:
        start_idx = max(30, len(weekly) - lookback_weeks)
        for i in range(start_idx, len(weekly)):
            evald = evaluate_conditions(weekly, i, rs_mkt_series, rs_sec_series)
            if evald:
                res = _build_scan_result(evald)
                res.symbol = symbol
                res.sector = sector
                results.append(res)
    else:
        evald = evaluate_conditions(weekly, len(weekly) - 1, rs_mkt_series, rs_sec_series)
        if evald:
            res = _build_scan_result(evald)
            res.symbol = symbol
            res.sector = sector
            results.append(res)

    return results

# ---------------------------------------------------------------------------
# Telegram Delivery with Auto-Chunking Engine
# ---------------------------------------------------------------------------

def format_results_message(buy_setups: List[ScanResult]) -> str:
    lines = [
        "⚡ *5-10-20-40 Weekly EMA Absorption Scanner* ⚡",
        f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}\n"
    ]
    if not buy_setups:
        lines.append("No candidates passed all strict high-probability absorption filters this week.")
        return "\n".join(lines)

    lines.append(f"🎯 *HIGH-PROBABILITY SETUPS ({len(buy_setups)})*\n")
    for r in buy_setups:
        lines.append(
            f"⭐ *{r.symbol}* [{r.sector}] — `{r.setup_type}`\n"
            f"   • Close: ₹{r.close} | Support: *{r.pullback_ema} EMA* | Spread: {r.ema_spread_pct}%\n"
            f"   • RVOL: *{r.breakout_vol_ratio}x* | RSI: {r.rsi14} | ADX: {r.adx14}\n"
            f"   • vwRS (vs Nifty 500): *{r.rs_vs_market_pct:+.1f}* | vs Sector: *{r.rs_vs_sector_pct:+.1f}*\n"
            f"   • Off 52W High: -{r.dist_from_52w_high_pct}%\n"
            f"   • 🎯 SL Anchor: ₹{r.stop_loss} ({r.risk_pct}% Risk Box)\n"
        )
    return "\n".join(lines)

def split_message_into_chunks(text: str, max_length: int = 3500) -> List[str]:
    if len(text) <= max_length:
        return [text]

    blocks = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for block in blocks:
        if len(current_chunk) + len(block) + 2 <= max_length:
            current_chunk = (current_chunk + "\n\n" + block) if current_chunk else block
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = block

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram Credentials Missing — TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is empty.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = split_message_into_chunks(text)
    print(f"[INFO] Sending Telegram alert in {len(chunks)} chunk(s)...")

    for i, chunk in enumerate(chunks, 1):
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "Markdown"}
        try:
            resp = requests.post(url, data=payload, timeout=15)
            if resp.status_code == 200:
                print(f"[SUCCESS] Telegram chunk {i}/{len(chunks)} delivered (HTTP 200).")
            else:
                print(f"[WARN] Markdown delivery failed (HTTP {resp.status_code}). Trying plain text fallback...")
                payload_plain = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk}
                resp_fallback = requests.post(url, data=payload_plain, timeout=15)
                if resp_fallback.status_code == 200:
                    print(f"[SUCCESS] Telegram chunk {i}/{len(chunks)} delivered via fallback.")
                else:
                    print(f"[ERROR] Delivery Failed (HTTP {resp_fallback.status_code}): {resp_fallback.text}")
        except Exception as e:
            print(f"[ERROR] Request failed on chunk {i}: {e}")
        time.sleep(0.5)

def log_matches_to_history(results: List[ScanResult], csv_path: str = MATCH_HISTORY_CSV):
    existing_keys = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                existing_keys.add((row.get("symbol", ""), row.get("week_date", "")))

    new_rows = []
    run_date = datetime.now().strftime("%Y-%m-%d")
    for r in results:
        key = (r.symbol, r.week_date)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        new_rows.append({
            "scan_run_date": run_date,
            "symbol": r.symbol,
            "sector": r.sector or "",
            "week_date": r.week_date,
            "close_at_match": r.close,
            "ema10_at_match": r.ema10,
            "rs_vs_market_at_match": r.rs_vs_market_pct,
            "rs_vs_sector_at_match": r.rs_vs_sector_pct,
            "monthly_confirmed": True,
            "buy_tag": r.buy_tag,
            "pullback_ema": r.pullback_ema or "",
            "stop_loss_at_match": r.stop_loss,
            "risk_pct_at_match": r.risk_pct,
        })

    if not new_rows:
        return

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MATCH_HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EMA Squeeze Base weekly scanner v2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--lookback-weeks", type=int, default=26)
    parser.add_argument("--forward-weeks", type=int, default=8)
    parser.add_argument("--delay", type=float, default=DEFAULT_PER_REQUEST_DELAY)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-log-history", action="store_true")
    parser.add_argument("--weekly-report", action="store_true")
    parser.add_argument("--report-lookback-weeks", type=int, default=8)
    args = parser.parse_args()

    print(f"Fetching Market Benchmark ({MARKET_INDEX_LABEL})...")
    market_daily = _download_with_retry(MARKET_INDEX_TICKER, period="5y")
    market_weekly = build_weekly(market_daily) if market_daily is not None else None

    if market_weekly is None:
        print("[ERROR] Could not load market index.")
        sys.exit(1)

    sector_cache = {}
    print("Prefetching Sector Benchmarks...")
    for sec_name, sec_ticker in set(SECTOR_BENCHMARK_MAP.items()):
        daily_sec = _download_with_retry(sec_ticker, period="5y")
        if daily_sec is not None:
            wk = build_weekly(daily_sec)
            if wk is not None:
                sector_cache[sec_ticker] = wk

    symbols = SYMBOLS[: args.limit] if args.limit else SYMBOLS
    print(f"Scanning {len(symbols)} candidates with {args.workers} workers...")

    all_candidates: List[ScanResult] = []
    weekly_cache = {} if args.backtest else None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(scan_symbol, s, market_weekly, sector_cache, args.backtest, args.lookback_weeks, weekly_cache): s for s in symbols}
        for f in as_completed(futures):
            try:
                res = f.result()
                all_candidates.extend(res)
            except Exception:
                pass

    msg = format_results_message(all_candidates)
    print("\n" + msg)

    if not args.no_log_history and all_candidates and not args.backtest:
        log_matches_to_history(all_candidates)

    if not args.dry_run and not args.backtest:
        send_telegram_message(msg)

if __name__ == "__main__":
    main()
