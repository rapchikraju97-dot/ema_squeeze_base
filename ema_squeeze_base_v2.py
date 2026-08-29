def evaluate_conditions(df: pd.DataFrame, idx: int, rs_mkt_series: pd.Series, rs_sec_series: pd.Series) -> Optional[dict]:
    if idx < 45 or idx >= len(df):
        return None

    row = df.iloc[idx]
    prev_row = df.iloc[idx - 1]

    # --- 1. CURRENT MOVING AVERAGE STACK (10W > 20W > 40W) ---
    if pd.isna(row.ema40) or pd.isna(row.ema20) or pd.isna(row.ema10):
        return None

    ema40_prev4 = df.iloc[idx - 4]["ema40"]
    ema10_prev4 = df.iloc[idx - 4]["ema10"]

    # 10W and 40W EMAs must be sloping upward
    emas_rising = (row.ema40 > ema40_prev4) and (row.ema10 > ema10_prev4)
    bullish_stack = (row.ema10 > row.ema20) and (row.ema20 > row.ema40) and (row.close > row.ema40)

    if not (emas_rising and bullish_stack):
        return None

    # --- 2. PRIOR 10W CROSSING 40W FROM BELOW ---
    # Look back between 10 to 45 weeks for the Stage 2 Initiation cross
    crossover_idx = None
    lookback_start = max(1, idx - 45)
    lookback_end = max(1, idx - 8)  # Must not be on the current week (need an upmove first)

    for i in range(lookback_end, lookback_start, -1):
        curr_bar = df.iloc[i]
        prev_bar = df.iloc[i - 1]
        if prev_bar.ema10 <= prev_bar.ema40 and curr_bar.ema10 > curr_bar.ema40:
            crossover_idx = i
            break

    if crossover_idx is None:
        return None  # No prior crossover found

    # --- 3. CONFIRMED PRIOR UPMOVE (Impulse wave from cross) ---
    cross_price = df.iloc[crossover_idx]["close"]
    highest_after_cross = df["high"].iloc[crossover_idx: idx + 1].max()
    upmove_pct = ((highest_after_cross - cross_price) / cross_price) * 100

    if upmove_pct < 20.0:  # Must have rallied at least +20% after crossover
        return None

    # --- 4. NO STRUCTURAL DAMAGE (40W EMA held continuously since cross) ---
    cycle_lows = df["low"].iloc[crossover_idx: idx + 1]
    cycle_ema40 = df["ema40"].iloc[crossover_idx: idx + 1]
    if (cycle_lows < cycle_ema40 * 0.98).any():
        return None  # Reject if price broke below 40W EMA

    # --- 5. THE PULLBACK POCKET: TESTING 10W / 20W EMA ---
    # Low touches 10W/20W EMA, while close defends the 20W EMA
    touch_10w = (row.low <= row.ema10 * 1.025) and (row.close >= row.ema20 * 0.985)
    touch_20w = (row.low <= row.ema20 * 1.025) and (row.close >= row.ema20 * 0.985)

    if not (touch_10w or touch_20w):
        return None

    pullback_ema = "10W" if abs(row.close - row.ema10) <= abs(row.close - row.ema20) else "20W"

    # --- 6. VOLUME DRY-UP ON THE PULLBACK (VPA Absorption) ---
    vol_avg = row.get("vol_sma10", 0)
    if pd.isna(vol_avg) or vol_avg <= 0:
        return None

    rvol = round(float(row.volume / vol_avg), 2)
    is_quiet_absorption = (rvol <= 0.85)  # Dry turnover during rest
    is_breakout_turn = (rvol >= 1.25) and (row.close > prev_row.high)

    if not (is_quiet_absorption or is_breakout_turn):
        return None

    # --- 7. DUAL RELATIVE STRENGTH (vwRS) ---
    vw_rs_mkt = rs_mkt_series.get(row.name, rs_mkt_series.iloc[-1]) if not rs_mkt_series.empty else 0.0
    vw_rs_sec = rs_sec_series.get(row.name, rs_sec_series.iloc[-1]) if not rs_sec_series.empty else 0.0

    if vw_rs_mkt < 0.0:  # Must outperform or equal broader market
        return None

    # Stop Loss & Risk Anchor (Swing Low of base)
    stop_loss = round(min(row.low, prev_row.low) * 0.99, 2)
    risk_pct = round((row.close - stop_loss) / row.close * 100, 2)
    ema_spread_pct = round(abs(row.ema10 - row.ema20) / row.close * 100, 2)
    weeks_since_cross = idx - crossover_idx

    return {
        "row": row,
        "pullback_ema": pullback_ema,
        "dist_from_52w_high_pct": round((row.high_52w - row.close) / row.high_52w * 100, 2) if pd.notna(row.high_52w) else 0.0,
        "stop_loss": stop_loss,
        "risk_pct": risk_pct,
        "rvol": rvol,
        "vw_rs_mkt": vw_rs_mkt,
        "vw_rs_sec": vw_rs_sec,
        "ema_spread_pct": ema_spread_pct,
        "weeks_since_crossover": weeks_since_cross,
        "post_cross_gain_pct": round(upmove_pct, 1),
        "setup_type": "🚀 Base Expansion" if is_breakout_turn else "🛡️ 10W/20W Absorption Base",
    }
