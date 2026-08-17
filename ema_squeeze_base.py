def _base_duration_weeks(df: pd.DataFrame, idx: int, near_pct: float = NEAR_EMA_PCT) -> int:
    """
    Counts consecutive weeks, walking backward from idx, where close stayed within near_pct of
    the 5W, 10W, or 20W EMA.
    """
    count = 0
    i = idx
    while i >= 0:
        row = df.iloc[i]
        if pd.isna(row.get("ema5")) or pd.isna(row.get("ema10")) or pd.isna(row.get("ema20")) or row.close == 0:
            break
        dist5 = abs(row.close - row.ema5) / row.close
        dist10 = abs(row.close - row.ema10) / row.close
        dist20 = abs(row.close - row.ema20) / row.close
        if dist5 <= near_pct or dist10 <= near_pct or dist20 <= near_pct:
            count += 1
            i -= 1
        else:
            break
    return count

def evaluate_conditions(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 1 or idx >= len(df):
        return None

    row = df.iloc[idx]

    required = ["ema5", "ema10", "ema20", "ema40"]
    if row[required].isna().any():
        return None

    dist_ema5 = abs(row.close - row.ema5) / row.close
    dist_ema10 = abs(row.close - row.ema10) / row.close
    dist_ema20 = abs(row.close - row.ema20) / row.close
    
    near_ema5 = dist_ema5 <= NEAR_EMA_PCT
    near_ema10 = dist_ema10 <= NEAR_EMA_PCT
    near_ema20 = dist_ema20 <= NEAR_EMA_PCT
    
    # Expanded to allow proximity to 5W, 10W, or 20W EMA
    near_any = near_ema5 or near_ema10 or near_ema20

    uptrend_ok = bool(row.ema10 > row.ema20 > row.ema40 and row.close > row.ema40)

    adx_val = row.get("adx14", float("nan"))
    adx_ok = bool(pd.notna(adx_val) and adx_val >= ADX_MIN)

    monthly_uptrend = bool(row.get("monthly_uptrend", False))
    monthly_dist = row.get("dist_to_ema6_m_pct", 999.0)
    monthly_retest = bool(pd.notna(monthly_dist) and monthly_dist <= MONTHLY_RETEST_PCT)
    monthly_confirmed = monthly_uptrend and monthly_retest

    # --- 52-week-high gate (Minervini Trend Template #7) ---
    high_52w = row.get("high_52w", float("nan"))
    if pd.notna(high_52w) and high_52w > 0:
        dist_from_52w_high_pct = round((high_52w - row.close) / high_52w * 100, 2)
        near_52w_high_ok = dist_from_52w_high_pct <= MAX_PCT_OFF_52W_HIGH
        pct_detail = f"{dist_from_52w_high_pct:.1f}% off 52W high (need <= {MAX_PCT_OFF_52W_HIGH:.0f}%)"
    else:
        dist_from_52w_high_pct = None
        near_52w_high_ok = False
        pct_detail = "52W high not available yet (needs 52 weeks of history)"

    # --- Minimum base duration gate ---
    base_weeks = _base_duration_weeks(df, idx)
    base_duration_ok = base_weeks >= MIN_BASE_WEEKS

    checks = {
        "near_5w_10w_or_20w_ema": (near_any,
            f"close {row.close:.2f} | dist to EMA5 {dist_ema5*100:.2f}% | dist to EMA10 {dist_ema10*100:.2f}% | dist to EMA20 {dist_ema20*100:.2f}% (need <= {NEAR_EMA_PCT*100:.0f}% to any)"),
        "uptrend":            (uptrend_ok if UPTREND_REQUIRED else True,
            f"ema10 {row.ema10:.2f} > ema20 {row.ema20:.2f} > ema40 {row.ema40:.2f}, close {row.close:.2f} > ema40: {uptrend_ok}"),
        "adx_min":            (adx_ok,
            f"ADX {adx_val:.1f} (need >= {ADX_MIN})" if pd.notna(adx_val) else "ADX not available"),
        "near_52w_high":      (near_52w_high_ok, pct_detail),
        "min_base_duration":  (base_duration_ok,
            f"base held {base_weeks} week(s) (need >= {MIN_BASE_WEEKS})"),
    }

    breakout_vol_ratio, vol_contracting = _volume_profile(df, idx, base_weeks)

    return {
        "row": row, "checks": checks, "monthly_confirmed": monthly_confirmed,
        "dist_from_52w_high_pct": dist_from_52w_high_pct, "base_weeks": base_weeks,
        "breakout_vol_ratio": breakout_vol_ratio, "vol_contracting": vol_contracting,
    }
