def evaluate_conditions(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 1 or idx >= len(df):
        return None

    row = df.iloc[idx]
    required = ["ema5", "ema10", "ema20", "ema40"]
    if row[required].isna().any():
        return None

    # 1. Price MUST hold above or at the 20W EMA (No 40W breakdown drifts)
    holds_ema20_support = bool(row.close >= row.ema20 * 0.99)

    # 2. Distance to 10W or 20W EMA
    dist_ema10 = abs(row.close - row.ema10) / row.close
    dist_ema20 = abs(row.close - row.ema20) / row.close
    near_support = (dist_ema10 <= NEAR_EMA_PCT) or (dist_ema20 <= NEAR_EMA_PCT)

    # 3. EMA Squeeze (5, 10, 20 bundled together tightly)
    ema_spread_pct = (max(row.ema5, row.ema10, row.ema20) - min(row.ema5, row.ema10, row.ema20)) / row.close * 100
    squeeze_ok = ema_spread_pct <= EMA_SPREAD_MAX_PCT

    # 4. Strict Bullish Alignment
    uptrend_ok = bool(row.ema10 >= row.ema20 > row.ema40 and row.close > row.ema20)

    # 5. 52-Week High Proximity (Strictly within 15%)
    high_52w = row.get("high_52w", float("nan"))
    dist_from_52w_high_pct = round((high_52w - row.close) / high_52w * 100, 2) if pd.notna(high_52w) else None
    near_52w_high_ok = bool(dist_from_52w_high_pct is not None and dist_from_52w_high_pct <= 15.0)

    # 6. ADX & Base Risk
    adx_val = row.get("adx14", float("nan"))
    adx_ok = bool(pd.notna(adx_val) and adx_val >= ADX_MIN)

    base_weeks = _base_duration_weeks(df, idx)
    base_duration_ok = base_weeks >= MIN_BASE_WEEKS

    start_base_idx = max(0, idx - max(base_weeks, 1))
    base_slice = df.iloc[start_base_idx: idx + 1]
    stop_loss = round(float(base_slice["low"].min()) * 0.99, 2)
    risk_pct = round((row.close - stop_loss) / row.close * 100, 2)
    risk_ok = risk_pct <= MAX_RISK_PCT

    checks = {
        "holds_ema_support": (holds_ema20_support, f"close {row.close:.2f} >= 20W EMA {row.ema20:.2f}"),
        "near_10w_or_20w_ema": (near_support, f"dist: 10W={dist_ema10*100:.1f}%, 20W={dist_ema20*100:.1f}%"),
        "ema_squeeze": (squeeze_ok, f"spread: {ema_spread_pct:.2f}% (<= {EMA_SPREAD_MAX_PCT:.1f}%)"),
        "uptrend": (uptrend_ok, f"stack: 10W>=20W>40W"),
        "leader_near_high": (near_52w_high_ok, f"{dist_from_52w_high_pct}% off 52W high (need <= 15%)"),
        "adx_min": (adx_ok, f"ADX: {adx_val:.1f} (>= {ADX_MIN})"),
        "min_base_duration": (base_duration_ok, f"base: {base_weeks}w (>= {MIN_BASE_WEEKS}w)"),
        "risk_within_max": (risk_ok, f"risk: {risk_pct:.2f}% (<= {MAX_RISK_PCT:.1f}%)"),
    }

    breakout_vol_ratio, vol_contracting = _volume_profile(df, idx, base_weeks)

    monthly_uptrend = bool(row.get("monthly_uptrend", False))
    monthly_dist = row.get("dist_to_ema6_m_pct", 999.0)
    monthly_confirmed = monthly_uptrend and (pd.notna(monthly_dist) and monthly_dist <= MONTHLY_RETEST_PCT)

    return {
        "row": row, "checks": checks, "monthly_confirmed": monthly_confirmed,
        "dist_from_52w_high_pct": dist_from_52w_high_pct, "base_weeks": base_weeks,
        "breakout_vol_ratio": breakout_vol_ratio, "vol_contracting": vol_contracting,
        "ema_spread_pct": ema_spread_pct, "stop_loss": stop_loss, "risk_pct": risk_pct,
    }
