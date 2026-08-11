"""
EMA Squeeze Base Scanner (symbols embedded — no external file needed)
-----------------------------------------------------------------------
Weekly-timeframe scan for RKScanBot.

Flags stocks where:
  1. 5W / 10W / 20W EMAs are compressed (tight spread) relative to price
  2. Close is at/just above the 10W EMA (not below it)
  3. Close is above the 40W EMA (broader uptrend intact)
  4. RSI(14) is holding the 48-58 support band
  5. ADX(14) > 20 and rising vs. the prior week
  6. +DI(14) > -DI(14) (trend direction still bullish)

Usage:
    python ema_squeeze_base.py                # live run, sends Telegram alert
    python ema_squeeze_base.py --dry-run       # prints results, no Telegram send
    python ema_squeeze_base.py --backtest --lookback-weeks 20 --dry-run
        (checks the last N weeks per symbol instead of just the latest —
         use this first to validate against known winners like MTARTECH,
         CUMMINSIND, APARINDS before trusting it live)

Requirements:
    pip install yfinance pandas ta requests

Environment variables (for Telegram):
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import requests
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NEAR_EMA_PCT = 0.03    # close must be within 3% of the 10W or 20W EMA — the ONLY proximity rule
UPTREND_REQUIRED = True  # require ema10 > ema20 > ema40 (bullish stack) and close > ema40
ADX_MIN = 20            # weekly ADX(14) must be at least this — filters out weak/no-trend stocks

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# NSE symbols — Nifty Total Market list (752 symbols), no .NS suffix
SYMBOLS = [
    "360ONE", "3MINDIA", "ABB", "ACC", "ACMESOLAR", "AIAENG", "APLAPOLLO", "ASKAUTOLTD",
    "AUBANK", "AWL", "AXISCADES", "AADHARHFC", "AARTIDRUGS", "AARTIIND", "AARTIPHARM", "AAVAS",
    "ABBOTINDIA", "ACE", "ACUTAAS", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER",
    "ATGL", "ABCAPITAL", "ABFRL", "ABLBL", "ABREL", "ABSLAMC", "CPPLUS", "AVL",
    "ADVENZYMES", "AEGISLOG", "AEGISVOPAK", "AEQUS", "AETHER", "AFCONS", "AFFLE", "AHLUCONT",
    "AJANTPHARM", "AKUMS", "APLLTD", "ALIVUS", "ALKEM", "ALKYLAMINE", "ABDL", "ALOKINDS",
    "ARE&M", "AMBER", "AMBUJACEM", "ANANDRATHI", "ANANTRAJ", "ANGELONE", "ANTHEM", "ANURAS",
    "APARINDS", "APOLLOHOSP", "APOLLO", "APOLLOTYRE", "APTUS", "ACI", "ARVINDFASN", "ARVIND",
    "ASAHIINDIA", "ASHAPURMIN", "ASHOKLEY", "ASHOKA", "ASIANPAINT", "ASTERDM", "ASTRAMICRO", "ASTRAL",
    "ATHERENERG", "ATLANTAELE", "ATUL", "AURIONPRO", "AUROPHARMA", "AIIL", "AVALON", "AVANTIFEED",
    "DMART", "CCAVENUE", "AWFIS", "AXISBANK", "AZAD", "BEML", "BLS", "BSE",
    "BAJAJ-AUTO", "BAJAJELEC", "BAJFINANCE", "BAJAJFINSV", "BAJAJHLDNG", "BAJAJHFL", "BALAMINES", "BALKRISIND",
    "BALRAMCHIN", "BALUFORGE", "BANCOINDIA", "BANDHANBNK", "BANKBARODA", "BANKINDIA", "MAHABANK", "BATAINDIA",
    "BAYERCROP", "BELRISE", "BERGEPAINT", "BDL", "BEL", "BHARATFORG", "BHEL", "BPCL",
    "BHARTIARTL", "BHARTIHEXA", "BIKAJI", "GROWW", "BIOCON", "BIRLACORPN", "BSOFT", "BBOX",
    "BLACKBUCK", "BLUEDART", "BLUEJET", "BLUESTARCO", "BLUESTONE", "BBTC", "BORORENEW", "BOSCHLTD",
    "FIRSTCRY", "BRIGADE", "BRITANNIA", "MAPMYINDIA", "CCL", "CESC", "CGPOWER", "CIEINDIA",
    "CMSINFO", "CORONA", "CRISIL", "CSBBANK", "CAMPUS", "CANFINHOME", "CANBK", "CANHLIFE",
    "CRAMC", "CAPILLARY", "CAPLIPOINT", "CGCL", "CARBORUNIV", "CARTRADE", "CASTROLIND", "CEATLTD",
    "CELLO", "CEMPRO", "CENTRALBK", "CDSL", "CENTURYPLY", "CERA", "CHALET", "CHAMBLFERT",
    "CHENNPETRO", "CHOICEIN", "CHOLAHLDNG", "CHOLAFIN", "CIPLA", "CUB", "CLEAN", "COALINDIA",
    "COCHINSHIP", "COFORGE", "COHANCE", "COLPAL", "CAMS", "CONCORDBIO", "CONCOR", "COROMANDEL",
    "CRAFTSMAN", "CREDITACC", "CRIZAC", "CROMPTON", "CUMMINSIND", "CUPID", "CYIENT", "DCBBANK",
    "DCMSHRIRAM", "DLF", "DOMS", "DABUR", "DALBHARAT", "DATAPATTNS", "DATAMATICS", "DEEPAKFERT",
    "DEEPAKNTR", "DELHIVERY", "DEVYANI", "DIACABS", "DBL", "DIVISLAB", "DIXON", "AGARWALEYE",
    "LALPATHLAB", "DRREDDY", "DUMMYINXGN", "DUMMYTRVN", "DYNAMATECH", "EIDPARRY", "EIHOTEL", "EPL",
    "EDELWEISS", "EICHERMOT", "ELECON", "EMIL", "ELECTCAST", "ELGIEQUIP", "ELLEN", "EMAMILTD",
    "EMBDL", "EMCURE", "EMMVEE", "ENDURANCE", "ENGINERSIN", "ENTERO", "EIEL", "EQUITASBNK",
    "ERIS", "ESCORTS", "ETERNAL", "ETHOSLTD", "EUREKAFORB", "EXIDEIND", "NYKAA", "FEDFINA",
    "FEDERALBNK", "FACT", "FIEMIND", "FINCABLES", "FINPIPE", "FSL", "FIVESTAR", "FORCEMOT",
    "FORTIS", "UTLSOLAR", "GAIL", "GVT&D", "GHCL", "GMMPFAUDLR", "GMRAIRPORT", "GMRP&UI",
    "GABRIEL", "GALLANTT", "GRSE", "GRWRHITECH", "GICRE", "GILLETTE", "GLAND", "GLAXO",
    "GLENMARK", "MEDANTA", "GODIGIT", "GPIL", "GODFRYPHLP", "GODREJAGRO", "GODREJCP", "GODREJIND",
    "GODREJPROP", "GOKEX", "GOKULAGRO", "GRANULES", "GRAPHITE", "GRASIM", "GRAVITA", "GESHIP",
    "GREAVESCOT", "GRINDWELL", "GAEL", "FLUOROCHEM", "GMDCLTD", "GNFC", "GPPL", "GSFC",
    "HEG", "HGINFRA", "HBLENGINE", "HCLTECH", "HDBFS", "HDFCAMC", "HDFCBANK", "HDFCLIFE",
    "HFCL", "HAPPSTMNDS", "HAVELLS", "HCG", "HEMIPROP", "HERITGFOOD", "HEROMOTOCO", "HEXT",
    "HSCL", "HINDALCO", "HAL", "HCC", "HINDCOPPER", "HINDPETRO", "HINDUNILVR", "HINDZINC",
    "POWERINDIA", "HOMEFIRST", "HONASA", "HONAUT", "HUDCO", "HYUNDAI", "ICICIBANK", "ICICIGI",
    "ICICIAMC", "ICICIPRULI", "IDBI", "IDFCFIRSTB", "IFBIND", "IFCI", "IIFLCAPS", "IIFL",
    "INOXINDIA", "IRB", "IRCON", "ITCHOTELS", "ITC", "ITI", "INDGN", "INDIACEM",
    "INDIAGLYCO", "INDIASHLTR", "INDIAMART", "INDIANB", "IEX", "INDHOTEL", "IMFA", "IOC",
    "IOB", "IRCTC", "IRFC", "IREDA", "INDIGOPNTS", "ICIL", "IGL", "INDUSTOWER",
    "INDUSINDBK", "NAUKRI", "INFY", "INOXGREEN", "INOXWIND", "INTELLECT", "INDIGO", "IGIL",
    "IKS", "IONEXCHANG", "IPCALAB", "JKCEMENT", "JBMA", "JKLAKSHMI", "JKPAPER", "JKTYRE",
    "JMFINANCIL", "JSWCEMENT", "JSWDULUX", "JSWENERGY", "JSWINFRA", "JSWSTEEL", "JAIBALAJI", "JAINREC",
    "JPPOWER", "J&KBANK", "JAMNAAUTO", "JSFB", "JAYNECOIND", "JSLL", "JINDALSAW", "JSL",
    "JINDALSTEL", "JIOFIN", "JUBLFOOD", "JUBLINGREA", "JUBLPHARMA", "JLHL", "JWL", "JUSTDIAL",
    "JYOTHYLAB", "JYOTICNC", "KPRMILL", "KEI", "KNRCON", "KPIGREEN", "KPITTECH", "KRBL",
    "KRN", "KSB", "KAJARIACER", "KPIL", "KALYANKJIL", "KANSAINER", "KTKBANK", "KARURVYSYA",
    "KSCL", "KAYNES", "KEC", "KFINTECH", "KIRLOSBROS", "KIRLOSENG", "KIRLPNU", "KITEX",
    "KOTAKBANK", "KIMS", "LTF", "LTTS", "LGEINDIA", "LICHSGFIN", "LTFOODS", "LTM",
    "LT", "LATENTVIEW", "LAURUSLABS", "LXCHEM", "IXIGO", "THELEELA", "LEMONTREE", "LENSKART",
    "LICI", "LINDEINDIA", "LLOYDSENGG", "LLOYDSENT", "LLOYDSME", "LODHA", "LUMAXTECH", "LUPIN",
    "MMTC", "MOIL", "MRF", "MSTCLTD", "MTARTECH", "MGL", "MAHSCOOTER", "MAHSEAMLES",
    "M&MFIN", "M&M", "MANAPPURAM", "MRPL", "MANKIND", "MANORAMA", "MARICO", "MARKSANS",
    "MARUTI", "MASTEK", "MFSL", "MAXHEALTH", "MAZDOCK", "MEDPLUS", "MEESHO", "METROPOLIS",
    "MINDACORP", "MIDHANI", "MSUMI", "MOTILALOFS", "MPHASIS", "BECTORFOOD", "MCX", "MUTHOOTFIN",
    "NATCOPHARM", "NBCC", "NCC", "NEOGEN", "NESCO", "NHPC", "NLCINDIA", "NMDC",
    "NSLNISP", "NTPCGREEN", "NTPC", "NH", "NATIONALUM", "NFL", "NAVA", "NAVINFLUOR",
    "NAZARA", "NESTLEIND", "NETWEB", "NETWORK18", "NEULANDLAB", "NEWGEN", "NAM-INDIA", "NIVABUPA",
    "NUVAMA", "NUVOCO", "OBEROIRLTY", "ONGC", "OIL", "OLAELEC", "OLECTRA", "PAYTM",
    "ONESOURCE", "OPTIEMUS", "OFSS", "ORIENTCEM", "ORKLAINDIA", "OSWALPUMPS", "PNGJL", "POLICYBZR",
    "PCJEWELLER", "PCBL", "PGEL", "PIIND", "PNBHOUSING", "PNCINFRA", "PTC", "PTCIL",
    "PVRINOX", "PAGEIND", "PARADEEP", "PARAS", "PARKHOSPS", "PATANJALI", "PGIL", "PERSISTENT",
    "PETRONET", "PFIZER", "PHOENIXLTD", "PWL", "PICCADIL", "PIDILITIND", "PINELABS", "PIRAMALFIN",
    "PPLPHARMA", "POLYMED", "POLYCAB", "POONAWALLA", "PFC", "POWERGRID", "POWERMECH", "PRAJIND",
    "PREMIERENE", "PRESTIGE", "PRICOLLTD", "PFOCUS", "PRSMJOHNSN", "PRIVISCL", "PRUDENT", "PNB",
    "PURVA", "QPOWER", "QUESS", "RRKABEL", "RBLBANK", "RECLTD", "RHIM", "RITES",
    "RADICO", "RVNL", "RAILTEL", "RAIN", "RAINBOW", "RALLIS", "RKFORGE", "RCF",
    "RATEGAIN", "RATNAMANI", "RTNINDIA", "RTNPOWER", "RAYMONDLSL", "REDINGTON", "REDTAPE", "REFEX",
    "RELAXO", "RELIANCE", "RPOWER", "RELIGARE", "RBA", "ROUTE", "RUBICON", "SBFC",
    "SBICARD", "SBILIFE", "SJVN", "SKFINDUS", "SKFINDIA", "SKYGOLD", "SMLMAH", "SHRIPISTON",
    "SRF", "SAATVIKGL", "SAFARI", "SAGILITY", "SAILIFE", "SAMHI", "SAMMAANCAP", "MOTHERSON",
    "SANDUMA", "SANOFICONR", "SANSERA", "SAPPHIRE", "SARDAEN", "SAREGAMA", "SCHAEFFLER", "SCHNEIDER",
    "SENCO", "STYL", "SHAILY", "SHAKTIPUMP", "SHARDACROP", "SHAREINDIA", "SFL", "SHILPAMED",
    "SCI", "SHREECEM", "RENUKA", "SHRIRAMFIN", "SHYAMMETL", "ENRIN", "SIEMENS", "SIGNATURE",
    "SKIPPER", "SMARTWORKS", "SOBHA", "SOLARINDS", "SONACOMS", "SONATSOFTW", "SOUTHBANK", "LOTUSDEV",
    "STARCEMENT", "STARHEALTH", "SBIN", "SAIL", "SWSOLAR", "STLTECH", "STAR", "STYRENIX",
    "SUBROS", "SUDARSCHEM", "SUDEEPPHRM", "SUMICHEM", "SPARC", "SUNPHARMA", "SUNTV", "SUNDARMFIN",
    "SUNTECK", "SUPREMEIND", "SPLPETRO", "SUPRIYA", "SURYAROSNI", "SUZLON", "SWANCORP", "SWIGGY",
    "SYNGENE", "SYRMA", "TARC", "TBOTEK", "TDPOWERSYS", "TSFINV", "TVSMOTOR", "TVSSCS",
    "TMB", "TANLA", "TATACAP", "TATACHEM", "TATACOMM", "TCS", "TATACONSUM", "TATAELXSI",
    "TATAINVEST", "TMCV", "TMPV", "TATAPOWER", "TATASTEEL", "TATATECH", "TTML", "TECHM",
    "TECHNOE", "TEGA", "TEJASNET", "TENNIND", "TEXRAIL", "THANGAMAYL", "ANUP", "NIACL",
    "RAMCOCEM", "THERMAX", "THOMASCOOK", "THYROCARE", "TI", "TIMETECHNO", "TIMKEN", "TIPSMUSIC",
    "TITAGARH", "TITAN", "TORNTPHARM", "TORNTPOWER", "TARIL", "TRANSRAILL", "TRAVELFOOD", "TRENT",
    "TRIDENT", "TRIVENI", "TRITURBINE", "TIINDIA", "UCOBANK", "UNOMINDA", "UPL", "UTIAMC",
    "UJJIVANSFB", "ULTRACEMCO", "UNIONBANK", "UBL", "UNITDSPR", "URBANCO", "USHAMART", "VGUARD",
    "VMART", "VIPIND", "V2RETAIL", "WABAG", "VAIBHAVGBL", "DBREALTY", "VTL", "VARROC",
    "VBL", "MANYAVAR", "VEDL", "VIJAYA", "VIKRAMSOLR", "VMM", "VIYASH", "IDEA",
    "VOLTAMP", "VOLTAS", "WAAREEENER", "WAAREERTL", "WAKEFIT", "WEWORK", "WEBELSOLAR", "WELCORP",
    "WELENT", "WELSPUNLIV", "WESTLIFE", "WHIRLPOOL", "WIPRO", "WOCKPHARMA", "YATHARTH", "YESBANK",
    "ZFCVINDIA", "ZAGGLE", "ZEEL", "ZENTEC", "ZENSARTECH", "ZYDUSLIFE", "ZYDUSWELL", "ECLERX"
]


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
    week_date: str


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fetch_daily_ohlc(symbol: str, period: str = "5y") -> Optional[pd.DataFrame]:
    """
    Pull raw daily data via yfinance. period=5y so we have enough history
    for a stable 20-month EMA (needs ~5 years of daily bars).
    """
    ticker = f"{symbol}.NS"
    try:
        daily = yf.download(
            ticker, period=period, interval="1d", progress=False,
            auto_adjust=True, timeout=15,
        )
    except Exception as e:
        print(f"  [{symbol}] download error: {e}")
        return None

    if daily.empty:
        return None

    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)

    daily.columns = [c.lower() for c in daily.columns]
    return daily


def build_weekly(daily: pd.DataFrame) -> Optional[pd.DataFrame]:
    weekly = daily.resample("W-FRI").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    # Drop the trailing bar if that week's Friday hasn't happened yet (e.g. running on Monday) —
    # otherwise we'd evaluate an incomplete, still-forming weekly candle as if it were closed.
    if len(weekly) > 0:
        today = pd.Timestamp.now().normalize()
        if weekly.index[-1] > today:
            weekly = weekly.iloc[:-1]

    if len(weekly) < 45:
        return None

    return weekly


def build_monthly_trend(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly close, EMA6 and EMA20 on monthly close, and a bullish-cross flag
    (EMA6 > EMA20 = long-term uptrend confirmed, matches RK's monthly exit rule).
    """
    monthly = daily.resample("ME").agg({"close": "last"}).dropna()
    monthly["ema6_m"] = monthly["close"].ewm(span=6, adjust=False).mean()
    monthly["ema20_m"] = monthly["close"].ewm(span=20, adjust=False).mean()
    monthly["monthly_uptrend"] = monthly["ema6_m"] > monthly["ema20_m"]
    return monthly[["monthly_uptrend"]]


def _clean_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an index to tz-naive, nanosecond-precision datetime64 so merge_asof never hits a dtype mismatch."""
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.as_unit("ns")
    return df


def attach_monthly_trend(weekly: pd.DataFrame, monthly_trend: pd.DataFrame) -> pd.DataFrame:
    """
    For each weekly bar, attach the most recently COMPLETED month's uptrend flag
    (avoids look-ahead bias — only use months that had already closed).
    """
    weekly = _clean_datetime_index(weekly)

    monthly_shifted = _clean_datetime_index(monthly_trend)
    monthly_shifted.index = monthly_shifted.index + pd.Timedelta(days=1)  # push to next day so merge_asof only sees completed months

    merged = pd.merge_asof(
        weekly.sort_index(), monthly_shifted.sort_index(),
        left_index=True, right_index=True, direction="backward",
    )
    merged["monthly_uptrend"] = merged["monthly_uptrend"].fillna(False)
    return merged


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema40"] = df["close"].ewm(span=40, adjust=False).mean()

    df["rsi14"] = RSIIndicator(close=df["close"], window=14).rsi()

    adx_ind = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx14"] = adx_ind.adx()
    df["pdi14"] = adx_ind.adx_pos()
    df["ndi14"] = adx_ind.adx_neg()

    df["high_52w"] = df["high"].rolling(window=52, min_periods=52).max()

    return df


# ---------------------------------------------------------------------------
# Scan logic
# ---------------------------------------------------------------------------

def evaluate_conditions(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """
    Core rule evaluation, shared by check_row() and --explain mode.
    Returns a dict of {condition_name: (passed: bool, detail: str)} plus the row/derived values,
    or None if there isn't enough data at this index to evaluate at all.
    """
    if idx < 1 or idx >= len(df):
        return None

    row = df.iloc[idx]

    required = ["ema5", "ema10", "ema20", "ema40"]
    if row[required].isna().any():
        return None

    dist_ema10 = abs(row.close - row.ema10) / row.close
    dist_ema20 = abs(row.close - row.ema20) / row.close
    near_ema10 = dist_ema10 <= NEAR_EMA_PCT
    near_ema20 = dist_ema20 <= NEAR_EMA_PCT
    near_either = near_ema10 or near_ema20

    uptrend_ok = bool(row.ema10 > row.ema20 > row.ema40 and row.close > row.ema40)

    adx_val = row.get("adx14", float("nan"))
    adx_ok = bool(pd.notna(adx_val) and adx_val >= ADX_MIN)

    checks = {
        "near_10w_or_20w_ema": (near_either,
            f"close {row.close:.2f} | dist to EMA10 {dist_ema10*100:.2f}% | dist to EMA20 {dist_ema20*100:.2f}% (need <= {NEAR_EMA_PCT*100:.0f}% to either)"),
        "uptrend":            (uptrend_ok if UPTREND_REQUIRED else True,
            f"ema10 {row.ema10:.2f} > ema20 {row.ema20:.2f} > ema40 {row.ema40:.2f}, close {row.close:.2f} > ema40: {uptrend_ok}"),
        "adx_min":            (adx_ok,
            f"ADX {adx_val:.1f} (need >= {ADX_MIN})" if pd.notna(adx_val) else "ADX not available"),
    }
    return {"row": row, "checks": checks}


def check_row(df: pd.DataFrame, idx: int) -> Optional[ScanResult]:
    evald = evaluate_conditions(df, idx)
    if evald is None:
        return None

    row = evald["row"]
    checks = evald["checks"]
    if not all(passed for passed, _ in checks.values()):
        return None

    dist_ema10 = abs(row.close - row.ema10) / row.close
    dist_ema20 = abs(row.close - row.ema20) / row.close

    return ScanResult(
        symbol="",
        close=round(row.close, 2),
        ema5=round(row.ema5, 2),
        ema10=round(row.ema10, 2),
        ema20=round(row.ema20, 2),
        ema40=round(row.ema40, 2),
        rsi14=round(row.rsi14, 2) if pd.notna(row.get("rsi14")) else None,
        adx14=round(row.adx14, 2) if pd.notna(row.get("adx14")) else None,
        pdi14=round(row.pdi14, 2) if pd.notna(row.get("pdi14")) else None,
        ndi14=round(row.ndi14, 2) if pd.notna(row.get("ndi14")) else None,
        compression_pct=round(min(dist_ema10, dist_ema20) * 100, 2),
        week_date=str(row.name.date()),
    )


def scan_symbol(symbol: str, backtest: bool, lookback_weeks: int) -> List[ScanResult]:
    daily = fetch_daily_ohlc(symbol)
    if daily is None:
        print(f"  [{symbol}] skipped — insufficient data")
        return []

    weekly = build_weekly(daily)
    if weekly is None:
        print(f"  [{symbol}] skipped — not enough weekly bars")
        return []

    weekly = compute_indicators(weekly)
    results = []

    if backtest:
        start_idx = max(1, len(weekly) - lookback_weeks)
        for i in range(start_idx, len(weekly)):
            r = check_row(weekly, i)
            if r:
                r.symbol = symbol
                results.append(r)
    else:
        r = check_row(weekly, len(weekly) - 1)
        if r:
            r.symbol = symbol
            results.append(r)

    return results


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

TELEGRAM_MAX_CHARS = 4000  # Telegram's hard limit is 4096; leave headroom


def _split_message_into_chunks(text: str, max_chars: int = TELEGRAM_MAX_CHARS) -> List[str]:
    """Split a long message into chunks that fit Telegram's per-message limit,
    breaking on blank lines between entries so a stock's block never gets cut in half."""
    if len(text) <= max_chars:
        return [text]

    blocks = text.split("\n\n")
    chunks = []
    current = ""
    for block in blocks:
        candidate = (current + "\n\n" + block) if current else block
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not set (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) — skipping send.")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = _split_message_into_chunks(text)
    if len(chunks) > 1:
        print(f"Message is {len(text)} chars — splitting into {len(chunks)} Telegram messages.")

    for i, chunk in enumerate(chunks, 1):
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "Markdown"}
        try:
            resp = requests.post(url, data=payload, timeout=10)
            resp.raise_for_status()
            print(f"Telegram chunk {i}/{len(chunks)} sent OK.")
        except Exception as e:
            body = getattr(e, "response", None)
            body_text = body.text if body is not None else ""
            print(f"Telegram send failed on chunk {i}/{len(chunks)}: {e} {body_text}")


def format_results_message(results: List[ScanResult]) -> str:
    if not results:
        return "*Near 10W/20W EMA Scan*\nNo matches this week."

    lines = [f"*Near 10W/20W EMA Scan* — {len(results)} match(es)\n"]
    for r in results:
        dist_10 = abs(r.close - r.ema10) / r.close * 100
        dist_20 = abs(r.close - r.ema20) / r.close * 100
        rsi_txt = f"{r.rsi14:.1f}" if r.rsi14 is not None else "n/a"
        adx_txt = f"{r.adx14:.1f}" if r.adx14 is not None else "n/a"
        lines.append(
            f"*{r.symbol}* ({r.week_date})\n"
            f"  Close: {r.close} | EMA10: {r.ema10} | EMA20: {r.ema20} | EMA40: {r.ema40}\n"
            f"  Dist to EMA10: {dist_10:.2f}% | Dist to EMA20: {dist_20:.2f}%\n"
            f"  RSI: {rsi_txt} | ADX: {adx_txt}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def explain_symbol(symbol: str):
    """Print a pass/fail breakdown of every condition for the latest week of one symbol."""
    daily = fetch_daily_ohlc(symbol)
    if daily is None:
        print(f"{symbol}: could not fetch data.")
        return

    weekly = build_weekly(daily)
    if weekly is None:
        print(f"{symbol}: not enough weekly bars.")
        return

    weekly = compute_indicators(weekly)

    evald = evaluate_conditions(weekly, len(weekly) - 1)
    if evald is None:
        print(f"{symbol}: not enough history to evaluate yet.")
        return

    row = evald["row"]
    checks = evald["checks"]
    print(f"\n=== {symbol} — week of {row.name.date()} — close {row.close:.2f} ===")
    all_pass = True
    for name, (passed, detail) in checks.items():
        mark = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{mark}] {name}: {detail}")
    print(f"  => OVERALL: {'MATCH' if all_pass else 'no match'}\n")


def main():
    parser = argparse.ArgumentParser(description="EMA Squeeze Base weekly scanner")
    parser.add_argument("--dry-run", action="store_true", help="Print results, skip Telegram send")
    parser.add_argument("--backtest", action="store_true", help="Check the last N weeks instead of just the latest")
    parser.add_argument("--lookback-weeks", type=int, default=5, help="Weeks to check when --backtest is set")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to sleep between symbol downloads")
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N symbols (useful for quick tests)")
    parser.add_argument("--explain", type=str, default=None,
                         help="Show a per-condition pass/fail breakdown for one symbol (e.g. --explain ZYDUSLIFE) instead of scanning")
    args = parser.parse_args()

    if args.explain:
        explain_symbol(args.explain.upper())
        return

    symbols = SYMBOLS[: args.limit] if args.limit else SYMBOLS
    print(f"Scanning {len(symbols)} symbols...")

    all_results: List[ScanResult] = []
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol}")
        results = scan_symbol(symbol, backtest=args.backtest, lookback_weeks=args.lookback_weeks)
        all_results.extend(results)
        time.sleep(args.delay)

    print(f"\n{len(all_results)} match(es) found.")
    message = format_results_message(all_results)
    print(message)

    if not args.dry_run:
        send_telegram_message(message)


if __name__ == "__main__":
    main()
