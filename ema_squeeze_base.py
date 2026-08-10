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

EMA_COMPRESSION_THRESHOLD = 0.04      # 4% max spread among 5W/10W/20W EMAs
PRICE_ABOVE_EMA10_MAX_PCT = 0.05      # close within 5% above the 10W EMA
RSI_LOW, RSI_HIGH = 48, 58
ADX_MIN = 20
NEAR_52W_HIGH_PCT = 0.85               # close must be within 15% of the 52-week high (tightened)
EMA40_TREND_LOOKBACK = 16              # weeks back to confirm ema40 is rising (was 8 — too short)
EMA40_MIN_RISE_PCT = 0.03              # ema40 must have risen at least 3% over that lookback
STACK_CONSISTENCY_WEEKS = 4            # bullish EMA order must hold for this many consecutive weeks
PRIOR_RALLY_LOOKBACK = 26              # weeks to look back for the pre-base rally
PRIOR_RALLY_MIN_GAIN = 0.20            # close must be >=20% above the lowest close in that lookback

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
    if idx < max(EMA40_TREND_LOOKBACK, PRIOR_RALLY_LOOKBACK, STACK_CONSISTENCY_WEEKS):
        return None

    row = df.iloc[idx]
    prev = df.iloc[idx - 1]
    ema40_then = df.iloc[idx - EMA40_TREND_LOOKBACK]["ema40"]

    required = ["ema5", "ema10", "ema20", "ema40", "rsi14", "adx14", "pdi14", "ndi14", "high_52w"]
    if row[required].isna().any() or pd.isna(prev["adx14"]) or pd.isna(ema40_then):
        return None

    ema_cluster = [row.ema5, row.ema10, row.ema20]
    compression = (max(ema_cluster) - min(ema_cluster)) / row.close

    stack_window = df.iloc[idx - STACK_CONSISTENCY_WEEKS + 1: idx + 1]
    stack_ok = bool(((stack_window["ema10"] > stack_window["ema20"]) &
                      (stack_window["ema20"] > stack_window["ema40"])).all())

    rally_window = df.iloc[idx - PRIOR_RALLY_LOOKBACK: idx]
    prior_low = rally_window["close"].min()
    prior_rally_ok = bool(pd.notna(prior_low) and row.close >= prior_low * (1 + PRIOR_RALLY_MIN_GAIN))
    prior_rally_gain = (row.close / prior_low - 1) * 100 if pd.notna(prior_low) else float("nan")

    monthly_uptrend_ok = bool(row.get("monthly_uptrend", False))

    checks = {
        "ema_compression":   (compression < EMA_COMPRESSION_THRESHOLD,
                               f"{compression*100:.2f}% (need < {EMA_COMPRESSION_THRESHOLD*100:.0f}%)"),
        "price_near_ema10":  (row.ema10 <= row.close <= row.ema10 * (1 + PRICE_ABOVE_EMA10_MAX_PCT),
                               f"close {row.close:.2f} vs ema10 {row.ema10:.2f} (need within +{PRICE_ABOVE_EMA10_MAX_PCT*100:.0f}%, not below)"),
        "price_above_ema40": (row.close > row.ema40,
                               f"close {row.close:.2f} vs ema40 {row.ema40:.2f}"),
        "rsi_band":          (RSI_LOW <= row.rsi14 <= RSI_HIGH,
                               f"RSI {row.rsi14:.1f} (need {RSI_LOW}-{RSI_HIGH})"),
        "adx_rising_above20": (row.adx14 > ADX_MIN and row.adx14 > prev.adx14,
                               f"ADX {row.adx14:.1f} (prev {prev.adx14:.1f}, need >{ADX_MIN} and rising)"),
        "pdi_above_ndi":     (row.pdi14 > row.ndi14,
                               f"+DI {row.pdi14:.1f} vs -DI {row.ndi14:.1f}"),
        "near_52w_high":     (row.close >= NEAR_52W_HIGH_PCT * row.high_52w,
                               f"close {row.close:.2f} is {(row.close/row.high_52w)*100:.1f}% of 52W high {row.high_52w:.2f} (need >= {NEAR_52W_HIGH_PCT*100:.0f}%)"),
        "ema40_rising":      (row.ema40 > ema40_then * (1 + EMA40_MIN_RISE_PCT),
                               f"ema40 {row.ema40:.2f} vs {EMA40_TREND_LOOKBACK}w-ago {ema40_then:.2f} (need +{EMA40_MIN_RISE_PCT*100:.0f}%+)"),
        "cluster_above_ema40": (min(ema_cluster) > row.ema40,
                               f"min(5/10/20 EMA) {min(ema_cluster):.2f} vs ema40 {row.ema40:.2f}"),
        "stack_sustained":   (stack_ok,
                               f"ema10>ema20>ema40 held for last {STACK_CONSISTENCY_WEEKS} weeks: {stack_ok}"),
        "prior_rally":       (prior_rally_ok,
                               f"+{prior_rally_gain:.1f}% from {PRIOR_RALLY_LOOKBACK}w low (need >= +{PRIOR_RALLY_MIN_GAIN*100:.0f}%)"),
        "monthly_uptrend":   (monthly_uptrend_ok,
                               f"monthly EMA6>EMA20: {monthly_uptrend_ok}"),
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

    ema_cluster = [row.ema5, row.ema10, row.ema20]
    compression = (max(ema_cluster) - min(ema_cluster)) / row.close

    return ScanResult(
        symbol="",
        close=round(row.close, 2),
        ema5=round(row.ema5, 2),
        ema10=round(row.ema10, 2),
        ema20=round(row.ema20, 2),
        ema40=round(row.ema40, 2),
        rsi14=round(row.rsi14, 2),
        adx14=round(row.adx14, 2),
        pdi14=round(row.pdi14, 2),
        ndi14=round(row.ndi14, 2),
        compression_pct=round(compression * 100, 2),
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

    monthly_trend = build_monthly_trend(daily)
    weekly = attach_monthly_trend(weekly, monthly_trend)
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

def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not set (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) — skipping send.")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, data=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"Telegram send failed: {e}")


def format_results_message(results: List[ScanResult]) -> str:
    if not results:
        return "*EMA Squeeze Base Scan*\nNo matches this week."

    lines = [f"*EMA Squeeze Base Scan* — {len(results)} match(es)\n"]
    for r in results:
        lines.append(
            f"*{r.symbol}* ({r.week_date})\n"
            f"  Close: {r.close} | EMA10: {r.ema10} | EMA40: {r.ema40}\n"
            f"  RSI: {r.rsi14} | ADX: {r.adx14} (+DI {r.pdi14} / -DI {r.ndi14})\n"
            f"  EMA compression: {r.compression_pct}%\n"
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

    monthly_trend = build_monthly_trend(daily)
    weekly = attach_monthly_trend(weekly, monthly_trend)
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
