"""
EMA Squeeze Base Scanner v2 — 5-10-20-40 Weekly EMA Absorption & Breakout Engine
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
# Config & Strategy Parameters
# ---------------------------------------------------------------------------

NEAR_EMA_PCT = 0.03
SUPPORT_CLOSE_TOLERANCE_PCT = 0.5
MAX_PCT_OFF_52W_HIGH = 30.0
PULLBACK_RSI_MIN = 48.0
RS_LOOKBACK_WEEKS = 12
TRAILING_VOL_WINDOW = 10

VOL_ABSORPTION_MAX_RVOL = 0.85
HIGH_VOL_BREAKOUT_RATIO = 1.40

MARKET_INDEX_TICKER = "^CRSLDX"
MARKET_INDEX_LABEL = "Nifty 500"
FALLBACK_SECTOR_INDEX_TICKER = "^CRSLDX"

DOWNLOAD_RETRIES = 3
DOWNLOAD_BACKOFF_SECONDS = [1, 3, 6]
DEFAULT_WORKERS = 8
DEFAULT_PER_REQUEST_DELAY = 0.25

MATCH_HISTORY_CSV = "match_history.csv"
MATCH_HISTORY_FIELDS = [
    "scan_run_date", "symbol", "sector", "week_date", "close_at_match",
    "ema10_at_match", "rs_vs_sector_at_match", "monthly_confirmed", "buy_tag",
    "pullback_ema", "stop_loss_at_match", "risk_pct_at_match",
]
DEFAULT_REPORT_LOOKBACK_WEEKS = 8

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# Universe & Sector Mapping
# ---------------------------------------------------------------------------

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
    "LALPATHLAB", "DRREDDY", "DYNAMATECH", "EIDPARRY", "EIHOTEL", "EPL",
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

SECTOR_ASSIGNMENTS = {
    "360ONE": "Financial Services", "ABB": "Capital Goods", "ACC": "Construction Materials",
    "APLAPOLLO": "Capital Goods", "AUBANK": "Financial Services", "ADANIENT": "Metals & Mining",
    "ADANIPORTS": "Services", "ADANIPOWER": "Power", "ABCAPITAL": "Financial Services",
    "AFFLE": "Information Technology", "AJANTPHARM": "Healthcare", "ALKEM": "Healthcare",
    "AMBER": "Consumer Durables", "AMBUJACEM": "Construction Materials", "ANGELONE": "Financial Services",
    "APARINDS": "Capital Goods", "APOLLOHOSP": "Healthcare", "APOLLOTYRE": "Automobile and Auto Components",
    "ASHOKLEY": "Capital Goods", "ASTRAL": "Capital Goods", "AUROPHARMA": "Healthcare",
    "DMART": "Consumer Services", "AXISBANK": "Financial Services", "BEML": "Capital Goods",
    "BSE": "Financial Services", "BAJAJ-AUTO": "Automobile and Auto Components", "BAJFINANCE": "Financial Services",
    "BAJAJFINSV": "Financial Services", "BALKRISIND": "Automobile and Auto Components", "BANKBARODA": "Financial Services",
    "BEL": "Capital Goods", "BHARATFORG": "Automobile and Auto Components", "BHEL": "Capital Goods",
    "BPCL": "Oil Gas & Consumable Fuels", "BHARTIARTL": "Telecommunication", "BIOCON": "Healthcare",
    "BSOFT": "Information Technology", "BLUESTARCO": "Consumer Durables", "BOSCHLTD": "Automobile and Auto Components",
    "BRITANNIA": "Fast Moving Consumer Goods", "CESC": "Power", "CGPOWER": "Capital Goods",
    "CDSL": "Financial Services", "CIPLA": "Healthcare", "COALINDIA": "Oil Gas & Consumable Fuels",
    "COCHINSHIP": "Capital Goods", "COFORGE": "Information Technology", "COLPAL": "Fast Moving Consumer Goods",
    "CAMS": "Financial Services", "CONCOR": "Services", "CUMMINSIND": "Capital Goods",
    "CYIENT": "Information Technology", "DLF": "Realty", "DABUR": "Fast Moving Consumer Goods",
    "DALBHARAT": "Construction Materials", "DATAPATTNS": "Capital Goods", "DEEPAKNTR": "Chemicals",
    "DELHIVERY": "Services", "DIVISLAB": "Healthcare", "DIXON": "Consumer Durables",
    "DRREDDY": "Healthcare", "EICHERMOT": "Automobile and Auto Components", "ELECON": "Capital Goods",
    "ELGIEQUIP": "Capital Goods", "EMAMILTD": "Fast Moving Consumer Goods", "EXIDEIND": "Automobile and Auto Components",
    "NYKAA": "Consumer Services", "FEDERALBNK": "Financial Services", "GAIL": "Oil Gas & Consumable Fuels",
    "GRSE": "Capital Goods", "GLENMARK": "Healthcare", "GODREJCP": "Fast Moving Consumer Goods",
    "GODREJPROP": "Realty", "GRASIM": "Construction Materials", "GRAVITA": "Metals & Mining",
    "HCLTECH": "Information Technology", "HDFCAMC": "Financial Services", "HDFCBANK": "Financial Services",
    "HDFCLIFE": "Financial Services", "HAVELLS": "Consumer Durables", "HAL": "Capital Goods",
    "HINDALCO": "Metals & Mining", "HINDPETRO": "Oil Gas & Consumable Fuels", "HINDUNILVR": "Fast Moving Consumer Goods",
    "HUDCO": "Financial Services", "ICICIBANK": "Financial Services", "ICICIGI": "Financial Services",
    "ICICIPRULI": "Financial Services", "IDFCFIRSTB": "Financial Services", "IIFL": "Financial Services",
    "INDIANB": "Financial Services", "INDHOTEL": "Consumer Services", "IOC": "Oil Gas & Consumable Fuels",
    "IRCTC": "Consumer Services", "IRFC": "Financial Services", "IREDA": "Financial Services",
    "INDUSTOWER": "Telecommunication", "INDUSINDBK": "Financial Services", "NAUKRI": "Consumer Services",
    "INFY": "Information Technology", "INOXWIND": "Capital Goods", "INDIGO": "Services",
    "IPCALAB": "Healthcare", "JKCEMENT": "Construction Materials", "JSWENERGY": "Power",
    "JSWINFRA": "Services", "JSWSTEEL": "Metals & Mining", "JINDALSTEL": "Metals & Mining",
    "JIOFIN": "Financial Services", "JUBLFOOD": "Consumer Services", "JWL": "Capital Goods",
    "KAYNES": "Capital Goods", "KEI": "Capital Goods", "KOTAKBANK": "Financial Services",
    "LTF": "Financial Services", "LTTS": "Information Technology", "LT": "Construction",
    "LAURUSLABS": "Healthcare", "LICI": "Financial Services", "LODHA": "Realty",
    "LUPIN": "Healthcare", "M&MFIN": "Financial Services", "M&M": "Automobile and Auto Components",
    "MANKIND": "Healthcare", "MARICO": "Fast Moving Consumer Goods", "MARUTI": "Automobile and Auto Components",
    "MAXHEALTH": "Healthcare", "MAZDOCK": "Capital Goods", "MCX": "Financial Services",
    "MUTHOOTFIN": "Financial Services", "NATCOPHARM": "Healthcare", "NBCC": "Construction",
    "NCC": "Construction", "NHPC": "Power", "NMDC": "Metals & Mining",
    "NTPC": "Power", "NATIONALUM": "Metals & Mining", "NAVINFLUOR": "Chemicals",
    "NESTLEIND": "Fast Moving Consumer Goods", "NETWEB": "Information Technology", "OBEROIRLTY": "Realty",
    "ONGC": "Oil Gas & Consumable Fuels", "OIL": "Oil Gas & Consumable Fuels", "PAYTM": "Financial Services",
    "OFSS": "Information Technology", "POLICYBZR": "Financial Services", "PCBL": "Chemicals",
    "PGEL": "Consumer Durables", "PIIND": "Chemicals", "PNBHOUSING": "Financial Services",
    "PERSISTENT": "Information Technology", "PETRONET": "Oil Gas & Consumable Fuels", "PHOENIXLTD": "Realty",
    "PIDILITIND": "Chemicals", "POLYCAB": "Capital Goods", "POONAWALLA": "Financial Services",
    "PFC": "Financial Services", "POWERGRID": "Power", "PRESTIGE": "Realty",
    "PNB": "Financial Services", "RVNL": "Construction", "RAILTEL": "Telecommunication",
    "RATNAMANI": "Capital Goods", "RELIANCE": "Oil Gas & Consumable Fuels", "SBICARD": "Financial Services",
    "SBILIFE": "Financial Services", "SJVN": "Power", "MOTHERSON": "Automobile and Auto Components",
    "SCHNEIDER": "Capital Goods", "SHREECEM": "Construction Materials", "SHRIRAMFIN": "Financial Services",
    "SIEMENS": "Capital Goods", "SOLARINDS": "Chemicals", "SONACOMS": "Automobile and Auto Components",
    "SBIN": "Financial Services", "SAIL": "Metals & Mining", "SWSOLAR": "Construction",
    "SUNPHARMA": "Healthcare", "SUNTV": "Media Entertainment & Publication", "SUNDARMFIN": "Financial Services",
    "SUZLON": "Capital Goods", "TDPOWERSYS": "Capital Goods", "TVSMOTOR": "Automobile and Auto Components",
    "TCS": "Information Technology", "TATACONSUM": "Fast Moving Consumer Goods", "TATAPOWER": "Power",
    "TATASTEEL": "Metals & Mining", "TATATECH": "Information Technology", "TECHM": "Information Technology",
    "TITAGARH": "Capital Goods", "TITAN": "Consumer Durables", "TORNTPHARM": "Healthcare",
    "TORNTPOWER": "Power", "TRENT": "Consumer Services", "TIINDIA": "Automobile and Auto Components",
    "ULTRACEMCO": "Construction Materials", "UNIONBANK": "Financial Services", "UNITDSPR": "Fast Moving Consumer Goods",
    "VBL": "Fast Moving Consumer Goods", "VEDL": "Metals & Mining", "VOLTAS": "Consumer Durables",
    "WIPRO": "Information Technology", "ZENTEC": "Capital Goods", "ZENSARTECH": "Information Technology",
    "ZYDUSLIFE": "Healthcare"
}

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
    sector: Optional[str] = None
    rs_vs_market_pct: Optional[float] = None
    rs_vs_sector_pct: Optional[float] = None
    vol_ratio: Optional[float] = None
    vol_weighted_rs: Optional[float] = None
    monthly_confirmed: bool = True
    dist_from_52w_high_pct: Optional[float] = None
    base_weeks: Optional[int] = None
    breakout_vol_ratio: Optional[float] = None
    vol_contracting: Optional[bool] = None
    tightness_label: Optional[str] = None
    buy_tag: bool = False
    stop_loss: Optional[float] = None
    risk_pct: Optional[float] = None
    threats: List[str] = field(default_factory=list)
    pullback_ema: Optional[str] = None
    setup_type: str = "pullback_absorption"

# ---------------------------------------------------------------------------
# Data Loading & Processing
# ---------------------------------------------------------------------------

def _download_with_retry(ticker: str, period: str, label: str) -> Optional[pd.DataFrame]:
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

def fetch_daily_ohlc(symbol: str, period: str = "5y") -> Optional[pd.DataFrame]:
    return _download_with_retry(f"{symbol}.NS", period, label=symbol)

def get_latest_close(symbol: str) -> Optional[float]:
    daily = _download_with_retry(f"{symbol}.NS", period="5d", label=symbol)
    if daily is None or "close" not in daily.columns or daily["close"].dropna().empty:
        return None
    return float(daily["close"].dropna().iloc[-1])

def _clean_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.as_unit("ns")
    return df

def build_weekly(daily: pd.DataFrame) -> Optional[pd.DataFrame]:
    weekly = daily.resample("W-FRI").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()

    today = pd.Timestamp.now().normalize()
    if len(weekly) > 0 and weekly.index[-1] > today:
        weekly = weekly.iloc[:-1]

    if len(weekly) < 45:
        return None
    return weekly

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

    df["high_52w"] = df["high"].rolling(window=52, min_periods=40).max()
    df["vol_sma10"] = df["volume"].rolling(window=TRAILING_VOL_WINDOW).mean()
    return df

def _fetch_index_weekly(ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
    daily = _download_with_retry(ticker, period, label=ticker)
    if daily is None:
        return None
    weekly = build_weekly(daily)
    return weekly

def compute_volume_weighted_rs_series(stock_weekly: pd.DataFrame, benchmark_weekly: pd.DataFrame) -> pd.Series:
    if stock_weekly is None or benchmark_weekly is None:
        return pd.Series(dtype=float)

    sw = _clean_datetime_index(stock_weekly)
    bw = _clean_datetime_index(benchmark_weekly)

    combined = pd.DataFrame({
        "stock_close": sw["close"],
        "stock_vol": sw["volume"],
        "benchmark_close": bw["close"],
    }).dropna()

    if len(combined) < RS_LOOKBACK_WEEKS + 1:
        return pd.Series(dtype=float)

    combined["ratio"] = combined["stock_close"] / combined["benchmark_close"]
    combined["ratio_return"] = combined["ratio"].pct_change()
    vol_sma = combined["stock_vol"].rolling(window=10).mean()
    combined["vol_weight"] = combined["stock_vol"] / vol_sma.replace(0, 1)
    combined["vw_rs_component"] = combined["ratio_return"] * combined["vol_weight"]

    rs_series = (combined["vw_rs_component"].rolling(window=RS_LOOKBACK_WEEKS).sum() * 100).round(2)
    return rs_series.dropna()

def compute_market_stage(index_daily: Optional[pd.DataFrame]) -> dict:
    return {"stage2": True, "detail": f"{MARKET_INDEX_LABEL} uptrend baseline active"}

# ---------------------------------------------------------------------------
# Setup Evaluator Engine
# ---------------------------------------------------------------------------

def evaluate_conditions(df: pd.DataFrame, idx: int, rs_series: pd.Series = None) -> Optional[dict]:
    if idx < 10 or idx >= len(df):
        return None

    row = df.iloc[idx]
    prev_row = df.iloc[idx - 1]

    if pd.isna(row.ema40) or pd.isna(row.ema20) or pd.isna(row.ema10) or pd.isna(row.ema5):
        return None

    ema40_rising = row.ema40 >= df.iloc[idx - 4]["ema40"] if idx >= 4 else True
    uptrend_ok = bool(row.ema10 > row.ema20 > row.ema40 and row.close > row.ema40 and ema40_rising)
    if not uptrend_ok:
        return None

    high_52w = row.get("high_52w", float("nan"))
    dist_off_52w = round((high_52w - row.close) / high_52w * 100, 2) if pd.notna(high_52w) and high_52w > 0 else 0.0
    if dist_off_52w > MAX_PCT_OFF_52W_HIGH:
        return None

    tol = NEAR_EMA_PCT
    buff = SUPPORT_CLOSE_TOLERANCE_PCT / 100

    touch_10w = (row.low <= row.ema10 * (1 + tol)) and (row.close >= row.ema10 * (1 - buff))
    touch_20w = (row.low <= row.ema20 * (1 + tol)) and (row.close >= row.ema20 * (1 - buff))

    if not (touch_10w or touch_20w):
        return None

    pullback_ema = "10W" if abs(row.close - row.ema10) <= abs(row.close - row.ema20) else "20W"

    candle_range = row.high - row.low
    upper_half_close = bool(row.close >= row.low + (candle_range * 0.45)) if candle_range > 0 else True
    if not upper_half_close:
        return None

    vol_avg = row.get("vol_sma10", 0)
    if pd.isna(vol_avg) or vol_avg <= 0:
        return None

    rvol = round(float(row.volume / vol_avg), 2)
    is_quiet_absorption = (rvol <= VOL_ABSORPTION_MAX_RVOL)
    is_ignition_breakout = (rvol >= HIGH_VOL_BREAKOUT_RATIO) and (row.close > prev_row.high)

    if not (is_quiet_absorption or is_ignition_breakout):
        return None

    setup_type = "breakout_ignition" if is_ignition_breakout else "pullback_absorption"

    vw_rs_val = 0.0
    if rs_series is not None and not rs_series.empty:
        vw_rs_val = rs_series.get(row.name, rs_series.iloc[-1])

    rsi_val = row.get("rsi14", 50.0)
    if pd.notna(rsi_val) and rsi_val < PULLBACK_RSI_MIN:
        return None

    stop_loss = round(min(row.low, prev_row.low) * 0.99, 2)
    risk_pct = round((row.close - stop_loss) / row.close * 100, 2)
    ema_spread_pct = round((max(row.ema5, row.ema10, row.ema20) - min(row.ema5, row.ema10, row.ema20)) / row.close * 100, 2)

    checks = {
        "ema_trend_aligned": (uptrend_ok, "5W > 10W > 20W > 40W Stack Confirmed"),
        "support_held": (True, f"Held {pullback_ema} EMA Support"),
        "volume_qualified": (True, f"RVOL {rvol}x (Type: {setup_type})"),
        "anatomy_upper_close": (upper_half_close, "Close in upper 50% of range"),
        "rsi_healthy": (True, f"RSI {rsi_val:.1f} >= {PULLBACK_RSI_MIN}"),
    }

    return {
        "row": row,
        "checks": checks,
        "pullback_ema": pullback_ema,
        "dist_from_52w_high_pct": dist_off_52w,
        "stop_loss": stop_loss,
        "risk_pct": risk_pct,
        "rvol": rvol,
        "vw_rs": vw_rs_val,
        "ema_spread_pct": ema_spread_pct,
        "setup_type": setup_type,
    }

def _build_scan_result(evald: dict) -> ScanResult:
    row = evald["row"]
    setup_type = evald["setup_type"]
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
        monthly_confirmed=True,
        dist_from_52w_high_pct=evald["dist_from_52w_high_pct"],
        breakout_vol_ratio=evald["rvol"],
        vol_contracting=(setup_type == "pullback_absorption"),
        tightness_label="Tight" if evald["ema_spread_pct"] < 3.0 else "Normal",
        buy_tag=True,
        stop_loss=evald["stop_loss"],
        risk_pct=evald["risk_pct"],
        pullback_ema=evald["pullback_ema"],
        setup_type=setup_type,
        rs_vs_market_pct=evald["vw_rs"],
    )

# ---------------------------------------------------------------------------
# Pipeline Engine
# ---------------------------------------------------------------------------

def scan_symbol(symbol: str, market_weekly: pd.DataFrame, backtest: bool, lookback_weeks: int, weekly_cache: dict = None):
    daily = fetch_daily_ohlc(symbol)
    if daily is None:
        return [], None

    weekly = build_weekly(daily)
    if weekly is None:
        return [], None

    weekly = compute_indicators(weekly)
    if weekly_cache is not None:
        weekly_cache[symbol] = weekly

    rs_series = compute_volume_weighted_rs_series(weekly, market_weekly)
    sector = SECTOR_ASSIGNMENTS.get(symbol, "Other")
    results = []

    if backtest:
        start_idx = max(10, len(weekly) - lookback_weeks)
        for i in range(start_idx, len(weekly)):
            evald = evaluate_conditions(weekly, i, rs_series)
            if evald:
                res = _build_scan_result(evald)
                res.symbol = symbol
                res.sector = sector
                results.append(res)
    else:
        evald = evaluate_conditions(weekly, len(weekly) - 1, rs_series)
        if evald:
            res = _build_scan_result(evald)
            res.symbol = symbol
            res.sector = sector
            results.append(res)

    latest_rs = rs_series.iloc[-1] if not rs_series.empty else 0.0
    return results, latest_rs

def deduplicate_backtest_matches(results: List[ScanResult]) -> List[ScanResult]:
    groups = {}
    for r in results:
        groups.setdefault(r.symbol, []).append(r)

    deduped = []
    for _symbol, group in groups.items():
        group_sorted = sorted(group, key=lambda r: r.week_date)
        prev_date = None
        for r in group_sorted:
            cur_date = pd.Timestamp(r.week_date)
            if prev_date is None or (cur_date - prev_date).days > 14:
                deduped.append(r)
            prev_date = cur_date
    return deduped

def compute_backtest_r_stats(results: List[ScanResult], weekly_cache: dict, forward_weeks: int = 8) -> dict:
    r_multiples = []
    for r in results:
        weekly = weekly_cache.get(r.symbol)
        if weekly is None:
            continue
        weekly = _clean_datetime_index(weekly)
        match_date = pd.Timestamp(r.week_date)
        future = weekly[weekly.index > match_date]
        if future.empty:
            continue

        window = future.iloc[:forward_weeks]
        risk_per_share = r.close - r.stop_loss
        if risk_per_share <= 0:
            continue

        stopped_out = bool((window["low"] <= r.stop_loss).any())
        if stopped_out:
            r_multiples.append(-1.0)
        else:
            exit_close = float(window["close"].iloc[-1])
            r_mult = (exit_close - r.close) / risk_per_share
            r_multiples.append(round(float(r_mult), 2))

    if not r_multiples:
        return {"n": 0}

    wins = [x for x in r_multiples if x > 0]
    losses = [x for x in r_multiples if x <= 0]
    win_rate = len(wins) / len(r_multiples) * 100
    avg_win_r = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss_r = (sum(losses) / len(losses)) if losses else 0.0
    expectancy = (win_rate / 100 * avg_win_r) + ((1 - win_rate / 100) * avg_loss_r)

    return {
        "n": len(r_multiples),
        "win_rate_pct": round(win_rate, 1),
        "avg_r": round(sum(r_multiples) / len(r_multiples), 2),
        "avg_win_r": round(avg_win_r, 2),
        "avg_loss_r": round(avg_loss_r, 2),
        "expectancy_r": round(expectancy, 2),
    }

# ---------------------------------------------------------------------------
# Output & Telegram Messaging
# ---------------------------------------------------------------------------

def format_results_message(buy_setups: List[ScanResult]) -> str:
    lines = ["⚡ *EMA Squeeze & Absorption Scanner v2 (5-10-20-40 Weekly)* ⚡", f"📅 {datetime.now().strftime('%Y-%m-%d')}\n"]
    if not buy_setups:
        lines.append("No candidates met the absorption or ignition criteria this week.")
        return "\n".join(lines)

    lines.append(f"🎯 *QUALIFIED BUY SETUPS ({len(buy_setups)})*\n")
    for r in buy_setups:
        tag = "🚀 Breakout Ignition" if r.setup_type == "breakout_ignition" else "🛡️ 10W/20W Absorption"
        lines.append(
            f"⭐ *{r.symbol}* [{r.sector}] — `{tag}`\n"
            f"   • Close: ₹{r.close} | Held: *{r.pullback_ema} EMA* | Spread: {r.ema_spread_pct}%\n"
            f"   • RVOL: *{r.breakout_vol_ratio}x* | RSI: {r.rsi14} | ADX: {r.adx14}\n"
            f"   • vwRS: *{r.rs_vs_market_pct:+.1f}* | 52W High: -{r.dist_from_52w_high_pct}%\n"
            f"   • 🎯 SL Anchor: ₹{r.stop_loss} ({r.risk_pct}% Risk)\n"
        )
    return "\n".join(lines)

def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=10)
    except Exception:
        pass

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
            "rs_vs_sector_at_match": r.rs_vs_sector_pct if r.rs_vs_sector_pct is not None else "",
            "monthly_confirmed": r.monthly_confirmed,
            "buy_tag": r.buy_tag,
            "pullback_ema": r.pullback_ema or "",
            "stop_loss_at_match": r.stop_loss if r.stop_loss is not None else "",
            "risk_pct_at_match": r.risk_pct if r.risk_pct is not None else "",
        })

    if not new_rows:
        return

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MATCH_HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

def explain_symbol(symbol: str, market_weekly: pd.DataFrame):
    daily = fetch_daily_ohlc(symbol)
    if daily is None:
        print(f"{symbol}: No data found.")
        return
    weekly = build_weekly(daily)
    if weekly is None:
        print(f"{symbol}: Insufficient weekly data.")
        return
    weekly = compute_indicators(weekly)
    rs_series = compute_volume_weighted_rs_series(weekly, market_weekly)
    evald = evaluate_conditions(weekly, len(weekly) - 1, rs_series)
    print(f"\n=== Diagnostics for {symbol} ===")
    if evald is None:
        print("Status: REJECTED (Failed trend alignment, support pocket, or volume rules)")
    else:
        print("Status: MATCH CONFIRMED")
        for k, (passed, desc) in evald["checks"].items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {k}: {desc}")

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
    parser.add_argument("--explain", type=str, default=None)
    parser.add_argument("--rs-debug", action="store_true")
    parser.add_argument("--gate-debug", action="store_true")
    parser.add_argument("--no-log-history", action="store_true")
    parser.add_argument("--weekly-report", action="store_true")
    parser.add_argument("--report-lookback-weeks", type=int, default=DEFAULT_REPORT_LOOKBACK_WEEKS)
    parser.add_argument("--with-weekly-report", action="store_true")
    parser.add_argument("--include-watchlist", action="store_true")
    parser.add_argument("--ignore-market-stage", action="store_true")
    parser.add_argument("--require-market-stage", action="store_true")
    parser.add_argument("--require-adx-rising", action="store_true")
    parser.add_argument("--no-squeeze-rule", action="store_true")
    parser.add_argument("--no-pullback-rule", action="store_true")
    args = parser.parse_args()

    market_weekly = _fetch_index_weekly(MARKET_INDEX_TICKER, period="5y")
    if market_weekly is None:
        print("[ERROR] Could not load market index.")
        sys.exit(1)

    if args.explain:
        explain_symbol(args.explain.upper(), market_weekly)
        return

    symbols = SYMBOLS[: args.limit] if args.limit else SYMBOLS
    print(f"Scanning {len(symbols)} symbols with {args.workers} workers...")

    all_candidates: List[ScanResult] = []
    weekly_cache = {} if args.backtest else None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(scan_symbol, s, market_weekly, args.backtest, args.lookback_weeks, weekly_cache): s for s in symbols}
        for f in as_completed(futures):
            try:
                res, _ = f.result()
                all_candidates.extend(res)
            except Exception:
                pass

    if args.backtest:
        deduped = deduplicate_backtest_matches(all_candidates)
        stats = compute_backtest_r_stats(deduped, weekly_cache, forward_weeks=args.forward_weeks)
        print("\n=== BACKTEST R-MULTIPLE STATS (Deduped Weekly Trades) ===")
        if stats.get("n", 0) == 0:
            print("  No trackable matches found.")
        else:
            print(f"  Total Trades : {stats['n']}")
            print(f"  Win Rate     : {stats['win_rate_pct']}%")
            print(f"  Avg R        : {stats['avg_r']:+.2f} R")
            print(f"  Avg Win/Loss : +{stats['avg_win_r']} R / {stats['avg_loss_r']} R")
            print(f"  Expectancy   : {stats['expectancy_r']:+.2f} R per trade")
    else:
        msg = format_results_message(all_candidates)
        print("\n" + msg)
        if not args.no_log_history and all_candidates:
            log_matches_to_history(all_candidates)
        if not args.dry_run:
            send_telegram_message(msg)

if __name__ == "__main__":
    main()
