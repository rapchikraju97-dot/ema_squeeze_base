"""
EMA Squeeze Base Scanner v2 — Stage-2 (Base-1 & Base-2) Absorption Engine
-------------------------------------------------------------------------
Coverage: Nifty Total Market (750 Equities)
Features:
  1. Base-1 Inceptions (8–24w post-crossover) & Base-2 Resets (25–40w post-crossover).
  2. Absorption Volume Dry-up (RVOL <= 0.85x) vs Breakout Turns (RVOL >= 1.25x).
  3. Continuous 40W EMA Defense + Zero Overhead Structural Damage.
  4. Dynamic vwRS: Evaluates Stock vs Nifty 500 and Sector Index (skips gracefully if unassigned).
  5. Thread-safe debugging diagnostics and multi-chunk Telegram alerts.
"""

import argparse
import csv
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator

# ---------------------------------------------------------------------------
# Strategy Parameters (Base-1 + Base-2 Coverage)
# ---------------------------------------------------------------------------

NEAR_EMA_TOLERANCE_PCT = 0.028     # 2.8% proximity to 10W / 20W EMA pocket
SUPPORT_CLOSE_TOLERANCE_PCT = 2.0  # Max buffer below 20W EMA line
MAX_PCT_OFF_52W_HIGH = 28.0        # Max distance from 52W high

# Stage-2 Expansion & Base Bounds
MIN_WEEKS_SINCE_CROSSOVER = 8      # Minimum maturity post-Stage 1 cross
MAX_WEEKS_SINCE_CROSSOVER = 40     # Captures both Base-1 and Base-2 setups
MIN_INITIAL_THRUST_PCT = 22.0      # Minimum impulse wave from crossover
MAX_INITIAL_THRUST_PCT = 110.0     # Rejects blow-off tops
MIN_DIST_FROM_40W_PCT = 6.0        # Separation above 40W baseline
MAX_DIST_FROM_40W_PCT = 38.0       # Proximity ceiling (filters extended runners)
MAX_EMA_SPREAD_PCT = 5.8           # 10W-20W spread compression limit
MAX_PULLBACK_NUMBER = 4            # Permits up to Base-2 resets

# Long-Term Overhang Check
PRIOR_HIGH_LOOKBACK_WEEKS = 156    # ~3 years measured before crossover
MAX_PRIOR_HIGH_OVERHANG_PCT = 8.0  # Reject if sitting directly beneath multi-year resistance

# Volume Signatures
VOL_ABSORPTION_MAX_RVOL = 0.85     # Volume dry-up threshold during base rest
HIGH_VOL_IGNITION_RATIO = 1.25     # Ignition volume on breakout turn
TRAILING_VOL_WINDOW = 10           # Baseline volume lookback window
RS_LOOKBACK_WEEKS = 12             # Rolling vwRS window

MARKET_INDEX_TICKER = "^CRSLDX"    # Nifty 500
MARKET_INDEX_LABEL = "Nifty 500"

DOWNLOAD_RETRIES = 3
DOWNLOAD_BACKOFF_SECONDS = [1, 3, 6]
DEFAULT_WORKERS = 10
DEFAULT_PER_REQUEST_DELAY = 0.20

MATCH_HISTORY_CSV = "match_history.csv"
MATCH_HISTORY_FIELDS = [
    "scan_run_date", "symbol", "sector", "week_date", "close_at_match",
    "ema10_at_match", "rs_vs_market_at_match", "rs_vs_sector_at_match",
    "monthly_confirmed", "buy_tag", "pullback_ema", "stop_loss_at_match", "risk_pct_at_match"
]

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# ---------------------------------------------------------------------------
# Sector Benchmarks & Universe Mapping
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
    rs_vs_sector_pct: Optional[float] = None
    dist_from_52w_high_pct: float = 0.0
    breakout_vol_ratio: float = 0.0
    vol_contracting: bool = False
    tightness_label: str = "Tight"
    buy_tag: bool = True
    stop_loss: float = 0.0
    risk_pct: float = 0.0
    pullback_ema: str = "10W"
    setup_type: str = "pullback_absorption"
    weeks_since_crossover: int = 0
    post_cross_gain_pct: float = 0.0
    dist_from_40w_pct: float = 0.0
    prior_high_overhang_pct: float = 0.0
    pullback_number: int = 1

# ---------------------------------------------------------------------------
# Technical Calculations
# ---------------------------------------------------------------------------

def _download_with_retry(ticker: str, period: str = "5y", request_delay: float = 0.0) -> Optional[pd.DataFrame]:
    if request_delay > 0:
        time.sleep(request_delay + random.uniform(0, request_delay * 0.5))
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

    if len(weekly) < 52:
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

    idx_s = pd.to_datetime(stock_weekly.index)
    if idx_s.tz is not None:
        idx_s = idx_s.tz_localize(None)
    idx_b = pd.to_datetime(benchmark_weekly.index)
    if idx_b.tz is not None:
        idx_b = idx_b.tz_localize(None)

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
# Setup Evaluator: Base-1 & Base-2 Absorption Engine
# ---------------------------------------------------------------------------

def evaluate_conditions(df: pd.DataFrame, idx: int, rs_mkt_series: pd.Series, rs_sec_series: Optional[pd.Series],
                        debug_callback=None) -> Optional[dict]:
    def _reject(stage: str):
        if debug_callback is not None:
            debug_callback(stage)
        return None

    if idx < 45 or idx >= len(df):
        return None

    row = df.iloc[idx]
    prev_row = df.iloc[idx - 1]

    # --- 1. CURRENT MOVING AVERAGE ALIGNMENT (10W > 20W > 40W) ---
    if pd.isna(row.ema40) or pd.isna(row.ema20) or pd.isna(row.ema10) or pd.isna(row.ema5):
        return _reject("1_missing_ema_data")

    ema40_prev4 = df.iloc[idx - 4]["ema40"]
    ema10_prev4 = df.iloc[idx - 4]["ema10"]

    # 40W and 10W EMAs must be actively rising
    emas_rising = (row.ema40 >= ema40_prev4) and (row.ema10 >= ema10_prev4)
    bullish_stack = (row.ema5 >= row.ema10 * 0.98) and (row.ema10 > row.ema20) and (row.ema20 > row.ema40) and (row.close > row.ema40)

    if not (emas_rising and bullish_stack):
        return _reject("1_not_bullish_stack")

    # --- 2. FRESH CROSSOVER DETECTION (10W crossed 40W 8 to 40 weeks ago) ---
    crossover_idx = None
    lookback_start = max(1, idx - MAX_WEEKS_SINCE_CROSSOVER)
    lookback_end = max(1, idx - MIN_WEEKS_SINCE_CROSSOVER)

    for i in range(lookback_end, lookback_start, -1):
        curr_bar = df.iloc[i]
        prev_bar = df.iloc[i - 1]
        if prev_bar.ema10 <= prev_bar.ema40 and curr_bar.ema10 > curr_bar.ema40:
            crossover_idx = i
            break

    if crossover_idx is None:
        return _reject("2_no_crossover_in_window")

    weeks_since_crossover = idx - crossover_idx

    # --- 3. FIRST EXPANSION THRUST (+22% to +110% move from cross) ---
    cross_price = df.iloc[crossover_idx]["close"]
    highest_after_cross = df["high"].iloc[crossover_idx: idx + 1].max()
    initial_thrust_pct = ((highest_after_cross - cross_price) / cross_price) * 100

    if not (MIN_INITIAL_THRUST_PCT <= initial_thrust_pct <= MAX_INITIAL_THRUST_PCT):
        return _reject("3_thrust_out_of_range")

    # --- 3b. LONG-TERM OVERHANG CHECK ---
    prior_high_start = max(0, crossover_idx - PRIOR_HIGH_LOOKBACK_WEEKS)
    prior_high = df["high"].iloc[prior_high_start:crossover_idx].max() if crossover_idx > prior_high_start else None

    overhang_pct = 0.0
    if prior_high is not None and pd.notna(prior_high) and prior_high > 0:
        overhang_pct = round((prior_high - row.close) / prior_high * 100, 2)
        if 0 <= overhang_pct <= MAX_PRIOR_HIGH_OVERHANG_PCT:
            return _reject("3b_old_resistance_overhang")

    # --- 4. PROXIMITY GATE: RESTING NEAR 40W BASELINE ---
    dist_from_40w = (row.close - row.ema40) / row.ema40 * 100
    if not (MIN_DIST_FROM_40W_PCT <= dist_from_40w <= MAX_DIST_FROM_40W_PCT):
        return _reject("4_dist_from_40w_out_of_range")

    # --- 5. ZERO STRUCTURAL DAMAGE (40W EMA defended since crossover) ---
    cycle_lows = df["low"].iloc[crossover_idx: idx + 1]
    cycle_ema40 = df["ema40"].iloc[crossover_idx: idx + 1]
    if (cycle_lows < cycle_ema40 * 0.98).any():
        return _reject("5_structural_damage")

    # --- 6. SHALLOW 10W / 20W POCKET TOUCH ---
    touch_10w = (row.low <= row.ema10 * (1 + NEAR_EMA_TOLERANCE_PCT)) and (row.close >= row.ema20 * (1 - SUPPORT_CLOSE_TOLERANCE_PCT / 100))
    touch_20w = (row.low <= row.ema20 * (1 + NEAR_EMA_TOLERANCE_PCT)) and (row.close >= row.ema20 * (1 - SUPPORT_CLOSE_TOLERANCE_PCT / 100))

    if not (touch_10w or touch_20w):
        return _reject("6_no_ema_touch")

    pullback_ema = "10W" if abs(row.close - row.ema10) <= abs(row.close - row.ema20) else "20W"

    # --- 6b. PULLBACK SEQUENCE NUMBER (Hysteresis model) ---
    state = "away"
    pullback_number = 0
    for j in range(crossover_idx, idx + 1):
        bar = df.iloc[j]
        is_near = (bar.low <= bar.ema10 * (1 + NEAR_EMA_TOLERANCE_PCT)) or (bar.low <= bar.ema20 * (1 + NEAR_EMA_TOLERANCE_PCT))
        is_away = bar.close > bar.ema10 * 1.05
        if is_near and state != "in_pullback":
            pullback_number += 1
            state = "in_pullback"
        elif is_away:
            state = "away"

    if pullback_number == 0:
        pullback_number = 1
    if pullback_number > MAX_PULLBACK_NUMBER:
        return _reject("6b_too_many_pullbacks")

    # --- 7. TIGHT CANDLE ANATOMY & LOWER WICK REJECTION ---
    candle_range = row.high - row.low
    bar_range_pct = (candle_range / row.close) * 100 if row.close > 0 else 10.0
    upper_half_close = (row.close >= row.low + (candle_range * 0.35)) if candle_range > 0 else True
    is_tight_bar = bar_range_pct <= 7.5

    if not (upper_half_close or is_tight_bar):
        return _reject("7_loose_candle")

    # --- 8. VOLUME DRY-UP ABSORPTION vs IGNITION ---
    vol_avg = row.get("vol_sma10", 0)
    if pd.isna(vol_avg) or vol_avg <= 0:
        return _reject("8_no_volume_data")

    rvol = round(float(row.volume / vol_avg), 2)
    is_dry_absorption = (rvol <= VOL_ABSORPTION_MAX_RVOL)
    is_ignition_turn = (rvol >= HIGH_VOL_IGNITION_RATIO) and (row.close >= prev_row.high * 0.99)

    if not (is_dry_absorption or is_ignition_turn):
        return _reject("8_volume_signature_fail")

    # --- 9. RELATIVE STRENGTH (vwRS) ---
    # Market RS is strictly mandatory (must be non-negative)
    vw_rs_mkt = rs_mkt_series.asof(row.name) if not rs_mkt_series.empty else np.nan
    if pd.isna(vw_rs_mkt) or vw_rs_mkt < 0.0:
        return _reject("9_rs_market_negative")

    # Sector RS check is executed ONLY if a valid sector series exists
    vw_rs_sec = None
    if rs_sec_series is not None and not rs_sec_series.empty:
        sec_val = rs_sec_series.asof(row.name)
        if pd.notna(sec_val):
            if sec_val < 0.0:
                return _reject("9_rs_sector_negative")
            vw_rs_sec = float(sec_val)

    # Distance from 52W High
    high_52w = row.get("high_52w", float("nan"))
    dist_off_52w = round((high_52w - row.close) / high_52w * 100, 2) if pd.notna(high_52w) and high_52w > 0 else 0.0
    if dist_off_52w > MAX_PCT_OFF_52W_HIGH:
        return _reject("9b_too_far_off_52w_high")

    stop_loss = round(min(row.low, prev_row.low) * 0.99, 2)
    risk_pct = round((row.close - stop_loss) / row.close * 100, 2)
    ema_spread_pct = round(abs(row.ema10 - row.ema20) / row.close * 100, 2)

    base_label = "Base-1" if weeks_since_crossover <= 24 else "Base-2"
    setup_label = f"🔥 {base_label} Breakout Pivot" if is_ignition_turn else f"🛡️ {base_label} Absorption Pocket"

    return {
        "row": row,
        "pullback_ema": pullback_ema,
        "dist_from_52w_high_pct": dist_off_52w,
        "stop_loss": stop_loss,
        "risk_pct": risk_pct,
        "rvol": rvol,
        "vw_rs_mkt": vw_rs_mkt,
        "vw_rs_sec": vw_rs_sec,
        "ema_spread_pct": ema_spread_pct,
        "weeks_since_crossover": weeks_since_crossover,
        "post_cross_gain_pct": round(initial_thrust_pct, 1),
        "dist_from_40w_pct": round(dist_from_40w, 1),
        "prior_high_overhang_pct": overhang_pct,
        "pullback_number": pullback_number,
        "setup_type": setup_label,
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
        weeks_since_crossover=evald["weeks_since_crossover"],
        post_cross_gain_pct=evald["post_cross_gain_pct"],
        dist_from_40w_pct=evald["dist_from_40w_pct"],
        prior_high_overhang_pct=evald.get("prior_high_overhang_pct", 0.0),
        pullback_number=evald.get("pullback_number", 1),
    )

# ---------------------------------------------------------------------------
# Parallel Scanning Pipeline
# ---------------------------------------------------------------------------

def scan_symbol(symbol: str, market_weekly: pd.DataFrame, sector_cache: dict, backtest: bool, lookback_weeks: int,
                weekly_cache: dict = None, request_delay: float = 0.0, debug_callback=None):
    daily = _download_with_retry(f"{symbol}.NS", period="5y", request_delay=request_delay)
    if daily is None:
        return []

    weekly = build_weekly(daily)
    if weekly is None:
        return []

    if weekly_cache is not None:
        weekly_cache[symbol] = weekly

    sector = SECTOR_ASSIGNMENTS.get(symbol, "Other")
    sec_ticker = SECTOR_BENCHMARK_MAP.get(sector, None)
    sector_weekly = sector_cache.get(sec_ticker, None) if sec_ticker else None

    rs_mkt_series = compute_volume_weighted_rs_series(weekly, market_weekly)
    rs_sec_series = compute_volume_weighted_rs_series(weekly, sector_weekly) if sector_weekly is not None else None

    results = []
    if backtest:
        start_idx = max(45, len(weekly) - lookback_weeks)
        for i in range(start_idx, len(weekly)):
            evald = evaluate_conditions(weekly, i, rs_mkt_series, rs_sec_series, debug_callback)
            if evald:
                res = _build_scan_result(evald)
                res.symbol = symbol
                res.sector = sector
                results.append(res)
    else:
        evald = evaluate_conditions(weekly, len(weekly) - 1, rs_mkt_series, rs_sec_series, debug_callback)
        if evald:
            res = _build_scan_result(evald)
            res.symbol = symbol
            res.sector = sector
            results.append(res)

    return results

# ---------------------------------------------------------------------------
# Output & Auto-Chunking Telegram Delivery
# ---------------------------------------------------------------------------

def format_results_message(buy_setups: List[ScanResult]) -> str:
    lines = [
        "⚡ *5-10-20-40 Weekly EMA Absorption Scanner (Stage-2)* ⚡",
        f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}\n"
    ]
    if not buy_setups:
        lines.append("No candidates passed all Stage-2 absorption filters this week across Nifty Total Market.")
        return "\n".join(lines)

    lines.append(f"🎯 *QUALIFIED STAGE-2 SETUPS ({len(buy_setups)})*\n")
    for r in buy_setups:
        sec_rs_str = f"*{r.rs_vs_sector_pct:+.1f}*" if r.rs_vs_sector_pct is not None else "`N/A (Unmapped)`"
        lines.append(
            f"⭐ *{r.symbol}* [{r.sector}] — `{r.setup_type}`\n"
            f"   • Close: ₹{r.close} | Support: *{r.pullback_ema} EMA* | Spread: {r.ema_spread_pct}%\n"
            f"   • Crossover Age: *{r.weeks_since_crossover}w ago* (Initial Thrust: +{r.post_cross_gain_pct}%)\n"
            f"   • Pullback #: *{r.pullback_number}* | Proximity to 40W: *+{r.dist_from_40w_pct}%*\n"
            f"   • Prior-High Overhang: *{r.prior_high_overhang_pct:+.1f}%*\n"
            f"   • RVOL: *{r.breakout_vol_ratio}x* | RSI: {r.rsi14} | ADX: {r.adx14}\n"
            f"   • vwRS (vs Nifty 500): *{r.rs_vs_market_pct:+.1f}* | vs Sector: {sec_rs_str}\n"
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
                print(f"[WARN] Markdown delivery failed (HTTP {resp.status_code}). Trying fallback...")
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
            "rs_vs_sector_at_match": r.rs_vs_sector_pct if r.rs_vs_sector_pct is not None else "",
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
    parser = argparse.ArgumentParser(description="EMA Squeeze Base weekly scanner v2 (Nifty Total Market)")
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
    parser.add_argument("--debug", action="store_true", help="Print rejection-reason breakdown across scanned candidates.")
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
        daily_sec = _download_with_retry(sec_ticker, period="5y", request_delay=args.delay)
        if daily_sec is not None:
            wk = build_weekly(daily_sec)
            if wk is not None:
                sector_cache[sec_ticker] = wk

    symbols = SYMBOLS[: args.limit] if args.limit else SYMBOLS
    print(f"Scanning {len(symbols)} candidates from Nifty Total Market with {args.workers} workers (delay={args.delay}s/req)...")

    all_candidates: List[ScanResult] = []
    weekly_cache = {} if args.backtest else None
    debug_counts = {} if args.debug else None
    debug_lock = threading.Lock() if args.debug else None

    def thread_safe_debug_record(stage: str):
        if debug_counts is not None and debug_lock is not None:
            with debug_lock:
                debug_counts[stage] = debug_counts.get(stage, 0) + 1

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                scan_symbol, s, market_weekly, sector_cache, args.backtest, args.lookback_weeks,
                weekly_cache, args.delay, thread_safe_debug_record if args.debug else None
            ): s
            for s in symbols
        }
        for f in as_completed(futures):
            try:
                res = f.result()
                all_candidates.extend(res)
            except Exception:
                pass

    if args.debug:
        print("\n[DEBUG] Rejection breakdown across all scanned candidates:")
        if not debug_counts:
            print("  (no rejections recorded — check network/data access)")
        else:
            for stage, count in sorted(debug_counts.items(), key=lambda kv: -kv[1]):
                print(f"  {stage:35s} {count}")

    msg = format_results_message(all_candidates)
    print("\n" + msg)

    if not args.no_log_history and all_candidates and not args.backtest:
        log_matches_to_history(all_candidates)

    if not args.dry_run and not args.backtest:
        send_telegram_message(msg)

if __name__ == "__main__":
    main()
