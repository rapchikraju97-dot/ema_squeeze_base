"""
EMA Squeeze Base Weekly Scanner — True Leaders & 10W/20W Support Edition
-----------------------------------------------------------------------
Weekly-timeframe scan for RKScanBot.
Filters across the full 752-stock Nifty Total Market universe.

Core Filter Criteria:
  1. Close MUST be holding at or above the 20W EMA (Strict 10W/20W bounce).
  2. Max 15% off 52-Week High (Only strong leaders, no deep pullbacks).
  3. 5W / 10W / 20W EMA spread tightly coiled (<= 2.8%).
  4. Base compression: Weekly (High-Low)/Close <= 10% for >= 4 weeks.
  5. Hard Risk Gate: Stop-Loss (Base low) <= 7.0% from CMP.
  6. Dynamic Cross-Sectional RS: Top 30% vs Nifty 500.

Requirements:
    pip install yfinance pandas numpy ta requests
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
# Strict Leader & Support Strategy Config
# ---------------------------------------------------------------------------

NEAR_EMA_PCT = 0.030           # Close within 3.0% of 10W or 20W EMA
EMA_SPREAD_MAX_PCT = 2.8      # Max spread between EMA5/10/20 as % of close
UPTREND_REQUIRED = True       # EMA10 >= EMA20 > EMA40 and close > EMA20

ADX_MIN = 20                  # Trend strength baseline
ADX_RISING_REQUIRED = False   # False during base stage so coiled setups aren't dropped

MAX_PCT_OFF_52W_HIGH = 15.0   # STRICT: Max 15% off 52W High (Eliminates 25% deep fallers)

MIN_BASE_WEEKS = 4            # Minimum 4 weeks in base
BASE_RANGE_COMPRESSION_PCT = 10.0  # Weekly (High-Low)/Close <= 10%

VOL_BASE_LOOKBACK_WEEKS = 6   # Volume contraction lookback window
HIGH_VOL_BREAKOUT_RATIO = 1.25 # Current week volume vs base average
TIGHT_COMPRESSION_PCT = 1.5   # Compression below this = "Very Tight"
MONTHLY_RETEST_PCT = 0.05     # Within 5% of 6-month EMA
RS_LOOKBACK_WEEKS = 12

# --- Hard Risk Gate ---
MAX_RISK_PCT = 7.0            # Max allowable stop-loss distance from CMP (Strict)

# --- Dynamic RS Percentile ---
RS_PERCENTILE_MIN = 70.0      # Top 30% of universe vs Market

# --- Market Regime ---
REQUIRE_MARKET_STAGE2 = False # Informational
MARKET_STAGE_MA_WEEKS = 30
MARKET_STAGE_SLOPE_LOOKBACK_WEEKS = 5

THREAT_DIST_52W_HIGH_PCT = 12.0
THREAT_RSI_EXTENDED = 68

MARKET_INDEX_TICKER = "^CRSLDX"             # Nifty 500
MARKET_INDEX_LABEL = "Nifty 500"
FALLBACK_SECTOR_INDEX_TICKER = "^CRSLDX"

DOWNLOAD_RETRIES = 3
DOWNLOAD_BACKOFF_SECONDS = [1, 3, 8]
DEFAULT_WORKERS = 8
DEFAULT_PER_REQUEST_DELAY = 0.2

MATCH_HISTORY_CSV = "match_history.csv"
MATCH_HISTORY_FIELDS = [
    "scan_run_date", "symbol", "sector", "week_date", "close_at_match",
    "ema10_at_match", "rs_vs_sector_at_match", "monthly_confirmed", "buy_tag",
]

# Verified Active Yahoo Finance Sector Benchmarks
SECTOR_BENCHMARK_MAP = {
    "Financial Services": "NIFTY_FIN_SERVICE.NS",
    "Automobile and Auto Components": "^CNXAUTO",
    "Fast Moving Consumer Goods": "^CNXFMCG",
    "Healthcare": "^CNXPHARMA",
    "Information Technology": "^CNXIT",
    "Metals & Mining": "^CNXMETAL",
    "Oil Gas & Consumable Fuels": "^CNXENERGY",
    "Realty": "^CNXREALTY",
}

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# 752-stock Nifty Total Market Universe
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

SECTOR_MAP = {
    "360ONE": "Financial Services", "3MINDIA": "Diversified", "ABB": "Capital Goods", "ACC": "Construction Materials",
    "ACMESOLAR": "Power", "AIAENG": "Capital Goods", "APLAPOLLO": "Capital Goods", "ASKAUTOLTD": "Automobile and Auto Components",
    "AUBANK": "Financial Services", "AWL": "Fast Moving Consumer Goods", "AXISCADES": "Capital Goods", "AADHARHFC": "Financial Services",
    "AARTIDRUGS": "Healthcare", "AARTIIND": "Chemicals", "AARTIPHARM": "Healthcare", "AAVAS": "Financial Services",
    "ABBOTINDIA": "Healthcare", "ACE": "Capital Goods", "ACUTAAS": "Healthcare", "ADANIENSOL": "Power",
    "ADANIENT": "Metals & Mining", "ADANIGREEN": "Power", "ADANIPORTS": "Services", "ADANIPOWER": "Power",
    "ATGL": "Oil Gas & Consumable Fuels", "ABCAPITAL": "Financial Services", "ABFRL": "Consumer Services", "ABLBL": "Consumer Services",
    "ABREL": "Realty", "ABSLAMC": "Financial Services", "CPPLUS": "Capital Goods", "AVL": "Consumer Services",
    "ADVENZYMES": "Healthcare", "AEGISLOG": "Oil Gas & Consumable Fuels", "AEGISVOPAK": "Oil Gas & Consumable Fuels", "AEQUS": "Capital Goods",
    "AETHER": "Chemicals", "AFCONS": "Construction", "AFFLE": "Information Technology", "AHLUCONT": "Construction",
    "AJANTPHARM": "Healthcare", "AKUMS": "Healthcare", "APLLTD": "Healthcare", "ALIVUS": "Healthcare",
    "ALKEM": "Healthcare", "ALKYLAMINE": "Chemicals", "ABDL": "Fast Moving Consumer Goods", "ALOKINDS": "Textiles",
    "ARE&M": "Automobile and Auto Components", "AMBER": "Consumer Durables", "AMBUJACEM": "Construction Materials", "ANANDRATHI": "Financial Services",
    "ANANTRAJ": "Realty", "ANGELONE": "Financial Services", "ANTHEM": "Healthcare", "ANURAS": "Chemicals",
    "APARINDS": "Capital Goods", "APOLLOHOSP": "Healthcare", "APOLLO": "Capital Goods", "APOLLOTYRE": "Automobile and Auto Components",
    "APTUS": "Financial Services", "ACI": "Chemicals", "ARVINDFASN": "Consumer Services", "ARVIND": "Textiles",
    "ASAHIINDIA": "Automobile and Auto Components", "ASHAPURMIN": "Metals & Mining", "ASHOKLEY": "Capital Goods", "ASHOKA": "Construction",
    "ASIANPAINT": "Consumer Durables", "ASTERDM": "Healthcare", "ASTRAMICRO": "Capital Goods", "ASTRAL": "Capital Goods",
    "ATHERENERG": "Automobile and Auto Components", "ATLANTAELE": "Capital Goods", "ATUL": "Chemicals", "AURIONPRO": "Information Technology",
    "AUROPHARMA": "Healthcare", "AIIL": "Financial Services", "AVALON": "Capital Goods", "AVANTIFEED": "Fast Moving Consumer Goods",
    "DMART": "Consumer Services", "CCAVENUE": "Financial Services", "AWFIS": "Services", "AXISBANK": "Financial Services",
    "AZAD": "Capital Goods", "BEML": "Capital Goods", "BLS": "Consumer Services", "BSE": "Financial Services",
    "BAJAJ-AUTO": "Automobile and Auto Components", "BAJAJELEC": "Consumer Durables", "BAJFINANCE": "Financial Services", "BAJAJFINSV": "Financial Services",
    "BAJAJHLDNG": "Financial Services", "BAJAJHFL": "Financial Services", "BALAMINES": "Chemicals", "BALKRISIND": "Automobile and Auto Components",
    "BALRAMCHIN": "Fast Moving Consumer Goods", "BALUFORGE": "Capital Goods", "BANCOINDIA": "Automobile and Auto Components", "BANDHANBNK": "Financial Services",
    "BANKBARODA": "Financial Services", "BANKINDIA": "Financial Services", "MAHABANK": "Financial Services", "BATAINDIA": "Consumer Durables",
    "BAYERCROP": "Chemicals", "BELRISE": "Automobile and Auto Components", "BERGEPAINT": "Consumer Durables", "BDL": "Capital Goods",
    "BEL": "Capital Goods", "BHARATFORG": "Automobile and Auto Components", "BHEL": "Capital Goods", "BPCL": "Oil Gas & Consumable Fuels",
    "BHARTIARTL": "Telecommunication", "BHARTIHEXA": "Telecommunication", "BIKAJI": "Fast Moving Consumer Goods", "GROWW": "Financial Services",
    "BIOCON": "Healthcare", "BIRLACORPN": "Construction Materials", "BSOFT": "Information Technology", "BBOX": "Information Technology",
    "BLACKBUCK": "Services", "BLUEDART": "Services", "BLUEJET": "Healthcare", "BLUESTARCO": "Consumer Durables",
    "BLUESTONE": "Consumer Durables", "BBTC": "Fast Moving Consumer Goods", "BORORENEW": "Capital Goods", "BOSCHLTD": "Automobile and Auto Components",
    "FIRSTCRY": "Consumer Services", "BRIGADE": "Realty", "BRITANNIA": "Fast Moving Consumer Goods", "MAPMYINDIA": "Information Technology",
    "CCL": "Fast Moving Consumer Goods", "CESC": "Power", "CGPOWER": "Capital Goods", "CIEINDIA": "Automobile and Auto Components",
    "CMSINFO": "Services", "CORONA": "Healthcare", "CRISIL": "Financial Services", "CSBBANK": "Financial Services",
    "CAMPUS": "Consumer Durables", "CANFINHOME": "Financial Services", "CANBK": "Financial Services", "CANHLIFE": "Financial Services",
    "CRAMC": "Financial Services", "CAPILLARY": "Information Technology", "CAPLIPOINT": "Healthcare", "CGCL": "Financial Services",
    "CARBORUNIV": "Capital Goods", "CARTRADE": "Consumer Services", "CASTROLIND": "Oil Gas & Consumable Fuels", "CEATLTD": "Automobile and Auto Components",
    "CELLO": "Consumer Durables", "CEMPRO": "Construction", "CENTRALBK": "Financial Services", "CDSL": "Financial Services",
    "CENTURYPLY": "Consumer Durables", "CERA": "Consumer Durables", "CHALET": "Consumer Services", "CHAMBLFERT": "Chemicals",
    "CHENNPETRO": "Oil Gas & Consumable Fuels", "CHOICEIN": "Financial Services", "CHOLAHLDNG": "Financial Services", "CHOLAFIN": "Financial Services",
    "CIPLA": "Healthcare", "CUB": "Financial Services", "CLEAN": "Chemicals", "COALINDIA": "Oil Gas & Consumable Fuels",
    "COCHINSHIP": "Capital Goods", "COFORGE": "Information Technology", "COHANCE": "Healthcare", "COLPAL": "Fast Moving Consumer Goods",
    "CAMS": "Financial Services", "CONCORDBIO": "Healthcare", "CONCOR": "Services", "COROMANDEL": "Chemicals",
    "CRAFTSMAN": "Automobile and Auto Components", "CREDITACC": "Financial Services", "CRIZAC": "Consumer Services", "CROMPTON": "Consumer Durables",
    "CUMMINSIND": "Capital Goods", "CUPID": "Fast Moving Consumer Goods", "CYIENT": "Information Technology", "DCBBANK": "Financial Services",
    "DCMSHRIRAM": "Diversified", "DLF": "Realty", "DOMS": "Fast Moving Consumer Goods", "DABUR": "Fast Moving Consumer Goods",
    "DALBHARAT": "Construction Materials", "DATAPATTNS": "Capital Goods", "DATAMATICS": "Information Technology", "DEEPAKFERT": "Chemicals",
    "DEEPAKNTR": "Chemicals", "DELHIVERY": "Services", "DEVYANI": "Consumer Services", "DIACABS": "Capital Goods",
    "DBL": "Construction", "DIVISLAB": "Healthcare", "DIXON": "Consumer Durables", "AGARWALEYE": "Healthcare",
    "LALPATHLAB": "Healthcare", "DRREDDY": "Healthcare",
    "DYNAMATECH": "Capital Goods", "EIDPARRY": "Fast Moving Consumer Goods", "EIHOTEL": "Consumer Services", "EPL": "Capital Goods",
    "EDELWEISS": "Financial Services", "EICHERMOT": "Automobile and Auto Components", "ELECON": "Capital Goods", "EMIL": "Consumer Services",
    "ELECTCAST": "Capital Goods", "ELGIEQUIP": "Capital Goods", "ELLEN": "Chemicals", "EMAMILTD": "Fast Moving Consumer Goods",
    "EMBDL": "Realty", "EMCURE": "Healthcare", "EMMVEE": "Capital Goods", "ENDURANCE": "Automobile and Auto Components",
    "ENGINERSIN": "Construction", "ENTERO": "Consumer Services", "EIEL": "Utilities", "EQUITASBNK": "Financial Services",
    "ERIS": "Healthcare", "ESCORTS": "Capital Goods", "ETERNAL": "Consumer Services", "ETHOSLTD": "Consumer Durables",
    "EUREKAFORB": "Consumer Durables", "EXIDEIND": "Automobile and Auto Components", "NYKAA": "Consumer Services", "FEDFINA": "Financial Services",
    "FEDERALBNK": "Financial Services", "FACT": "Chemicals", "FIEMIND": "Automobile and Auto Components", "FINCABLES": "Capital Goods",
    "FINPIPE": "Capital Goods", "FSL": "Services", "FIVESTAR": "Financial Services", "FORCEMOT": "Automobile and Auto Components",
    "FORTIS": "Healthcare", "UTLSOLAR": "Capital Goods", "GAIL": "Oil Gas & Consumable Fuels", "GVT&D": "Capital Goods",
    "GHCL": "Chemicals", "GMMPFAUDLR": "Capital Goods", "GMRAIRPORT": "Services", "GMRP&UI": "Power",
    "GABRIEL": "Automobile and Auto Components", "GALLANTT": "Capital Goods", "GRSE": "Capital Goods", "GRWRHITECH": "Capital Goods",
    "GICRE": "Financial Services", "GILLETTE": "Fast Moving Consumer Goods", "GLAND": "Healthcare", "GLAXO": "Healthcare",
    "GLENMARK": "Healthcare", "MEDANTA": "Healthcare", "GODIGIT": "Financial Services", "GPIL": "Capital Goods",
    "GODFRYPHLP": "Fast Moving Consumer Goods", "GODREJAGRO": "Fast Moving Consumer Goods", "GODREJCP": "Fast Moving Consumer Goods", "GODREJIND": "Diversified",
    "GODREJPROP": "Realty", "GOKEX": "Textiles", "GOKULAGRO": "Fast Moving Consumer Goods", "GRANULES": "Healthcare",
    "GRAPHITE": "Capital Goods", "GRASIM": "Construction Materials", "GRAVITA": "Metals & Mining", "GESHIP": "Services",
    "GREAVESCOT": "Capital Goods", "GRINDWELL": "Capital Goods", "GAEL": "Fast Moving Consumer Goods", "FLUOROCHEM": "Chemicals",
    "GMDCLTD": "Metals & Mining", "GNFC": "Chemicals", "GPPL": "Services", "GSFC": "Chemicals",
    "HEG": "Capital Goods", "HGINFRA": "Construction", "HBLENGINE": "Capital Goods", "HCLTECH": "Information Technology",
    "HDBFS": "Financial Services", "HDFCAMC": "Financial Services", "HDFCBANK": "Financial Services", "HDFCLIFE": "Financial Services",
    "HFCL": "Telecommunication", "HAPPSTMNDS": "Information Technology", "HAVELLS": "Consumer Durables", "HCG": "Healthcare",
    "HEMIPROP": "Services", "HERITGFOOD": "Fast Moving Consumer Goods", "HEROMOTOCO": "Automobile and Auto Components", "HEXT": "Information Technology",
    "HSCL": "Chemicals", "HINDALCO": "Metals & Mining", "HAL": "Capital Goods", "HCC": "Construction",
    "HINDCOPPER": "Metals & Mining", "HINDPETRO": "Oil Gas & Consumable Fuels", "HINDUNILVR": "Fast Moving Consumer Goods", "HINDZINC": "Metals & Mining",
    "POWERINDIA": "Capital Goods", "HOMEFIRST": "Financial Services", "HONASA": "Fast Moving Consumer Goods", "HONAUT": "Capital Goods",
    "HUDCO": "Financial Services", "HYUNDAI": "Automobile and Auto Components", "ICICIBANK": "Financial Services", "ICICIGI": "Financial Services",
    "ICICIAMC": "Financial Services", "ICICIPRULI": "Financial Services", "IDBI": "Financial Services", "IDFCFIRSTB": "Financial Services",
    "IFBIND": "Consumer Durables", "IFCI": "Financial Services", "IIFLCAPS": "Financial Services", "IIFL": "Financial Services",
    "INOXINDIA": "Capital Goods", "IRB": "Construction", "IRCON": "Construction", "ITCHOTELS": "Consumer Services",
    "ITC": "Fast Moving Consumer Goods", "ITI": "Telecommunication", "INDGN": "Healthcare", "INDIACEM": "Construction Materials",
    "INDIAGLYCO": "Fast Moving Consumer Goods", "INDIASHLTR": "Financial Services", "INDIAMART": "Consumer Services", "INDIANB": "Financial Services",
    "IEX": "Financial Services", "INDHOTEL": "Consumer Services", "IMFA": "Metals & Mining", "IOC": "Oil Gas & Consumable Fuels",
    "IOB": "Financial Services", "IRCTC": "Consumer Services", "IRFC": "Financial Services", "IREDA": "Financial Services",
    "INDIGOPNTS": "Consumer Durables", "ICIL": "Textiles", "IGL": "Oil Gas & Consumable Fuels", "INDUSTOWER": "Telecommunication",
    "INDUSINDBK": "Financial Services", "NAUKRI": "Consumer Services", "INFY": "Information Technology", "INOXGREEN": "Services",
    "INOXWIND": "Capital Goods", "INTELLECT": "Information Technology", "INDIGO": "Services", "IGIL": "Services",
    "IKS": "Information Technology", "IONEXCHANG": "Utilities", "IPCALAB": "Healthcare", "JKCEMENT": "Construction Materials",
    "JBMA": "Automobile and Auto Components", "JKLAKSHMI": "Construction Materials", "JKPAPER": "Forest Materials", "JKTYRE": "Automobile and Auto Components",
    "JMFINANCIL": "Financial Services", "JSWCEMENT": "Construction Materials", "JSWDULUX": "Consumer Durables", "JSWENERGY": "Power",
    "JSWINFRA": "Services", "JSWSTEEL": "Metals & Mining", "JAIBALAJI": "Metals & Mining", "JAINREC": "Metals & Mining",
    "JPPOWER": "Power", "J&KBANK": "Financial Services", "JAMNAAUTO": "Automobile and Auto Components", "JSFB": "Financial Services",
    "JAYNECOIND": "Capital Goods", "JSLL": "Consumer Services", "JINDALSAW": "Capital Goods", "JSL": "Metals & Mining",
    "JINDALSTEL": "Metals & Mining", "JIOFIN": "Financial Services", "JUBLFOOD": "Consumer Services", "JUBLINGREA": "Chemicals",
    "JUBLPHARMA": "Healthcare", "JLHL": "Healthcare", "JWL": "Capital Goods", "JUSTDIAL": "Consumer Services",
    "JYOTHYLAB": "Fast Moving Consumer Goods", "JYOTICNC": "Capital Goods", "KPRMILL": "Textiles", "KEI": "Capital Goods",
    "KNRCON": "Construction", "KPIGREEN": "Power", "KPITTECH": "Information Technology", "KRBL": "Fast Moving Consumer Goods",
    "KRN": "Capital Goods", "KSB": "Capital Goods", "KAJARIACER": "Consumer Durables", "KPIL": "Construction",
    "KALYANKJIL": "Consumer Durables", "KANSAINER": "Consumer Durables", "KTKBANK": "Financial Services", "KARURVYSYA": "Financial Services",
    "KSCL": "Fast Moving Consumer Goods", "KAYNES": "Capital Goods", "KEC": "Construction", "KFINTECH": "Financial Services",
    "KIRLOSBROS": "Capital Goods", "KIRLOSENG": "Capital Goods", "KIRLPNU": "Capital Goods", "KITEX": "Textiles",
    "KOTAKBANK": "Financial Services", "KIMS": "Healthcare", "LTF": "Financial Services", "LTTS": "Information Technology",
    "LGEINDIA": "Consumer Durables", "LICHSGFIN": "Financial Services", "LTFOODS": "Fast Moving Consumer Goods", "LTM": "Information Technology",
    "LT": "Construction", "LATENTVIEW": "Information Technology", "LAURUSLABS": "Healthcare", "LXCHEM": "Chemicals",
    "IXIGO": "Consumer Services", "THELEELA": "Consumer Services", "LEMONTREE": "Consumer Services", "LENSKART": "Consumer Services",
    "LICI": "Financial Services", "LINDEINDIA": "Chemicals", "LLOYDSENGG": "Capital Goods", "LLOYDSENT": "Metals & Mining",
    "LLOYDSME": "Metals & Mining", "LODHA": "Realty", "LUMAXTECH": "Automobile and Auto Components", "LUPIN": "Healthcare",
    "MMTC": "Services", "MOIL": "Metals & Mining", "MRF": "Automobile and Auto Components", "MSTCLTD": "Services",
    "MTARTECH": "Capital Goods", "MGL": "Oil Gas & Consumable Fuels", "MAHSCOOTER": "Financial Services", "MAHSEAMLES": "Capital Goods",
    "M&MFIN": "Financial Services", "M&M": "Automobile and Auto Components", "MANAPPURAM": "Financial Services", "MRPL": "Oil Gas & Consumable Fuels",
    "MANKIND": "Healthcare", "MANORAMA": "Fast Moving Consumer Goods", "MARICO": "Fast Moving Consumer Goods", "MARKSANS": "Healthcare",
    "MARUTI": "Automobile and Auto Components", "MASTEK": "Information Technology", "MFSL": "Financial Services", "MAXHEALTH": "Healthcare",
    "MAZDOCK": "Capital Goods", "MEDPLUS": "Consumer Services", "MEESHO": "Consumer Services", "METROPOLIS": "Healthcare",
    "MINDACORP": "Automobile and Auto Components", "MIDHANI": "Capital Goods", "MSUMI": "Automobile and Auto Components", "MOTILALOFS": "Financial Services",
    "MPHASIS": "Information Technology", "BECTORFOOD": "Fast Moving Consumer Goods", "MCX": "Financial Services", "MUTHOOTFIN": "Financial Services",
    "NATCOPHARM": "Healthcare", "NBCC": "Construction", "NCC": "Construction", "NEOGEN": "Chemicals",
    "NESCO": "Services", "NHPC": "Power", "NLCINDIA": "Power", "NMDC": "Metals & Mining",
    "NSLNISP": "Metals & Mining", "NTPCGREEN": "Power", "NTPC": "Power", "NH": "Healthcare",
    "NATIONALUM": "Metals & Mining", "NFL": "Chemicals", "NAVA": "Power", "NAVINFLUOR": "Chemicals",
    "NAZARA": "Media Entertainment & Publication", "NESTLEIND": "Fast Moving Consumer Goods", "NETWEB": "Information Technology", "NETWORK18": "Media Entertainment & Publication",
    "NEULANDLAB": "Healthcare", "NEWGEN": "Information Technology", "NAM-INDIA": "Financial Services", "NIVABUPA": "Financial Services",
    "NUVAMA": "Financial Services", "NUVOCO": "Construction Materials", "OBEROIRLTY": "Realty", "ONGC": "Oil Gas & Consumable Fuels",
    "OIL": "Oil Gas & Consumable Fuels", "OLAELEC": "Automobile and Auto Components", "OLECTRA": "Automobile and Auto Components", "PAYTM": "Financial Services",
    "ONESOURCE": "Healthcare", "OPTIEMUS": "Telecommunication", "OFSS": "Information Technology", "ORIENTCEM": "Construction Materials",
    "ORKLAINDIA": "Fast Moving Consumer Goods", "OSWALPUMPS": "Capital Goods", "PNGJL": "Consumer Durables", "POLICYBZR": "Financial Services",
    "PCJEWELLER": "Consumer Durables", "PCBL": "Chemicals", "PGEL": "Consumer Durables", "PIIND": "Chemicals",
    "PNBHOUSING": "Financial Services", "PNCINFRA": "Construction", "PTC": "Power", "PTCIL": "Capital Goods",
    "PVRINOX": "Media Entertainment & Publication", "PAGEIND": "Textiles", "PARADEEP": "Chemicals", "PARAS": "Capital Goods",
    "PARKHOSPS": "Healthcare", "PATANJALI": "Fast Moving Consumer Goods", "PGIL": "Textiles", "PERSISTENT": "Information Technology",
    "PETRONET": "Oil Gas & Consumable Fuels", "PFIZER": "Healthcare", "PHOENIXLTD": "Realty", "PWL": "Consumer Services",
    "PICCADIL": "Fast Moving Consumer Goods", "PIDILITIND": "Chemicals", "PINELABS": "Financial Services", "PIRAMALFIN": "Financial Services",
    "PPLPHARMA": "Healthcare", "POLYMED": "Healthcare", "POLYCAB": "Capital Goods", "POONAWALLA": "Financial Services",
    "PFC": "Financial Services", "POWERGRID": "Power", "POWERMECH": "Construction", "PRAJIND": "Capital Goods",
    "PREMIERENE": "Capital Goods", "PRESTIGE": "Realty", "PRICOLLTD": "Automobile and Auto Components", "PFOCUS": "Media Entertainment & Publication",
    "PRSMJOHNSN": "Construction Materials", "PRIVISCL": "Chemicals", "PRUDENT": "Financial Services", "PNB": "Financial Services",
    "PURVA": "Realty", "QPOWER": "Capital Goods", "QUESS": "Services", "RRKABEL": "Capital Goods",
    "RBLBANK": "Financial Services", "RECLTD": "Financial Services", "RHIM": "Capital Goods", "RITES": "Construction",
    "RADICO": "Fast Moving Consumer Goods", "RVNL": "Construction", "RAILTEL": "Telecommunication", "RAIN": "Chemicals",
    "RAINBOW": "Healthcare", "RALLIS": "Chemicals", "RKFORGE": "Automobile and Auto Components", "RCF": "Chemicals",
    "RATEGAIN": "Information Technology", "RATNAMANI": "Capital Goods", "RTNINDIA": "Consumer Services", "RTNPOWER": "Power",
    "RAYMONDLSL": "Textiles", "REDINGTON": "Services", "REDTAPE": "Consumer Durables", "REFEX": "Utilities",
    "RELAXO": "Consumer Durables", "RELIANCE": "Oil Gas & Consumable Fuels", "RPOWER": "Power", "RELIGARE": "Financial Services",
    "RBA": "Consumer Services", "ROUTE": "Telecommunication", "RUBICON": "Healthcare", "SBFC": "Financial Services",
    "SBICARD": "Financial Services", "SBILIFE": "Financial Services", "SJVN": "Power", "SKFINDUS": "Capital Goods",
    "SKFINDIA": "Automobile and Auto Components", "SKYGOLD": "Consumer Durables", "SMLMAH": "Capital Goods", "SHRIPISTON": "Automobile and Auto Components",
    "SRF": "Chemicals", "SAATVIKGL": "Capital Goods", "SAFARI": "Consumer Durables", "SAGILITY": "Information Technology",
    "SAILIFE": "Healthcare", "SAMHI": "Consumer Services", "SAMMAANCAP": "Financial Services", "MOTHERSON": "Automobile and Auto Components",
    "SANDUMA": "Metals & Mining", "SANOFICONR": "Healthcare", "SANSERA": "Automobile and Auto Components", "SAPPHIRE": "Consumer Services",
    "SARDAEN": "Metals & Mining", "SAREGAMA": "Media Entertainment & Publication", "SCHAEFFLER": "Automobile and Auto Components", "SCHNEIDER": "Capital Goods",
    "SENCO": "Consumer Durables", "STYL": "Financial Services", "SHAILY": "Consumer Durables", "SHAKTIPUMP": "Capital Goods",
    "SHARDACROP": "Chemicals", "SHAREINDIA": "Financial Services", "SFL": "Consumer Durables", "SHILPAMED": "Healthcare",
    "SCI": "Services", "SHREECEM": "Construction Materials", "RENUKA": "Fast Moving Consumer Goods", "SHRIRAMFIN": "Financial Services",
    "SHYAMMETL": "Capital Goods", "ENRIN": "Capital Goods", "SIEMENS": "Capital Goods", "SIGNATURE": "Realty",
    "SKIPPER": "Capital Goods", "SMARTWORKS": "Services", "SOBHA": "Realty", "SOLARINDS": "Chemicals",
    "SONACOMS": "Automobile and Auto Components", "SONATSOFTW": "Information Technology", "SOUTHBANK": "Financial Services", "LOTUSDEV": "Realty",
    "STARCEMENT": "Construction Materials", "STARHEALTH": "Financial Services", "SBIN": "Financial Services", "SAIL": "Metals & Mining",
    "SWSOLAR": "Construction", "STLTECH": "Telecommunication", "STAR": "Healthcare", "STYRENIX": "Chemicals",
    "SUBROS": "Capital Goods", "SUDARSCHEM": "Chemicals", "SUDEEPPHRM": "Chemicals", "SUMICHEM": "Chemicals",
    "SPARC": "Healthcare", "SUNPHARMA": "Healthcare", "SUNTV": "Media Entertainment & Publication", "SUNDARMFIN": "Financial Services",
    "SUNTECK": "Realty", "SUPREMEIND": "Capital Goods", "SPLPETRO": "Chemicals", "SUPRIYA": "Healthcare",
    "SURYAROSNI": "Capital Goods", "SUZLON": "Capital Goods", "SWANCORP": "Chemicals", "SWIGGY": "Consumer Services",
    "SYNGENE": "Healthcare", "SYRMA": "Capital Goods", "TARC": "Realty", "TBOTEK": "Consumer Services",
    "TDPOWERSYS": "Capital Goods", "TSFINV": "Financial Services", "TVSMOTOR": "Automobile and Auto Components", "TVSSCS": "Services",
    "TMB": "Financial Services", "TANLA": "Information Technology", "TATACAP": "Financial Services", "TATACHEM": "Chemicals",
    "TATACOMM": "Telecommunication", "TCS": "Information Technology", "TATACONSUM": "Fast Moving Consumer Goods", "TATAELXSI": "Information Technology",
    "TATAINVEST": "Financial Services", "TMCV": "Capital Goods", "TMPV": "Automobile and Auto Components", "TATAPOWER": "Power",
    "TATASTEEL": "Metals & Mining", "TATATECH": "Information Technology", "TTML": "Telecommunication", "TECHM": "Information Technology",
    "TECHNOE": "Construction", "TEGA": "Capital Goods", "TEJASNET": "Telecommunication", "TENNIND": "Automobile and Auto Components",
    "TEXRAIL": "Capital Goods", "THANGAMAYL": "Consumer Durables", "ANUP": "Capital Goods", "NIACL": "Financial Services",
    "RAMCOCEM": "Construction Materials", "THERMAX": "Capital Goods", "THOMASCOOK": "Consumer Services", "THYROCARE": "Healthcare",
    "TI": "Fast Moving Consumer Goods", "TIMETECHNO": "Capital Goods", "TIMKEN": "Capital Goods", "TIPSMUSIC": "Media Entertainment & Publication",
    "TITAGARH": "Capital Goods", "TITAN": "Consumer Durables", "TORNTPHARM": "Healthcare", "TORNTPOWER": "Power",
    "TARIL": "Capital Goods", "TRANSRAILL": "Capital Goods", "TRAVELFOOD": "Consumer Services", "TRENT": "Consumer Services",
    "TRIDENT": "Textiles", "TRIVENI": "Fast Moving Consumer Goods", "TRITURBINE": "Capital Goods", "TIINDIA": "Automobile and Auto Components",
    "UCOBANK": "Financial Services", "UNOMINDA": "Automobile and Auto Components", "UPL": "Chemicals", "UTIAMC": "Financial Services",
    "UJJIVANSFB": "Financial Services", "ULTRACEMCO": "Construction Materials", "UNIONBANK": "Financial Services", "UBL": "Fast Moving Consumer Goods",
    "UNITDSPR": "Fast Moving Consumer Goods", "URBANCO": "Consumer Services", "USHAMART": "Capital Goods", "VGUARD": "Consumer Durables",
    "VMART": "Consumer Services", "VIPIND": "Consumer Durables", "V2RETAIL": "Consumer Services", "WABAG": "Utilities",
    "VAIBHAVGBL": "Consumer Durables", "DBREALTY": "Realty", "VTL": "Textiles", "VARROC": "Automobile and Auto Components",
    "VBL": "Fast Moving Consumer Goods", "MANYAVAR": "Consumer Services", "VEDL": "Metals & Mining", "VIJAYA": "Healthcare",
    "VIKRAMSOLR": "Capital Goods", "VMM": "Consumer Services", "VIYASH": "Healthcare", "IDEA": "Telecommunication",
    "VOLTAMP": "Capital Goods", "VOLTAS": "Consumer Durables", "WAAREEENER": "Capital Goods", "WAAREERTL": "Capital Goods",
    "WAKEFIT": "Consumer Durables", "WEWORK": "Services", "WEBELSOLAR": "Capital Goods", "WELCORP": "Capital Goods",
    "WELENT": "Construction", "WELSPUNLIV": "Textiles", "WESTLIFE": "Consumer Services", "WHIRLPOOL": "Consumer Durables",
    "WIPRO": "Information Technology", "WOCKPHARMA": "Healthcare", "YATHARTH": "Healthcare", "YESBANK": "Financial Services",
    "ZFCVINDIA": "Automobile and Auto Components", "ZAGGLE": "Information Technology", "ZEEL": "Media Entertainment & Publication", "ZENTEC": "Capital Goods",
    "ZENSARTECH": "Information Technology", "ZYDUSLIFE": "Healthcare", "ZYDUSWELL": "Fast Moving Consumer Goods", "ECLERX": "Services",
}


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
    monthly_confirmed: bool = False
    dist_from_52w_high_pct: Optional[float] = None
    base_weeks: Optional[int] = None
    breakout_vol_ratio: Optional[float] = None
    vol_contracting: Optional[bool] = None
    tightness_label: Optional[str] = None
    buy_tag: bool = False
    stop_loss: Optional[float] = None
    risk_pct: Optional[float] = None
    threats: List[str] = field(default_factory=list)


def _download_with_retry(ticker: str, period: str, label: str) -> Optional[pd.DataFrame]:
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            daily = yf.download(
                ticker, period=period, interval="1d", progress=False,
                auto_adjust=True, timeout=15,
            )
            if daily is not None and not daily.empty:
                return daily
        except Exception:
            pass
        if attempt < DOWNLOAD_RETRIES - 1:
            time.sleep(DOWNLOAD_BACKOFF_SECONDS[min(attempt, len(DOWNLOAD_BACKOFF_SECONDS) - 1)])

    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True, timeout=15)
        if hist is not None and not hist.empty:
            return hist
    except Exception:
        pass
    return None


def fetch_daily_ohlc(symbol: str, period: str = "5y") -> Optional[pd.DataFrame]:
    ticker = f"{symbol}.NS"
    daily = _download_with_retry(ticker, period, label=symbol)
    if daily is None:
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

    if len(weekly) > 0:
        today = pd.Timestamp.now().normalize()
        if weekly.index[-1] > today:
            weekly = weekly.iloc[:-1]

    if len(weekly) < 45:
        return None
    return weekly


def build_monthly_trend(daily: pd.DataFrame) -> pd.DataFrame:
    monthly = daily.resample("ME").agg({"close": "last"}).dropna()
    monthly["ema6_m"] = monthly["close"].ewm(span=6, adjust=False).mean()
    monthly["ema20_m"] = monthly["close"].ewm(span=20, adjust=False).mean()
    monthly["monthly_uptrend"] = monthly["ema6_m"] > monthly["ema20_m"]
    monthly["dist_to_ema6_m_pct"] = (monthly["close"] - monthly["ema6_m"]).abs() / monthly["close"]
    return monthly[["monthly_uptrend", "dist_to_ema6_m_pct", "ema6_m", "ema20_m"]]


def _clean_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.as_unit("ns")
    return df


def attach_monthly_trend(weekly: pd.DataFrame, monthly_trend: pd.DataFrame) -> pd.DataFrame:
    weekly = _clean_datetime_index(weekly)
    monthly_shifted = _clean_datetime_index(monthly_trend)
    monthly_shifted.index = monthly_shifted.index + pd.Timedelta(days=1)

    merged = pd.merge_asof(
        weekly.sort_index(), monthly_shifted.sort_index(),
        left_index=True, right_index=True, direction="backward",
    )
    merged["monthly_uptrend"] = merged["monthly_uptrend"].fillna(False)
    merged["dist_to_ema6_m_pct"] = merged["dist_to_ema6_m_pct"].fillna(999.0)
    return merged


def _fetch_index_weekly(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    daily = _download_with_retry(ticker, period, label=ticker)
    if daily is None:
        return None
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    daily.columns = [c.lower() for c in daily.columns]
    weekly = daily.resample("W-FRI").agg({"close": "last", "volume": "sum"}).dropna()
    if len(weekly) > 0:
        today = pd.Timestamp.now().normalize()
        if weekly.index[-1] > today:
            weekly = weekly.iloc[:-1]
    return weekly if len(weekly) > RS_LOOKBACK_WEEKS else None


def compute_volume_weighted_rs(
    stock_weekly: pd.DataFrame,
    benchmark_weekly: pd.DataFrame,
    symbol: str = "?",
) -> Optional[float]:
    if stock_weekly is None or benchmark_weekly is None:
        return None
    if len(stock_weekly) < 15 or len(benchmark_weekly) < 15:
        return None

    sw = _clean_datetime_index(stock_weekly)
    bw = _clean_datetime_index(benchmark_weekly)

    combined = pd.DataFrame({
        "stock_close": sw["close"],
        "stock_vol": sw["volume"],
        "benchmark_close": bw["close"],
    }).dropna()

    if len(combined) < 15:
        return None

    combined["ratio"] = combined["stock_close"] / combined["benchmark_close"]
    combined["ratio_return"] = combined["ratio"].pct_change()
    combined["vol_sma10"] = combined["stock_vol"].rolling(window=10).mean()
    combined["vol_weight"] = combined["stock_vol"] / combined["vol_sma10"].replace(0, 1)

    recent = combined.iloc[-RS_LOOKBACK_WEEKS:].copy()
    if recent.empty:
        return None

    recent["vw_rs"] = recent["ratio_return"] * recent["vol_weight"]
    score = recent["vw_rs"].sum() * 100
    return round(float(score), 2)


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


def compute_market_stage(market_weekly: Optional[pd.DataFrame]) -> dict:
    if market_weekly is None or len(market_weekly) < MARKET_STAGE_MA_WEEKS + MARKET_STAGE_SLOPE_LOOKBACK_WEEKS:
        return {"stage2": False, "detail": "insufficient market history"}

    df = market_weekly.copy()
    df["stage_ma"] = df["close"].rolling(window=MARKET_STAGE_MA_WEEKS).mean()
    last = df.iloc[-1]
    prev = df.iloc[-1 - MARKET_STAGE_SLOPE_LOOKBACK_WEEKS]

    if pd.isna(last["stage_ma"]) or pd.isna(prev["stage_ma"]):
        return {"stage2": False, "detail": f"{MARKET_STAGE_MA_WEEKS}W MA not available"}

    ma_rising = bool(last["stage_ma"] > prev["stage_ma"])
    price_above_ma = bool(last["close"] > last["stage_ma"])
    stage2 = ma_rising and price_above_ma

    detail = (
        f"{MARKET_INDEX_LABEL} close {last['close']:.1f} "
        f"{'>' if price_above_ma else '<='} {MARKET_STAGE_MA_WEEKS}W MA {last['stage_ma']:.1f} "
        f"({'rising' if ma_rising else 'flat/falling'})"
    )
    return {"stage2": stage2, "detail": detail}


def _base_duration_weeks(df: pd.DataFrame, idx: int, range_pct_max: float = BASE_RANGE_COMPRESSION_PCT) -> int:
    count = 0
    i = idx
    while i >= 0:
        row = df.iloc[i]
        if row.close == 0:
            break
        week_range_pct = (row.high - row.low) / row.close * 100
        if week_range_pct <= range_pct_max:
            count += 1
            i -= 1
        else:
            break
    return count


def _volume_profile(df: pd.DataFrame, idx: int, base_weeks: int):
    window = min(base_weeks, VOL_BASE_LOOKBACK_WEEKS)
    if window < 2 or idx - window < 0:
        return None, None

    base_slice = df["volume"].iloc[idx - window: idx]
    if base_slice.empty or base_slice.mean() == 0:
        return None, None

    breakout_vol_ratio = round(float(df["volume"].iloc[idx] / base_slice.mean()), 2)
    half = max(1, window // 2)
    earlier_half_avg = base_slice.iloc[:half].mean()
    recent_half_avg = base_slice.iloc[half:].mean() if window > half else base_slice.iloc[:half].mean()
    vol_contracting = bool(recent_half_avg < earlier_half_avg)

    return breakout_vol_ratio, vol_contracting


def evaluate_conditions(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 1 or idx >= len(df):
        return None

    row = df.iloc[idx]
    required = ["ema5", "ema10", "ema20", "ema40"]
    if row[required].isna().any():
        return None

    # 1. STRICT SUPPORT RULE: Price must hold AT OR ABOVE 20W EMA (Max 1% buffer)
    # This completely eliminates 40W breakdown drifts like BORORENEW
    holds_20w_support = bool(row.close >= (row.ema20 * 0.99))

    # 2. Strict Distance to Support (10W or 20W EMA)
    dist_ema10 = abs(row.close - row.ema10) / row.close
    dist_ema20 = abs(row.close - row.ema20) / row.close
    near_support = (dist_ema10 <= NEAR_EMA_PCT) or (dist_ema20 <= NEAR_EMA_PCT)

    # 3. EMA Squeeze (5, 10, 20 bundled together tightly)
    ema_spread_pct = (max(row.ema5, row.ema10, row.ema20) - min(row.ema5, row.ema10, row.ema20)) / row.close * 100
    squeeze_ok = ema_spread_pct <= EMA_SPREAD_MAX_PCT

    # 4. Strict Bullish Trend Alignment
    ema40_sloping_up = True
    if idx >= 4:
        ema40_prev = df.iloc[idx - 4]["ema40"]
        ema40_sloping_up = bool(row.ema40 >= ema40_prev)

    uptrend_ok = bool(row.ema10 >= row.ema20 > row.ema40 and row.close > row.ema20 and ema40_sloping_up)

    # 5. STRICT LEADER RULE: Max 15% off 52-Week High (No 25% deep losers)
    high_52w = row.get("high_52w", float("nan"))
    if pd.notna(high_52w) and high_52w > 0:
        dist_from_52w_high_pct = round((high_52w - row.close) / high_52w * 100, 2)
        near_52w_high_ok = dist_from_52w_high_pct <= MAX_PCT_OFF_52W_HIGH
        pct_detail = f"{dist_from_52w_high_pct:.1f}% off 52W high (need <= {MAX_PCT_OFF_52W_HIGH:.0f}%)"
    else:
        dist_from_52w_high_pct = None
        near_52w_high_ok = False
        pct_detail = "52W high n/a"

    # 6. ADX & Base Duration
    adx_val = row.get("adx14", float("nan"))
    adx_ok = bool(pd.notna(adx_val) and adx_val >= ADX_MIN)

    base_weeks = _base_duration_weeks(df, idx)
    base_duration_ok = base_weeks >= MIN_BASE_WEEKS

    # 7. HARD RISK GATE: Base Low to Close <= 7.0%
    start_base_idx = max(0, idx - max(base_weeks, 1))
    base_slice_for_stop = df.iloc[start_base_idx: idx + 1]
    lowest_low = base_slice_for_stop["low"].min()
    stop_loss = round(float(lowest_low) * 0.99, 2)
    risk_pct = round((row.close - stop_loss) / row.close * 100, 2)
    risk_ok = risk_pct <= MAX_RISK_PCT

    checks = {
        "holds_20w_support": (holds_20w_support, f"close {row.close:.2f} >= 20W EMA {row.ema20:.2f}"),
        "near_10w_or_20w_ema": (near_support, f"dist: 10W={dist_ema10*100:.1f}%, 20W={dist_ema20*100:.1f}% (<= {NEAR_EMA_PCT*100:.1f}%)"),
        "ema_squeeze": (squeeze_ok, f"spread: {ema_spread_pct:.2f}% (<= {EMA_SPREAD_MAX_PCT:.1f}%)"),
        "uptrend": (uptrend_ok if UPTREND_REQUIRED else True, f"stack 10W>=20W>40W, 40W rising: {uptrend_ok}"),
        "leader_near_52w_high": (near_52w_high_ok, pct_detail),
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


def _build_scan_result(evald: dict, df: pd.DataFrame, idx: int, market_stage2: bool) -> ScanResult:
    row = evald["row"]
    base_weeks = evald["base_weeks"] or MIN_BASE_WEEKS
    stop_loss = evald["stop_loss"]
    risk_pct = evald["risk_pct"]
    ema_spread_pct = evald["ema_spread_pct"]

    compression_pct = round(ema_spread_pct, 2)
    tightness_label = "Very Tight" if compression_pct < TIGHT_COMPRESSION_PCT else \
                       "Tight" if compression_pct < EMA_SPREAD_MAX_PCT else "Moderate"

    threats = []
    if evald["vol_contracting"] is False:
        threats.append("Volume elevated through base")
    if evald["dist_from_52w_high_pct"] is not None and evald["dist_from_52w_high_pct"] > THREAT_DIST_52W_HIGH_PCT:
        threats.append(f"Deep off high ({evald['dist_from_52w_high_pct']}%)")
    if row.rsi14 is not None and row.rsi14 > THREAT_RSI_EXTENDED:
        threats.append(f"RSI slightly extended ({row.rsi14:.1f})")

    breakout_vol_ratio = evald["breakout_vol_ratio"]
    vol_contracting = evald["vol_contracting"]

    buy_tag = bool(
        evald["monthly_confirmed"]
        and tightness_label in ("Very Tight", "Tight")
        and vol_contracting is True
        and breakout_vol_ratio is not None
        and breakout_vol_ratio >= HIGH_VOL_BREAKOUT_RATIO
        and (market_stage2 or not REQUIRE_MARKET_STAGE2)
    )

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
        compression_pct=compression_pct,
        ema_spread_pct=round(ema_spread_pct, 2),
        week_date=str(row.name.date()),
        monthly_confirmed=evald["monthly_confirmed"],
        dist_from_52w_high_pct=evald["dist_from_52w_high_pct"],
        base_weeks=base_weeks,
        breakout_vol_ratio=breakout_vol_ratio,
        vol_contracting=vol_contracting,
        tightness_label=tightness_label,
        buy_tag=buy_tag,
        stop_loss=stop_loss,
        risk_pct=risk_pct,
        threats=threats,
    )


def check_row_with_reasons(df: pd.DataFrame, idx: int, market_stage2: bool):
    evald = evaluate_conditions(df, idx)
    if evald is None:
        return None, None
    checks = evald["checks"]
    if not all(passed for passed, _ in checks.values()):
        return None, checks
    return _build_scan_result(evald, df, idx, market_stage2), checks


_sector_cache_lock = threading.Lock()
_gate_counter_lock = threading.Lock()


def prefetch_sector_benchmarks(symbols: List[str], sector_cache: dict):
    needed = {SECTOR_BENCHMARK_MAP.get(SECTOR_MAP.get(s), FALLBACK_SECTOR_INDEX_TICKER) for s in symbols}
    needed.add(FALLBACK_SECTOR_INDEX_TICKER)
    for ticker in sorted(needed):
        fetched = _fetch_index_weekly(ticker, period="5y")
        sector_cache[ticker] = fetched


def scan_symbol(symbol: str, sector_cache: dict, market_weekly: Optional[pd.DataFrame],
                market_stage2: bool, gate_counter: dict = None):
    daily = fetch_daily_ohlc(symbol)
    if daily is None:
        return [], None

    weekly = build_weekly(daily)
    if weekly is None:
        return [], None

    monthly_trend = build_monthly_trend(daily)
    weekly = attach_monthly_trend(weekly, monthly_trend)
    weekly = compute_indicators(weekly)
    results = []

    sector = SECTOR_MAP.get(symbol)
    benchmark_ticker = SECTOR_BENCHMARK_MAP.get(sector, FALLBACK_SECTOR_INDEX_TICKER)

    if benchmark_ticker not in sector_cache or sector_cache[benchmark_ticker] is None:
        with _sector_cache_lock:
            if benchmark_ticker not in sector_cache or sector_cache[benchmark_ticker] is None:
                sector_cache[benchmark_ticker] = _fetch_index_weekly(benchmark_ticker, period="5y")

    benchmark_weekly = sector_cache.get(benchmark_ticker)
    if benchmark_weekly is None:
        benchmark_weekly = sector_cache.get(FALLBACK_SECTOR_INDEX_TICKER)

    vw_rs_sector = compute_volume_weighted_rs(weekly, benchmark_weekly, symbol=symbol)
    vw_rs_market = compute_volume_weighted_rs(weekly, market_weekly, symbol=symbol)

    r, checks = check_row_with_reasons(weekly, len(weekly) - 1, market_stage2)
    if gate_counter is not None and checks is not None:
        with _gate_counter_lock:
            gate_counter["_total_scanned"] = gate_counter.get("_total_scanned", 0) + 1
            for check_name, (passed, _) in checks.items():
                if not passed:
                    gate_counter[check_name] = gate_counter.get(check_name, 0) + 1
    if r:
        r.symbol = symbol
        r.sector = sector
        r.rs_vs_sector_pct = vw_rs_sector
        r.rs_vs_market_pct = vw_rs_market
        results.append(r)

    return results, vw_rs_market


TELEGRAM_MAX_CHARS = 4000

def _split_message_into_chunks(text: str, max_chars: int = TELEGRAM_MAX_CHARS) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    blocks = text.split("\n\n")
    chunks, current = [], ""
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
        print("Telegram credentials not set — skipping send.")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = _split_message_into_chunks(text)
    for i, chunk in enumerate(chunks, 1):
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "Markdown"}
        try:
            resp = requests.post(url, data=payload, timeout=10)
            if resp.status_code == 400:
                payload.pop("parse_mode", None)
                resp = requests.post(url, data=payload, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print(f"Telegram send failed on chunk {i}: {e}")


def format_results_message(buy_setups: List[ScanResult], watchlist: List[ScanResult],
                           market_stage_detail: str, rs_cutoff: Optional[float],
                           include_watchlist: bool) -> str:
    def _fmt_entry(r: ScanResult, star: bool) -> str:
        dist_5 = abs(r.close - r.ema5) / r.close * 100
        dist_10 = abs(r.close - r.ema10) / r.close * 100
        dist_20 = abs(r.close - r.ema20) / r.close * 100
        rsi_txt = f"{r.rsi14:.1f}" if r.rsi14 is not None else "n/a"
        adx_txt = f"{r.adx14:.1f}" if r.adx14 is not None else "n/a"
        rs_mkt = f"{r.rs_vs_market_pct:+.1f}%" if r.rs_vs_market_pct is not None else "n/a"
        rs_sec = f"{r.rs_vs_sector_pct:+.1f}%" if r.rs_vs_sector_pct is not None else "n/a"
        sector_txt = f" [{r.sector}]" if r.sector else ""
        prefix = "⭐" if star else "•"

        off_high_txt = f"{r.dist_from_52w_high_pct:.1f}% off 52W high" if r.dist_from_52w_high_pct is not None else "52W high n/a"
        base_txt = f"{r.base_weeks}w base" if r.base_weeks is not None else "base n/a"
        tightness_txt = r.tightness_label or "n/a"
        sl_txt = f"SL: ₹{r.stop_loss} ({r.risk_pct}% risk)" if r.stop_loss is not None else "SL n/a"

        vol_bits = []
        if r.breakout_vol_ratio is not None:
            if r.breakout_vol_ratio >= HIGH_VOL_BREAKOUT_RATIO:
                vol_bits.append(f"🔊 {r.breakout_vol_ratio}x avg vol")
            else:
                vol_bits.append(f"vol {r.breakout_vol_ratio}x avg")
        if r.vol_contracting is True:
            vol_bits.append("contracted into base")
        vol_txt = " | ".join(vol_bits) if vol_bits else "vol profile n/a"

        threats_txt = f"\n  ⚠️ *Threats:* {', '.join(r.threats)}" if r.threats else ""

        return (
            f"{prefix} *{r.symbol}* ({r.week_date}){sector_txt}{' ✅ *BUY SETUP*' if r.buy_tag else ''}\n"
            f"  Close: {r.close} | EMA5: {r.ema5} | EMA10: {r.ema10} | EMA20: {r.ema20}\n"
            f"  Dist: 5W={dist_5:.1f}% | 10W={dist_10:.1f}% | 20W={dist_20:.1f}% | Spread: {r.ema_spread_pct:.2f}%\n"
            f"  RSI: {rsi_txt} | ADX: {adx_txt}\n"
            f"  RS vs {MARKET_INDEX_LABEL}: {rs_mkt} | RS vs Sector: {rs_sec}\n"
            f"  {off_high_txt} | {base_txt} ({tightness_txt})\n"
            f"  🎯 *{sl_txt}* | {vol_txt}{threats_txt}\n"
        )

    lines = [f"*EMA Squeeze Base Scanner (True Leaders)*"]
    lines.append(f"Market: {market_stage_detail}")
    if rs_cutoff is not None:
        lines.append(f"RS gate: top {100 - RS_PERCENTILE_MIN:.0f}% of universe (vw-RS >= {rs_cutoff:+.1f})")
    lines.append("")

    if buy_setups:
        lines.append(f"✅ *BUY SETUPS* — {len(buy_setups)} (all gates + volume-confirmed)\n")
        for r in buy_setups:
            lines.append(_fmt_entry(r, star=True))
    else:
        lines.append("✅ *BUY SETUPS* — none this week.\n")

    if include_watchlist:
        if watchlist:
            lines.append(f"👀 *Watchlist* (holding 10W/20W base, awaiting volume trigger — {len(watchlist)})\n")
            for r in watchlist:
                lines.append(_fmt_entry(r, star=False))
        else:
            lines.append("👀 *Watchlist* — empty.\n")

    return "\n".join(lines)


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
        })

    if not new_rows:
        return

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MATCH_HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)


def main():
    parser = argparse.ArgumentParser(description="EMA Squeeze Base weekly scanner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gate-debug", action="store_true")
    parser.add_argument("--include-watchlist", action="store_true", default=True)
    args = parser.parse_args()

    market_weekly = _fetch_index_weekly(MARKET_INDEX_TICKER, period="5y")
    stage_info = compute_market_stage(market_weekly)
    market_stage2_effective = stage_info["stage2"] or not REQUIRE_MARKET_STAGE2
    print(f"[MARKET] {stage_info['detail']} -> stage2={stage_info['stage2']}")

    symbols = SYMBOLS[: args.limit] if args.limit else SYMBOLS
    print(f"Scanning {len(symbols)} symbols with {args.workers} parallel workers...")

    all_candidates: List[ScanResult] = []
    rs_pool: List[float] = []
    sector_cache: dict = {}
    gate_counter: dict = {} if args.gate_debug else None

    prefetch_sector_benchmarks(symbols, sector_cache)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_symbol = {
            executor.submit(
                scan_symbol, symbol, sector_cache, market_weekly,
                market_stage2_effective, gate_counter,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                results, rs_value = future.result()
            except Exception:
                continue
            if rs_value is not None:
                rs_pool.append(rs_value)
            all_candidates.extend(results)

    if gate_counter is not None:
        total = gate_counter.pop("_total_scanned", 0)
        print(f"\nGATE FAILURE HISTOGRAM (out of {total} symbols)")
        for check_name, fail_count in sorted(gate_counter.items(), key=lambda x: -x[1]):
            pct = round(fail_count / total * 100, 1) if total > 0 else 0.0
            print(f"  {check_name}: failed for {fail_count}/{total} ({pct}%)")

    rs_cutoff = None
    if rs_pool:
        rs_cutoff = float(np.percentile(rs_pool, RS_PERCENTILE_MIN))
        all_candidates = [
            r for r in all_candidates
            if r.rs_vs_market_pct is not None and r.rs_vs_market_pct >= rs_cutoff
        ]
        print(f"[RS] universe n={len(rs_pool)} | top-{100 - RS_PERCENTILE_MIN:.0f}% cutoff = {rs_cutoff:+.2f}")

    buy_setups = [r for r in all_candidates if r.buy_tag]
    watchlist = [r for r in all_candidates if not r.buy_tag]

    print(f"\n{len(buy_setups)} BUY SETUP(s), {len(watchlist)} watchlist candidate(s).")
    message = format_results_message(buy_setups, watchlist, stage_info["detail"], rs_cutoff, args.include_watchlist)
    print(message)

    if buy_setups and not args.dry_run:
        log_matches_to_history(buy_setups)

    if not args.dry_run:
        send_telegram_message(message)


if __name__ == "__main__":
    main()
