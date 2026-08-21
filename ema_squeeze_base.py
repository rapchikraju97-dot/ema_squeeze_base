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
    "FINPIPE": "Capital 