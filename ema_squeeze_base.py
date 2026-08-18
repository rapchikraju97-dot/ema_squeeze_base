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
    python ema_squeeze_base.py --rs-debug --limit 100 --dry-run
        (prints a reason for every stock where RS vs Sector comes back N/A)
    python ema_squeeze_base.py --weekly-report
        (skips scanning; reads match_history.csv, checks current price vs.
         price-at-match for every logged match in the lookback window, and
         sends a performance/journal digest to Telegram)

    NOTE on GitHub Actions: match_history.csv is written to local disk, which
    does NOT persist between Actions runs on its own. If running via Actions,
    add a step after the scan that commits the file back to the repo, e.g.:
        git config user.email "bot@rkscanbot" && git config user.name "RKScanBot"
        git add match_history.csv && git commit -m "log matches" || true
        git push

Requirements:
    pip install yfinance pandas ta requests

Environment variables (for Telegram):
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
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
MAX_PCT_OFF_52W_HIGH = 25.0  # Minervini Trend Template rule #7: close must be within this % of the
                              # 52-week high. High_52w was already computed but never gated on —
                              # this is the "is the stock actually strong" filter, distinct from
                              # "is it currently paused near its EMAs."
MIN_BASE_WEEKS = 4      # loosened from 8 → 4 (2026-08-17): 8 weeks was rejecting ~most of the
                         # universe (see the --gate-debug histogram from that run). 4 weeks
                         # (~1 month) still requires a real pause, just not as demanding a one.
                         # Revisit upward again once there's enough weekly-report data to tell
                         # whether 4-week bases actually hold up as well as longer ones.
VOL_BASE_LOOKBACK_WEEKS = 6   # window used to judge volume contraction going into the current week
HIGH_VOL_BREAKOUT_RATIO = 1.5  # current week's volume vs. base average, to flag a real breakout push
TIGHT_COMPRESSION_PCT = 1.0    # compression_pct below this = "Very Tight"; used only for labeling
MONTHLY_RETEST_PCT = 0.05   # monthly close must be within 5% of the 6-month EMA to count as a "retest"
RS_LOOKBACK_WEEKS = 12        # ~1 quarter, for RS vs market/sector
# NIFTYMIDSML400.NS (no '^' prefix) was consistently unfetchable via yfinance in testing —
# download(), retries, and the Ticker().history() fallback all failed while every '^'-prefixed
# sector index worked fine. Switched to Nifty 500 (^CRSLDX), which uses the same convention as
# the working sector benchmarks. Trade-off: broad market rather than mid/smallcap-specific, but
# a working broad benchmark beats a broken precise one.
MARKET_INDEX_TICKER = "^CRSLDX"
MARKET_INDEX_LABEL = "Nifty 500"  # used in messages/logs — keep in sync with MARKET_INDEX_TICKER
FALLBACK_SECTOR_INDEX_TICKER = "^CRSLDX"  # same broad-market fallback for sectors without a dedicated index

# --- Retry / concurrency ---
DOWNLOAD_RETRIES = 3
DOWNLOAD_BACKOFF_SECONDS = [1, 3, 8]   # sleep before retry attempt 1, 2, 3
DEFAULT_WORKERS = 8                     # parallel symbol scans (I/O-bound, so threads not processes)
DEFAULT_PER_REQUEST_DELAY = 0.3         # small per-worker delay to avoid hammering Yahoo simultaneously

# --- Match history / weekly report ---
MATCH_HISTORY_CSV = "match_history.csv"
MATCH_HISTORY_FIELDS = [
    "scan_run_date", "symbol", "sector", "week_date", "close_at_match",
    "ema10_at_match", "rs_vs_sector_at_match", "monthly_confirmed", "buy_tag",
]
DEFAULT_REPORT_LOOKBACK_WEEKS = 8
REQUIRE_RS_GATE = True  # Recalibrated after switching benchmark to ^CRSLDX (Nifty 500).
                          # Real run (99 matches): min=-23.72, p25=2.83, median=16.50, p75=39.64,
                          # p90=75.65, max=240.61. Sweet spot 0-76: drops the small negative/weak-RS
                          # tail below p25 and the top long-tail outliers beyond p90.
RS_VS_SECTOR_MIN = 0.0
RS_VS_SECTOR_MAX = 76.0

# Sector -> Yahoo Finance benchmark index ticker. Verified real tickers used where a dedicated
# NSE sector index exists; everything else falls back to Nifty 500 (broad market, not one sector).
SECTOR_BENCHMARK_MAP = {
    "Financial Services": "NIFTY_FIN_SERVICE.NS",  # ^CNXFIN was silently repointed by Yahoo to a
                                                     # different, narrower "NIFTY FINSRV25 50" index
                                                     # and no longer returns usable history — this
                                                     # is the correct current ticker for the broad
                                                     # Nifty Financial Services index.
    "Automobile and Auto Components": "^CNXAUTO",
    "Fast Moving Consumer Goods": "^CNXFMCG",
    "Healthcare": "^CNXPHARMA",
    "Information Technology": "^CNXIT",
    "Metals & Mining": "^CNXMETAL",
    "Oil Gas & Consumable Fuels": "^CNXENERGY",
    "Realty": "^CNXREALTY",
}
RS_VS_MARKET_MIN = -25.0      # Loosened (2026-08-17): the strict 0.0 floor combined with the new
RS_VS_MARKET_MAX = 20.0       # 52W-high + base-duration gates was over-filtering to near zero
                               # matches. This wider band still excludes the worst market laggards
                               # (below -25%) and extreme outliers (above +20%) without requiring
                               # every match to already be beating the market outright. Revisit with
                               # a fresh --dry-run distribution once more weeks of data exist —
                               # this was set to unblock matches, not from a calibrated distribution.

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
    week_date: str
    sector: Optional[str] = None
    rs_vs_market_pct: Optional[float] = None
    rs_vs_sector_pct: Optional[float] = None
    vol_ratio: Optional[float] = None
    vol_weighted_rs: Optional[float] = None
    monthly_confirmed: bool = False
    dist_from_52w_high_pct: Optional[float] = None
    base_weeks: Optional[int] = None
    breakout_vol_ratio: Optional[float] = None   # this week's volume vs. base average
    vol_contracting: Optional[bool] = None        # did volume shrink in the back half of the base
    tightness_label: Optional[str] = None
    buy_tag: bool = False   # strongest-conviction subset — see _build_scan_result for the exact rule


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _download_with_retry(ticker: str, period: str, label: str) -> Optional[pd.DataFrame]:
    """
    Wraps yf.download with retries + backoff, then falls back to the yf.Ticker().history()
    API path if download() still comes back empty. These two paths hit Yahoo slightly
    differently and have been known to diverge for certain tickers (particularly index
    tickers) depending on yfinance version / session/cookie state — one succeeding where
    the other returns empty isn't unusual.
    """
    last_err = None
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            daily = yf.download(
                ticker, period=period, interval="1d", progress=False,
                auto_adjust=True, timeout=15,
            )
            if daily is not None and not daily.empty:
                return daily
            last_err = "download() returned empty DataFrame"
        except Exception as e:
            last_err = f"download() raised: {e}"

        if attempt < DOWNLOAD_RETRIES - 1:
            sleep_for = DOWNLOAD_BACKOFF_SECONDS[min(attempt, len(DOWNLOAD_BACKOFF_SECONDS) - 1)]
            time.sleep(sleep_for)

    # Last resort: try the Ticker().history() path once before giving up entirely.
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True, timeout=15)
        if hist is not None and not hist.empty:
            print(f"  [{label}] download() failed but Ticker().history() succeeded — using that.")
            return hist
        last_err = f"{last_err} | Ticker().history() also returned empty"
    except Exception as e:
        last_err = f"{last_err} | Ticker().history() raised: {e}"

    print(f"  [{label}] download failed after {DOWNLOAD_RETRIES} attempts + history() fallback: {last_err}")
    return None


def _download_with_retry_since(ticker: str, start_date, label: str) -> Optional[pd.DataFrame]:
    """
    Like _download_with_retry, but pulls from a specific start date forward instead of a fixed
    period — used where we only need data since some known point (e.g. a match date), so we're
    not re-downloading years of history just to check the last few weeks.
    """
    last_err = None
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            daily = yf.download(
                ticker, start=start_date, interval="1d", progress=False,
                auto_adjust=True, timeout=15,
            )
            if daily is not None and not daily.empty:
                return daily
            last_err = "download() returned empty DataFrame"
        except Exception as e:
            last_err = f"download() raised: {e}"

        if attempt < DOWNLOAD_RETRIES - 1:
            sleep_for = DOWNLOAD_BACKOFF_SECONDS[min(attempt, len(DOWNLOAD_BACKOFF_SECONDS) - 1)]
            time.sleep(sleep_for)

    try:
        hist = yf.Ticker(ticker).history(start=start_date, interval="1d", auto_adjust=True, timeout=15)
        if hist is not None and not hist.empty:
            return hist
    except Exception as e:
        last_err = f"{last_err} | Ticker().history() raised: {e}"

    print(f"  [{label}] since-fetch failed: {last_err}")
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


def get_latest_close(symbol: str) -> Optional[float]:
    """Cheap fetch of just the most recent daily close — used by the weekly
    performance report so we don't have to pull 5y of history per symbol."""
    ticker = f"{symbol}.NS"
    daily = _download_with_retry(ticker, period="5d", label=symbol)
    if daily is None:
        return None
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    daily.columns = [c.lower() for c in daily.columns]
    if "close" not in daily.columns or daily["close"].dropna().empty:
        return None
    return float(daily["close"].dropna().iloc[-1])


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
    """Like fetch_daily_ohlc but for an index ticker that shouldn't get a .NS suffix."""
    daily = _download_with_retry(ticker, period, label=ticker)
    if daily is None:
        return None
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    daily.columns = [c.lower() for c in daily.columns]
    weekly = daily.resample("W-FRI").agg({"close": "last"}).dropna()
    if len(weekly) > 0:
        today = pd.Timestamp.now().normalize()
        if weekly.index[-1] > today:
            weekly = weekly.iloc[:-1]
    return weekly if len(weekly) > RS_LOOKBACK_WEEKS else None


def compute_period_return(weekly_close: pd.Series, lookback: int = RS_LOOKBACK_WEEKS) -> Optional[float]:
    if weekly_close is None or len(weekly_close) <= lookback:
        return None
    now = weekly_close.iloc[-1]
    then = weekly_close.iloc[-1 - lookback]
    if pd.isna(now) or pd.isna(then) or then == 0:
        return None
    return (now / then - 1) * 100


def compute_volume_weighted_rs(
    stock_weekly: pd.DataFrame,
    benchmark_weekly: pd.DataFrame,
    symbol: str = "?",
    verbose: bool = True,
) -> Optional[float]:
    """
    Volume-weighted relative strength vs a benchmark. Now logs WHY it returns None,
    instead of failing silently — this is the change that fixes the "N/A with no
    explanation" issue.
    """
    if stock_weekly is None:
        if verbose:
            print(f"  [{symbol}] RS=N/A reason: stock_weekly is None")
        return None
    if benchmark_weekly is None:
        if verbose:
            print(f"  [{symbol}] RS=N/A reason: benchmark_weekly is None (fetch failed, incl. fallback)")
        return None
    if len(stock_weekly) < 15 or len(benchmark_weekly) < 15:
        if verbose:
            print(f"  [{symbol}] RS=N/A reason: too few bars "
                  f"(stock={len(stock_weekly)}, bench={len(benchmark_weekly)}, need >=15)")
        return None

    sw = _clean_datetime_index(stock_weekly)
    bw = _clean_datetime_index(benchmark_weekly)

    combined = pd.DataFrame({
        "stock_close": sw["close"],
        "stock_vol": sw["volume"],
        "benchmark_close": bw["close"],
    })
    pre_dropna_len = len(combined)
    combined = combined.dropna()

    if len(combined) < 15:
        if verbose:
            print(f"  [{symbol}] RS=N/A reason: only {len(combined)}/{pre_dropna_len} overlapping "
                  f"weekly dates survived alignment (need >=15). "
                  f"stock range: {sw.index.min().date()}..{sw.index.max().date()} | "
                  f"bench range: {bw.index.min().date()}..{bw.index.max().date()}")
        return None

    combined["ratio"] = combined["stock_close"] / combined["benchmark_close"]
    combined["ratio_return"] = combined["ratio"].pct_change()
    combined["vol_sma10"] = combined["stock_vol"].rolling(window=10).mean()
    combined["vol_weight"] = combined["stock_vol"] / combined["vol_sma10"].replace(0, 1)

    recent = combined.iloc[-RS_LOOKBACK_WEEKS:].copy()
    if recent.empty:
        if verbose:
            print(f"  [{symbol}] RS=N/A reason: recent {RS_LOOKBACK_WEEKS}-week slice empty")
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


# ---------------------------------------------------------------------------
# Scan logic
# ---------------------------------------------------------------------------

def _base_duration_weeks(df: pd.DataFrame, idx: int, near_pct: float = NEAR_EMA_PCT) -> int:
    """
    Counts consecutive weeks, walking backward from idx, where close stayed within near_pct of
    the 5W, 10W, or 20W EMA. This is the "how long has it actually been coiled" check that a
    single near-EMA snapshot can't tell you — a stock that touched the EMA for the first time
    this week and one that's been sitting there for 10 weeks look identical to proximity alone.
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


def _volume_profile(df: pd.DataFrame, idx: int, base_weeks: int):
    """
    Returns (breakout_vol_ratio, vol_contracting) for the current week vs. the base that
    preceded it. breakout_vol_ratio: this week's volume vs. the base's average volume — a
    number well above 1 signals real buying pressure behind the move, not a low-volume drift.
    vol_contracting: whether volume shrank in the more recent half of the base vs. the earlier
    half — genuine accumulation bases tend to go quiet before they break out; sustained high
    volume through the whole base is more often distribution.
    """
    window = min(base_weeks, VOL_BASE_LOOKBACK_WEEKS)
    if window < 2 or idx - window < 0:
        return None, None

    base_slice = df["volume"].iloc[idx - window: idx]  # excludes current (breakout) week
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

    dist_ema5 = abs(row.close - row.ema5) / row.close
    dist_ema10 = abs(row.close - row.ema10) / row.close
    dist_ema20 = abs(row.close - row.ema20) / row.close
    near_ema5 = dist_ema5 <= NEAR_EMA_PCT
    near_ema10 = dist_ema10 <= NEAR_EMA_PCT
    near_ema20 = dist_ema20 <= NEAR_EMA_PCT
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
        # Not enough history yet (needs 52 weeks) — can't confirm strength, so this fails rather
        # than passing by default. A recently-listed stock just doesn't qualify for this check yet.
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


def _build_scan_result(evald: dict) -> ScanResult:
    row = evald["row"]
    dist_ema10 = abs(row.close - row.ema10) / row.close
    dist_ema20 = abs(row.close - row.ema20) / row.close
    compression_pct = round(min(dist_ema10, dist_ema20) * 100, 2)
    tightness_label = "Very Tight" if compression_pct < TIGHT_COMPRESSION_PCT else \
                       "Tight" if compression_pct < NEAR_EMA_PCT * 100 else "Moderate"

    # BUY tag: the strictest-conviction subset among matches that already cleared all 5 gates.
    # Every result here already passed 52W-high, base-duration, uptrend, ADX, and EMA proximity —
    # this tag marks the ones that ALSO have monthly confirmation, a tight (not just "near") base,
    # and a genuinely healthy volume signature (real breakout push + volume that contracted going
    # into the base, not stayed elevated). This is a mechanical rule, not a recommendation — it
    # reflects your own stated criteria, still needs your own chart/judgment before acting on it.
    breakout_vol_ratio = evald["breakout_vol_ratio"]
    vol_contracting = evald["vol_contracting"]
    buy_tag = bool(
        evald["monthly_confirmed"]
        and tightness_label in ("Very Tight", "Tight")
        and vol_contracting is True
        and breakout_vol_ratio is not None
        and breakout_vol_ratio >= HIGH_VOL_BREAKOUT_RATIO
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
        week_date=str(row.name.date()),
        monthly_confirmed=evald["monthly_confirmed"],
        dist_from_52w_high_pct=evald["dist_from_52w_high_pct"],
        base_weeks=evald["base_weeks"],
        breakout_vol_ratio=breakout_vol_ratio,
        vol_contracting=vol_contracting,
        tightness_label=tightness_label,
        buy_tag=buy_tag,
    )


def check_row(df: pd.DataFrame, idx: int) -> Optional[ScanResult]:
    evald = evaluate_conditions(df, idx)
    if evald is None:
        return None
    if not all(passed for passed, _ in evald["checks"].values()):
        return None
    return _build_scan_result(evald)


def check_row_with_reasons(df: pd.DataFrame, idx: int):
    """
    Like check_row, but also returns the checks dict regardless of pass/fail — used to build
    a gate-failure histogram (--gate-debug) so you can see WHICH condition is actually
    rejecting most stocks instead of just getting a silent zero.
    """
    evald = evaluate_conditions(df, idx)
    if evald is None:
        return None, None
    checks = evald["checks"]
    if not all(passed for passed, _ in checks.values()):
        return None, checks
    return _build_scan_result(evald), checks


_sector_cache_lock = threading.Lock()  # guards sector_cache writes when scan_symbol runs across threads


def prefetch_sector_benchmarks(symbols: List[str], sector_cache: dict):
    """
    Fetch every benchmark index needed by this symbol list ONCE, sequentially, before the
    parallel per-symbol scan starts. There are only ~9 distinct benchmark tickers total
    (8 sector indices + the fallback) regardless of how many stocks you scan, so this is
    cheap — and it means scan_symbol never needs to write to sector_cache from multiple
    threads at once in the common case.
    """
    needed = {SECTOR_BENCHMARK_MAP.get(SECTOR_MAP.get(s), FALLBACK_SECTOR_INDEX_TICKER) for s in symbols}
    needed.add(FALLBACK_SECTOR_INDEX_TICKER)
    for ticker in sorted(needed):
        print(f"Prefetching benchmark: {ticker}")
        fetched = _fetch_index_weekly(ticker, period="5y")
        sector_cache[ticker] = fetched
        if fetched is None:
            print(f"  Warning: benchmark '{ticker}' failed to fetch — symbols using it will "
                  f"fall back to {FALLBACK_SECTOR_INDEX_TICKER}.")


_gate_counter_lock = threading.Lock()


def scan_symbol(symbol: str, sector_cache: dict, backtest: bool, lookback_weeks: int,
                 rs_debug: bool = False, per_request_delay: float = 0.0, gate_counter: dict = None):
    """Returns (results, stock_return_pct_or_None). Safe to call from multiple threads
    concurrently — sector_cache is expected to already be populated by
    prefetch_sector_benchmarks(); any fallback writes here are lock-protected.
    If gate_counter is passed (a plain dict), records which check(s) failed for symbols
    that didn't match — powers --gate-debug's end-of-run histogram."""
    if per_request_delay:
        time.sleep(per_request_delay + random.uniform(0, per_request_delay))  # jitter avoids thundering herd

    daily = fetch_daily_ohlc(symbol)
    if daily is None:
        print(f"  [{symbol}] skipped — insufficient data")
        return [], None

    weekly = build_weekly(daily)
    if weekly is None:
        print(f"  [{symbol}] skipped — not enough weekly bars")
        return [], None

    monthly_trend = build_monthly_trend(daily)
    weekly = attach_monthly_trend(weekly, monthly_trend)
    weekly = compute_indicators(weekly)
    results = []

    stock_return = compute_period_return(weekly["close"])

    sector = SECTOR_MAP.get(symbol)
    benchmark_ticker = SECTOR_BENCHMARK_MAP.get(sector, FALLBACK_SECTOR_INDEX_TICKER)

    # Normally already populated by prefetch_sector_benchmarks(). This is just a safety net
    # (e.g. --explain on a single symbol, which skips the prefetch step) — lock-protected
    # since scan_symbol can run concurrently across threads.
    if benchmark_ticker not in sector_cache or sector_cache[benchmark_ticker] is None:
        with _sector_cache_lock:
            if benchmark_ticker not in sector_cache or sector_cache[benchmark_ticker] is None:
                fetched = _fetch_index_weekly(benchmark_ticker, period="5y")
                sector_cache[benchmark_ticker] = fetched
                if fetched is None:
                    print(f"  Warning: benchmark '{benchmark_ticker}' (sector: {sector}) failed to fetch — "
                          f"will fall back to {FALLBACK_SECTOR_INDEX_TICKER} for this symbol.")

    benchmark_weekly = sector_cache.get(benchmark_ticker)

    if benchmark_weekly is None and benchmark_ticker != FALLBACK_SECTOR_INDEX_TICKER:
        with _sector_cache_lock:
            if FALLBACK_SECTOR_INDEX_TICKER not in sector_cache or sector_cache[FALLBACK_SECTOR_INDEX_TICKER] is None:
                sector_cache[FALLBACK_SECTOR_INDEX_TICKER] = _fetch_index_weekly(FALLBACK_SECTOR_INDEX_TICKER, period="5y")
        benchmark_weekly = sector_cache.get(FALLBACK_SECTOR_INDEX_TICKER)

    vw_rs = compute_volume_weighted_rs(weekly, benchmark_weekly, symbol=symbol, verbose=rs_debug)

    if backtest:
        start_idx = max(1, len(weekly) - lookback_weeks)
        for i in range(start_idx, len(weekly)):
            r = check_row(weekly, i)
            if r:
                r.symbol = symbol
                r.sector = sector
                r.rs_vs_sector_pct = vw_rs
                results.append(r)
    else:
        r, checks = check_row_with_reasons(weekly, len(weekly) - 1)
        if gate_counter is not None and checks is not None:
            with _gate_counter_lock:
                gate_counter["_total_scanned"] = gate_counter.get("_total_scanned", 0) + 1
                for check_name, (passed, _) in checks.items():
                    if not passed:
                        gate_counter[check_name] = gate_counter.get(check_name, 0) + 1
        if r:
            r.symbol = symbol
            r.sector = sector
            r.rs_vs_sector_pct = vw_rs
            results.append(r)

    return results, stock_return


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

TELEGRAM_MAX_CHARS = 4000

def _split_message_into_chunks(text: str, max_chars: int = TELEGRAM_MAX_CHARS) -> List[str]:
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
            if resp.status_code == 400:
                print(f"Markdown parse issue on chunk {i} — retrying as plain text.")
                payload.pop("parse_mode", None)
                resp = requests.post(url, data=payload, timeout=10)
            resp.raise_for_status()
            print(f"Telegram chunk {i}/{len(chunks)} sent OK.")
        except Exception as e:
            body = getattr(e, "response", None)
            body_text = body.text if body is not None else ""
            print(f"Telegram send failed on chunk {i}/{len(chunks)}: {e} {body_text}")


def format_results_message(results: List[ScanResult]) -> str:
    if not results:
        return "*Near 5W/10W/20W EMA Scan*\nNo matches this week."

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

        vol_bits = []
        if r.breakout_vol_ratio is not None:
            if r.breakout_vol_ratio >= HIGH_VOL_BREAKOUT_RATIO:
                vol_bits.append(f"🔊 {r.breakout_vol_ratio}x avg vol — real buying pressure")
            else:
                vol_bits.append(f"vol {r.breakout_vol_ratio}x avg")
        if r.vol_contracting is True:
            vol_bits.append("contracted into the base (healthy)")
        elif r.vol_contracting is False:
            vol_bits.append("⚠️ stayed elevated through the base (watch for distribution)")
        vol_txt = " | ".join(vol_bits) if vol_bits else "vol profile n/a"

        return (
            f"{prefix} *{r.symbol}* ({r.week_date}){sector_txt}{' ✅ *BUY SETUP*' if r.buy_tag else ''}\n"
            f"  Close: {r.close} | EMA5: {r.ema5} | EMA10: {r.ema10} | EMA20: {r.ema20} | EMA40: {r.ema40}\n"
            f"  Dist to EMA5: {dist_5:.2f}% | Dist to EMA10: {dist_10:.2f}% | Dist to EMA20: {dist_20:.2f}%\n"
            f"  RSI: {rsi_txt} | ADX: {adx_txt}\n"
            f"  RS vs {MARKET_INDEX_LABEL}: {rs_mkt} | RS vs Sector: {rs_sec}\n"
            f"  {off_high_txt} | {base_txt} ({tightness_txt})\n"
            f"  {vol_txt}\n"
        )

    buy_setups = [r for r in results if r.buy_tag]
    high_conviction = [r for r in results if r.monthly_confirmed and not r.buy_tag]
    tactical = [r for r in results if not r.monthly_confirmed and not r.buy_tag]

    lines = [f"*Near 5W/10W/20W EMA Scan* — {len(results)} match(es)\n"]

    if buy_setups:
        lines.append(
            f"✅ *BUY SETUPS* — {len(buy_setups)}\n"
            f"_(monthly-confirmed + tight base + healthy volume signature — still your call, "
            f"not a recommendation)_\n"
        )
        for r in buy_setups:
            lines.append(_fmt_entry(r, star=True))

    if high_conviction:
        lines.append(f"🔥 *High-Conviction (Weekly + Monthly Confluence)* — {len(high_conviction)}\n")
        for r in high_conviction:
            lines.append(_fmt_entry(r, star=True))

    if tactical:
        lines.append(f"📊 *Tactical (Weekly Setup Only)* — {len(tactical)}\n")
        for r in tactical:
            lines.append(_fmt_entry(r, star=False))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Match history / weekly performance report
# ---------------------------------------------------------------------------

def _ensure_history_schema(csv_path: str = MATCH_HISTORY_CSV):
    """
    Migrates match_history.csv in place if it was written before a field (e.g. buy_tag) existed
    in MATCH_HISTORY_FIELDS. Reads all existing rows, backfills any missing columns with an
    empty value, and rewrites the file with the current header — old rows are preserved, they
    just show blank for the new field(s) rather than breaking the CSV structure.
    """
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        current_fields = reader.fieldnames or []
        rows = list(reader)

    if current_fields == MATCH_HISTORY_FIELDS:
        return  # already up to date

    missing = [f for f in MATCH_HISTORY_FIELDS if f not in current_fields]
    migrated = [{field: row.get(field, "") for field in MATCH_HISTORY_FIELDS} for row in rows]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MATCH_HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(migrated)
    print(f"Migrated {csv_path} schema — added column(s): {missing}")


def log_matches_to_history(results: List[ScanResult], csv_path: str = MATCH_HISTORY_CSV):
    """
    Appends each match to a local CSV for later performance tracking. Dedupes on
    (symbol, week_date) so re-running the same week's scan twice doesn't double-log.
    NOTE: on GitHub Actions this file does not persist across runs unless your workflow
    commits it back to the repo after the scan step — see the module docstring.
    """
    _ensure_history_schema(csv_path)

    existing_keys = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                existing_keys.add((row["symbol"], row["week_date"]))

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
    print(f"Logged {len(new_rows)} new match(es) to {csv_path}.")


MIN_MATURITY_WEEKS = 1.0  # matches held less than this are excluded from all-time stats (too new to mean anything)
STALE_SCAN_WARNING_DAYS = 10  # scans run Mon/Fri, so a gap this long means something's likely broken
MAX_LISTED_PER_SECTION = 10  # cap detailed per-stock listings in the report; rest gets a "+N more" summary


def compute_exit_rule_outcome(symbol: str, match_week_date_str: str, entry_close: float,
                               ema10_at_match: Optional[float],
                               fallback_current_price: Optional[float]):
    """
    Checks whether the stated exit rule — a weekly close below the 10W EMA — would have
    triggered since the match date. Buy-and-hold-to-today (what the rest of the report tracks)
    isn't what the system actually does; this answers "what would this trade have actually
    captured under the real exit rule," which is the honest measure of whether the system works.

    LIGHTWEIGHT approach: EMA is a pure recursive function of (prior EMA, new close) — so instead
    of re-pulling years of history to recompute the 10W EMA from scratch, this seeds the
    recursion from ema10_at_match (already stored in match_history.csv at scan time) and only
    fetches price data from the week AFTER the match forward. For a typical few-weeks-old match
    that's a handful of days fetched, not two years.

    Falls back to the hold-to-today number if the seed is missing (e.g. a pre-migration CSV row)
    or the fetch comes back empty (nothing's happened since the match yet).
    """
    fallback = lambda: {
        "exited": False, "exit_date": None, "exit_close": None,
        "rule_return_pct": round((fallback_current_price / entry_close - 1) * 100, 2) if fallback_current_price else None,
    }

    if not ema10_at_match:
        return fallback()

    match_date = pd.to_datetime(match_week_date_str)
    start = match_date + pd.Timedelta(days=1)
    ticker = f"{symbol}.NS"
    daily = _download_with_retry_since(ticker, start, label=symbol)
    if daily is None or daily.empty:
        return fallback()  # nothing's traded since the match yet, or fetch failed

    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    daily.columns = [c.lower() for c in daily.columns]
    if "close" not in daily.columns:
        return fallback()

    weekly = daily.resample("W-FRI").agg({"close": "last"}).dropna()
    if weekly.empty:
        return fallback()
    today = pd.Timestamp.now().normalize()
    if weekly.index[-1] > today:
        weekly = weekly.iloc[:-1]  # drop an incomplete in-progress week
    if weekly.empty:
        return fallback()

    alpha = 2 / (10 + 1)  # span=10 EMA, matching compute_indicators()
    ema = float(ema10_at_match)
    for date, row in weekly.iterrows():
        close = float(row["close"])
        ema = close * alpha + ema * (1 - alpha)
        if close < ema:
            exit_close = round(close, 2)
            return {
                "exited": True,
                "exit_date": str(date.date()),
                "exit_close": exit_close,
                "rule_return_pct": round((exit_close / entry_close - 1) * 100, 2),
            }

    return fallback()


def generate_weekly_report(lookback_weeks: int = DEFAULT_REPORT_LOOKBACK_WEEKS,
                            csv_path: str = MATCH_HISTORY_CSV, workers: int = DEFAULT_WORKERS) -> str:
    """
    Reads match_history.csv, builds the recent-window digest (last `lookback_weeks`) plus
    an all-time cumulative track record (all matured matches, ever) so you have a running
    scoreboard for whether the system actually has edge, not just a snapshot of this week.
    """
    if not os.path.exists(csv_path):
        return f"*Weekly Performance Report*\nNo match history found at `{csv_path}` yet — nothing to report."

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return "*Weekly Performance Report*\nMatch history is empty — nothing to report."

    # --- Stale-scan check: catches a silently-broken scan schedule (e.g. the cron trigger not
    # firing) instead of the report just quietly showing fewer new matches with no explanation. ---
    stale_warning = ""
    try:
        last_scan_date = max(pd.to_datetime(r["scan_run_date"]) for r in rows if r.get("scan_run_date"))
        days_since_scan = (pd.Timestamp.now().normalize() - last_scan_date).days
        if days_since_scan > STALE_SCAN_WARNING_DAYS:
            stale_warning = (
                f"⚠️ *No new matches logged in {days_since_scan} days* (last: {last_scan_date.date()}). "
                f"The scan workflow may not be running — check the Actions tab.\n\n"
            )
    except (ValueError, KeyError):
        pass

    # Fetch current price for EVERY symbol ever logged, once — powers both the recent-window
    # section and the all-time scoreboard below, so we're not double-fetching.
    unique_symbols = sorted({r["symbol"] for r in rows})
    print(f"Fetching current price for {len(unique_symbols)} symbol(s) for the weekly report...")

    latest_prices = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_symbol = {executor.submit(get_latest_close, sym): sym for sym in unique_symbols}
        for future in as_completed(future_to_symbol):
            sym = future_to_symbol[future]
            try:
                latest_prices[sym] = future.result()
            except Exception as e:
                print(f"  [{sym}] latest-price fetch error: {e}")
                latest_prices[sym] = None

    all_enriched = []
    for r in rows:
        current = latest_prices.get(r["symbol"])
        entry = float(r["close_at_match"])
        pct_return = round((current / entry - 1) * 100, 2) if current else None
        weeks_held = round((pd.Timestamp.now().normalize() - pd.to_datetime(r["week_date"])).days / 7, 1)
        is_buy = str(r.get("buy_tag", "")).strip().lower() in ("true", "1")
        all_enriched.append({**r, "current_price": current, "pct_return": pct_return,
                              "weeks_held": weeks_held, "is_buy": is_buy})

    # Exit-rule enrichment: buy-and-hold-to-today isn't what the system actually does — check
    # every MATURED buy-tagged match, across the FULL history (not just this report's lookback
    # window), against the real exit rule (weekly close < 10W EMA). Runs here, before the window
    # is sliced out, so the all-time scoreboard and the recent-window section both use the same
    # rule-based numbers for buy-tagged stocks. Scoped to buy-tagged + matured only. The check
    # itself is now lightweight (seeded EMA + since-match fetch only, not a full history repull),
    # so this scales fine even as buy-tagged history grows.
    buy_matured_all = [e for e in all_enriched if e["weeks_held"] >= MIN_MATURITY_WEEKS]
    if buy_matured_all:
        print(f"Checking exit rule for {len(buy_matured_all)} matured buy setup(s)...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_entry = {
                executor.submit(
                    compute_exit_rule_outcome, e["symbol"], e["week_date"],
                    float(e["close_at_match"]),
                    float(e["ema10_at_match"]) if e.get("ema10_at_match") not in (None, "") else None,
                    e["current_price"],
                ): e
                for e in buy_matured_all
            }
            for future in as_completed(future_to_entry):
                e = future_to_entry[future]
                try:
                    e["exit_rule"] = future.result()
                except Exception as ex:
                    print(f"  [{e['symbol']}] exit-rule check error: {ex}")
                    e["exit_rule"] = None
    for e in all_enriched:
        e.setdefault("exit_rule", None)

    def _rule_return(e: dict):
        """Effective return for a match: realized/unrealized exit-rule return where computable
        (now checked for every matured match, not just buy-tagged), otherwise falls back to the
        plain hold-to-today return."""
        er = e.get("exit_rule")
        if er and er["rule_return_pct"] is not None:
            return er["rule_return_pct"]
        return e["pct_return"]

    def _scoreboard_for(subset: list, title: str) -> str:
        matured = [e for e in subset if e["pct_return"] is not None and e["weeks_held"] >= MIN_MATURITY_WEEKS]
        if not matured:
            return f"\n{title}\n   No matches have matured yet (need ≥{MIN_MATURITY_WEEKS:g} week held).\n"
        returns_ = [_rule_return(e) for e in matured]
        wins = [x for x in returns_ if x > 0]
        win_rate_ = round(len(wins) / len(returns_) * 100, 1)
        avg_ = round(sum(returns_) / len(returns_), 2)
        median_ = round(pd.Series(returns_).median(), 2)
        best_idx = returns_.index(max(returns_))
        worst_idx = returns_.index(min(returns_))
        return (
            f"\n{title}\n"
            f"   {len(matured)} matured match(es)\n"
            f"   Win rate: {win_rate_}% | Avg return: {avg_:+.2f}% | Median: {median_:+.2f}%\n"
            f"   Best: {matured[best_idx]['symbol']} {returns_[best_idx]:+.2f}% | "
            f"Worst: {matured[worst_idx]['symbol']} {returns_[worst_idx]:+.2f}%\n"
        )

    # --- All-time scoreboards: overall AND buy-tag-only, side by side for comparison ---
    buy_all = [e for e in all_enriched if e["is_buy"]]
    scoreboard = (
        _scoreboard_for(all_enriched, "📊 *All-Time Track Record* (all matches)")
        + _scoreboard_for(buy_all, "✅ *All-Time Track Record — BUY SETUPS ONLY (rule-based return)*")
    )

    # --- Recent-window section (unchanged behavior, just reuses the shared price fetch) ---
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(weeks=lookback_weeks)
    enriched = [e for e in all_enriched if pd.to_datetime(e["week_date"]) >= cutoff]

    if not enriched:
        return (f"*Weekly Performance Report*\nNo matches logged in the last {lookback_weeks} weeks."
                + scoreboard)

    valid = [e for e in enriched if e["pct_return"] is not None]
    no_price = [e for e in enriched if e["pct_return"] is None]
    valid.sort(key=lambda e: e["pct_return"], reverse=True)

    buy_valid = [e for e in valid if e["is_buy"]]
    other_valid = [e for e in valid if not e["is_buy"]]

    winners = [e for e in other_valid if _rule_return(e) > 0]
    losers = [e for e in other_valid if _rule_return(e) <= 0]
    # "Other matches" stats now also use the exit-rule return where computable, same as buy setups.
    other_rule_returns = [_rule_return(e) for e in other_valid]
    win_rate = round(len([x for x in other_rule_returns if x > 0]) / len(other_rule_returns) * 100, 1) if other_rule_returns else 0.0
    avg_return = round(sum(other_rule_returns) / len(other_rule_returns), 2) if other_rule_returns else 0.0
    median_return = round(pd.Series(other_rule_returns).median(), 2) if other_rule_returns else 0.0

    # Buy-setup stats use the RULE-BASED return where available (realized exit, or still-open
    # unrealized if the rule hasn't triggered) rather than a naive hold-to-today number.
    buy_rule_returns = [_rule_return(e) for e in buy_valid]
    buy_win_rate = round(len([x for x in buy_rule_returns if x > 0]) / len(buy_rule_returns) * 100, 1) if buy_rule_returns else 0.0
    buy_avg_return = round(sum(buy_rule_returns) / len(buy_rule_returns), 2) if buy_rule_returns else 0.0

    # Report the ACTUAL age spread of what's in this report, not just the search window —
    # "last 8 weeks" describes how far back we looked, not how long these matches have been
    # live. A report full of same-day matches (weeks_held ~0) has near-zero win-rate signal
    # regardless of what the numbers show, so flag that explicitly instead of implying maturity.
    max_age = max((e["weeks_held"] for e in enriched), default=0)
    min_age = min((e["weeks_held"] for e in enriched), default=0)

    lines = [
        f"*Weekly Performance Report* — searched last {lookback_weeks} weeks of match history\n",
        stale_warning,
        f"Matches span {min_age}–{max_age} week(s) old.\n",
    ]
    if max_age < 1:
        lines.append("⚠️ All matches are <1 week old — win rate/avg return below aren't meaningful yet, "
                      "just noise from measuring too early. Check back after a few more weeks.\n")

    def _fmt_stock_block(e: dict, label: str = "Scanned on") -> str:
        arrow = "🟢" if e["pct_return"] > 0 else "🔴"
        if e["weeks_held"] >= 1:
            held_txt = f"held {e['weeks_held']} week(s)"
        else:
            held_txt = "held <1 week — still forming, don't read into this yet"
        base = (
            f"{arrow} *{e['symbol']}* [{e['sector']}]\n"
            f"   {label}: {e['week_date']}\n"
            f"   Entry price: ₹{e['close_at_match']}\n"
            f"   Current price: ₹{e['current_price']:.2f}\n"
            f"   Move so far (hold-to-today): {e['pct_return']:+.2f}% ({held_txt})\n"
        )
        er = e.get("exit_rule")
        if er:
            if er["exited"]:
                base += (
                    f"   📤 Exit rule triggered {er['exit_date']} @ ₹{er['exit_close']} "
                    f"(weekly close < 10W EMA) — rule-based return: {er['rule_return_pct']:+.2f}%\n"
                )
            else:
                base += "   📈 Exit rule not yet triggered — still holding per the system's own rule\n"
        return base

    def _append_capped(entries: list, label_fn, max_shown: int = MAX_LISTED_PER_SECTION):
        """Lists up to max_shown entries in full detail, then collapses the rest into a
        one-line summary — prevents the report from growing unbounded as match history piles
        up over months (a big lookback window could otherwise list 50-100+ stocks every week)."""
        shown, rest = entries[:max_shown], entries[max_shown:]
        for e in shown:
            lines.append(label_fn(e))
        if rest:
            rest_returns = [_rule_return(e) for e in rest]
            avg_rest = round(sum(rest_returns) / len(rest_returns), 2) if rest_returns else None
            avg_txt = f", avg return {avg_rest:+.2f}%" if avg_rest is not None else ""
            lines.append(f"_...and {len(rest)} more{avg_txt}_\n")

    # --- Dedicated BUY SETUP tracking block — this is the accountability piece ---
    if buy_valid:
        lines.append(
            f"✅ *BUY SETUP TRACKING* ({len(buy_valid)}) — rule-based "
            f"Win rate: {buy_win_rate}% | Avg return: {buy_avg_return:+.2f}%\n"
            f"_(uses realized exit-rule return where matured; hold-to-today for anything too new to check)_\n"
        )
        _append_capped(buy_valid, lambda e: _fmt_stock_block(e, label="Bought on"))

    lines.append(
        f"\n*Other matches:* {len(other_valid)} | Win rate: {win_rate}% | "
        f"Avg return: {avg_return:+.2f}% | Median return: {median_return:+.2f}%\n"
    )
    if abs(avg_return - median_return) >= 5:
        lines.append("_Avg and median diverge a lot — a few outliers are skewing the average; "
                      "median is the more honest read here._\n")

    if winners:
        lines.append(f"\n🟢 *Winners* ({len(winners)}):")
        _append_capped(winners, _fmt_stock_block)

    if losers:
        lines.append(f"🔴 *Losers / flat* ({len(losers)}):")
        _append_capped(sorted(losers, key=lambda e: _rule_return(e)), _fmt_stock_block)  # worst-first

    if no_price:
        lines.append(f"_Could not fetch current price for: {', '.join(e['symbol'] for e in no_price)}_")

    # --- Sector breakdown for this window — flags whether specific sectors are systematically
    # dragging or driving performance, using the same rule-based return as everything else. ---
    sector_groups: dict = {}
    for e in valid:
        sector_groups.setdefault(e.get("sector") or "Unknown", []).append(_rule_return(e))
    if sector_groups:
        sector_stats = []
        for sector, rets in sector_groups.items():
            wr = round(len([x for x in rets if x > 0]) / len(rets) * 100, 1)
            avg = round(sum(rets) / len(rets), 2)
            sector_stats.append((sector, len(rets), wr, avg))
        sector_stats.sort(key=lambda x: -x[3])  # best avg return first
        lines.append("\n📂 *Sector Breakdown* (this window):")
        for sector, n, wr, avg in sector_stats:
            lines.append(f"   {sector}: {n} match(es) | {wr}% win rate | {avg:+.2f}% avg")
        lines.append("")

    lines.append(scoreboard)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def explain_symbol(symbol: str):
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
    sector = SECTOR_MAP.get(symbol)
    benchmark_ticker = SECTOR_BENCHMARK_MAP.get(sector, FALLBACK_SECTOR_INDEX_TICKER)
    benchmark_weekly = _fetch_index_weekly(benchmark_ticker, period="5y")
    if benchmark_weekly is None and benchmark_ticker != FALLBACK_SECTOR_INDEX_TICKER:
        print(f"  '{benchmark_ticker}' failed — trying fallback {FALLBACK_SECTOR_INDEX_TICKER}")
        benchmark_weekly = _fetch_index_weekly(FALLBACK_SECTOR_INDEX_TICKER, period="5y")
    vw_rs = compute_volume_weighted_rs(weekly, benchmark_weekly, symbol=symbol, verbose=True)

    print(f"\n=== {symbol} ({sector}) — week of {row.name.date()} — close {row.close:.2f} ===")
    all_pass = True
    for name, (passed, detail) in checks.items():
        mark = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{mark}] {name}: {detail}")
    print(f"  [INFO] monthly_confirmed (bonus, not gating): {evald['monthly_confirmed']}")
    print(f"  [INFO] VW-RS vs {benchmark_ticker}: {vw_rs}")
    print(f"  => OVERALL: {'MATCH' if all_pass else 'no match'}\n")


def main():
    parser = argparse.ArgumentParser(description="EMA Squeeze Base weekly scanner")
    parser.add_argument("--dry-run", action="store_true", help="Print results, skip Telegram send")
    parser.add_argument("--backtest", action="store_true", help="Check the last N weeks instead of just the latest")
    parser.add_argument("--lookback-weeks", type=int, default=5, help="Weeks to check when --backtest is set")
    parser.add_argument("--delay", type=float, default=DEFAULT_PER_REQUEST_DELAY,
                         help="Base per-worker seconds to sleep before each symbol download (jitter added on top)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                         help="Number of symbols to scan in parallel (I/O-bound; keep modest to avoid Yahoo rate limits)")
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N symbols (useful for quick tests)")
    parser.add_argument("--explain", type=str, default=None,
                         help="Show a per-condition pass/fail breakdown for one symbol (e.g. --explain ZYDUSLIFE) instead of scanning")
    parser.add_argument("--rs-debug", action="store_true",
                         help="Print a reason for every stock where RS vs Sector comes back N/A")
    parser.add_argument("--gate-debug", action="store_true",
                         help="Print a histogram of which condition (52W high, base duration, "
                              "uptrend, ADX, EMA proximity) is rejecting the most stocks this run")
    parser.add_argument("--no-log-history", action="store_true",
                         help="Don't append this run's matches to match_history.csv")
    parser.add_argument("--weekly-report", action="store_true",
                         help="Skip scanning; send a performance/journal digest of past matches to Telegram instead")
    parser.add_argument("--report-lookback-weeks", type=int, default=DEFAULT_REPORT_LOOKBACK_WEEKS,
                         help="How many weeks of match history to include in --weekly-report")
    parser.add_argument("--with-weekly-report", action="store_true",
                         help="After a normal scan, append the performance/journal digest to the SAME "
                              "Telegram message instead of sending it separately")
    args = parser.parse_args()

    if args.explain:
        explain_symbol(args.explain.upper())
        return

    if args.weekly_report:
        message = generate_weekly_report(lookback_weeks=args.report_lookback_weeks, workers=args.workers)
        print(message)
        if not args.dry_run:
            send_telegram_message(message)
        return

    symbols = SYMBOLS[: args.limit] if args.limit else SYMBOLS
    print(f"Scanning {len(symbols)} symbols with {args.workers} parallel workers...")

    market_weekly = _fetch_index_weekly(MARKET_INDEX_TICKER)
    market_return = compute_period_return(market_weekly["close"]) if market_weekly is not None else None
    if market_return is None:
        print(f"Warning: could not fetch {MARKET_INDEX_LABEL} benchmark ({MARKET_INDEX_TICKER}) — "
              f"RS vs market will show as n/a.")

    all_results: List[ScanResult] = []
    sector_cache: dict = {}
    stock_return_map: dict = {}
    all_vw_rs_values: List[float] = []
    gate_counter: dict = {} if args.gate_debug else None

    # Fetch all needed sector/fallback benchmark indices ONCE, sequentially, up front —
    # keeps the parallel stock scan below free of concurrent sector_cache writes.
    prefetch_sector_benchmarks(symbols, sector_cache)

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_symbol = {
            executor.submit(
                scan_symbol, symbol, sector_cache, args.backtest, args.lookback_weeks,
                args.rs_debug, args.delay, gate_counter,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            completed += 1
            try:
                results, stock_return = future.result()
            except Exception as e:
                print(f"  [{symbol}] scan error: {e}")
                continue
            print(f"[{completed}/{len(symbols)}] {symbol}"
                  + (f" — {len(results)} match(es)" if results else ""))
            all_results.extend(results)
            if stock_return is not None:
                stock_return_map[symbol] = stock_return

    if gate_counter is not None:
        total = gate_counter.pop("_total_scanned", 0)
        print("\n" + "=" * 60)
        print(f"GATE FAILURE HISTOGRAM (out of {total} symbols with enough data to evaluate)")
        if total == 0:
            print("  No symbols had enough data to evaluate — check the fetch logs above.")
        else:
            for check_name, fail_count in sorted(gate_counter.items(), key=lambda x: -x[1]):
                pct = round(fail_count / total * 100, 1)
                print(f"  {check_name}: failed for {fail_count}/{total} ({pct}%)")
            print("\nThe check with the highest %% here is your current bottleneck — "
                  "loosen that constant first, not the others, and re-run.")
        print("=" * 60 + "\n")

    all_market_rs_values: List[float] = []
    for r in all_results:
        sr = stock_return_map.get(r.symbol)
        if sr is not None and market_return is not None:
            r.rs_vs_market_pct = round(sr - market_return, 2)
            all_market_rs_values.append(r.rs_vs_market_pct)
        if r.rs_vs_sector_pct is not None:
            all_vw_rs_values.append(r.rs_vs_sector_pct)

    def _print_distribution(label: str, values: List[float], const_names: str):
        if not values:
            print(f"\nNo {label} values computed this run — can't report a distribution yet.\n")
            return
        s = pd.Series(values)
        print("\n" + "=" * 60)
        print(f"{label} DISTRIBUTION (from {len(values)} matched stocks this run)")
        print(f"  min={s.min():.2f}  p25={s.quantile(.25):.2f}  median={s.median():.2f}  "
              f"p75={s.quantile(.75):.2f}  p90={s.quantile(.90):.2f}  max={s.max():.2f}")
        print(f"Use these numbers to set {const_names}.")
        print("=" * 60 + "\n")

    _print_distribution("VW-RS (sector)", all_vw_rs_values, "RS_VS_SECTOR_MIN/MAX")
    _print_distribution("RS vs market", all_market_rs_values, "RS_VS_MARKET_MIN/MAX")

    na_count = sum(1 for r in all_results if r.rs_vs_sector_pct is None)
    if na_count and not args.rs_debug:
        print(f"\n{na_count}/{len(all_results)} matched stocks have RS vs Sector = N/A. "
              f"Re-run with --rs-debug to see the reason for each.\n")

    before_gate = len(all_results)
    if REQUIRE_RS_GATE:
        def _in_sweet_spot(r: ScanResult) -> bool:
            mkt_ok = (r.rs_vs_market_pct is None) or (RS_VS_MARKET_MIN <= r.rs_vs_market_pct <= RS_VS_MARKET_MAX)
            sec_ok = (r.rs_vs_sector_pct is None) or (RS_VS_SECTOR_MIN <= r.rs_vs_sector_pct <= RS_VS_SECTOR_MAX)
            return mkt_ok and sec_ok

        all_results = [r for r in all_results if _in_sweet_spot(r)]
        dropped = before_gate - len(all_results)
        if dropped:
            print(f"RS sweet-spot filter: dropped {dropped} match(es) outside "
                  f"[{RS_VS_MARKET_MIN}%, {RS_VS_MARKET_MAX}%] vs market / "
                  f"[{RS_VS_SECTOR_MIN}%, {RS_VS_SECTOR_MAX}%] vs sector.")

    print(f"\n{len(all_results)} match(es) found.")
    message = format_results_message(all_results)
    print(message)

    # Log final (post-gate) matches for the weekly performance report — skipped for
    # --backtest runs since those replay history rather than reflect a real live match,
    # and would otherwise flood the log with dozens of past weeks per symbol.
    if all_results and not args.backtest and not args.no_log_history:
        log_matches_to_history(all_results)

    if args.with_weekly_report:
        report = generate_weekly_report(lookback_weeks=args.report_lookback_weeks, workers=args.workers)
        print("\n" + report)
        message = message + "\n\n" + ("─" * 20) + "\n\n" + report

    if not args.dry_run:
        send_telegram_message(message)


if __name__ == "__main__":
    main()
