"""
EMA Squeeze Base Scanner v2 — RS-percentile, hard-gated risk/base/ADX,
market-stage aware, with real R-multiple backtest stats.
-----------------------------------------------------------------------
CHANGES FROM v1 (see accompanying notes for the "why"):

  1. RS gate is now PERCENTILE-based against the scanned universe's
     vw_rs distribution, not an uncalibrated absolute band. Only the
     top RS_PERCENTILE_MIN percent of the universe survives.
  2. buy_tag is now a HARD gate on what gets sent to you. Non-buy_tag
     matches are demoted to an optional "Watchlist" section, off by
     default (--include-watchlist to see them).
  3. Risk (stop-loss distance) is now a HARD gate (risk_within_max),
     computed inside evaluate_conditions, not just a post-hoc "threat"
     annotation that gets ignored.
  4. Base duration is redefined as WEEKLY RANGE COMPRESSION (high-low
     range as % of close), not proximity to a fast EMA — a stock can
     hug its 10W EMA while trending hard; that isn't a base. Minimum
     base length raised from 4 to 7 weeks (Weinstein/O'Neil minimum).
  5. ADX floor raised 20 -> 25, and now must also be RISING vs 4 weeks
     ago (not just "a trend exists" but "the trend has force").
  6. NEW: market-regime gate. Computes Weinstein Stage Analysis (price
     vs rising 30W MA) on the Nifty 500. If the market itself is not
     in Stage 2, buy_tag is suppressed scan-wide by default (override
     with --ignore-market-stage).
  7. NEW: --backtest now reports real R-multiple stats (win rate, avg
     R, avg win/loss R, expectancy) using each match's own stop_loss
     as the risk unit and a configurable forward-week window — not
     just raw % return, which hides risk-adjusted quality.

KNOWN SIMPLIFICATIONS (be aware before trusting this fully):
  - RS percentile in --backtest mode is computed once from each
    matched week's own rs value pool, not a true point-in-time weekly
    cross-sectional percentile across history. Good enough to sanity
    check the new gates; not a rigorous walk-forward backtest.
  - R-multiple backtest uses weekly OHLC (checks if the LOW of any
    forward week undercuts the stop) — it does not model intraweek
    stop execution precision, slippage, or gaps.

Usage:
    python ema_squeeze_base_v2.py                       # live run, sends Telegram alert
    python ema_squeeze_base_v2.py --dry-run              # prints results, no Telegram send
    python ema_squeeze_base_v2.py --include-watchlist --dry-run
    python ema_squeeze_base_v2.py --backtest --lookback-weeks 20 --dry-run
    python ema_squeeze_base_v2.py --backtest --lookback-weeks 52 --forward-weeks 8 --dry-run
    python ema_squeeze_base_v2.py --rs-debug --limit 100 --dry-run
    python ema_squeeze_base_v2.py --weekly-report

Requirements:
    pip install yfinance pandas numpy ta requests

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
# Config
# ---------------------------------------------------------------------------

NEAR_EMA_PCT = 0.03     # close must be within 3% of the 5W, 10W, or 20W EMA
EMA_SPREAD_MAX_PCT = 3.0   # max allowed spread between EMA5/EMA10/EMA20, as % of close ("squeeze" gate)
UPTREND_REQUIRED = True  # require ema10 > ema20 > ema40 (bullish stack) and close > ema40

ADX_MIN = 22             # candidate FLOOR — v3: eased 25 -> 22. "Trend has force" is now
                          # enforced via adx_rising as a CONVICTION signal, not stacked
                          # onto the floor as a second hard AND-condition.
ADX_RISING_REQUIRED = False  # v3: no longer blocks candidacy — see "adx_rising" in checks,
                              # which is now informational (still shown, still logged).

MAX_PCT_OFF_52W_HIGH = 25.0  # Minervini Trend Template rule #7

MIN_BASE_WEEKS = 6          # v3: eased 7 -> 6 — 7 was defensible on paper but combined
                             # with 8% range compression it was rejecting almost everything
                             # on volatile mid/smallcap NSE names.
BASE_RANGE_COMPRESSION_PCT = 10.0  # v3: eased 8.0 -> 10.0 for the same reason. A week only
                                    # counts as "in base" if weekly (high-low)/close range
                                    # stays under this — replaces the old EMA-proximity
                                    # definition, which double-counted trending weeks as
                                    # "base" weeks.

VOL_BASE_LOOKBACK_WEEKS = 6   # window used to judge volume contraction going into the current week
HIGH_VOL_BREAKOUT_RATIO = 1.3  # v3: eased 1.5 -> 1.3 for buy_tag volume confirmation —
                                # this is a CONVICTION gate (buy_tag only), not a candidacy gate.
TIGHT_COMPRESSION_PCT = 1.0    # compression_pct below this = "Very Tight"
MONTHLY_RETEST_PCT = 0.05   # monthly close must be within 5% of the 6-month EMA
RS_LOOKBACK_WEEKS = 12        # ~1 quarter, for RS vs market/sector

# --- Risk gate — this one stays HARD and is NOT loosened. Risk-first means this
# is the one gate that doesn't flex for "not enough setups." If nothing clears
# a 7% stop this week, that's the market telling you, not the filter being wrong. ---
MAX_RISK_PCT = 7.0

# --- RS gate: PERCENTILE against the scanned universe. v3: this now only gates
# buy_tag (conviction), not candidacy — watchlist shows the full technical pool
# regardless of RS rank, so a strict RS bar can no longer zero out the whole run. ---
RS_PERCENTILE_MIN = 60.0   # v3: eased 70 -> 60 — top 40% of the scanned universe

# --- Market regime gate — v3: DEMOTED from a hard buy_tag-suppressor to an
# informational threat flag by default. A single macro on/off switch AND'd on
# top of 7+ other conditions was too blunt an instrument — it could zero out
# buy_tag scan-wide regardless of how strong an individual setup was. Re-enable
# strict enforcement with --require-market-stage if you want it back as a gate. ---
REQUIRE_MARKET_STAGE2 = False

# --- Other threat annotations (informational only, not gates) ---
THREAT_DIST_52W_HIGH_PCT = 18.0
THREAT_RSI_EXTENDED = 68

MARKET_INDEX_TICKER = "^CRSLDX"
MARKET_INDEX_LABEL = "Nifty 500"
FALLBACK_SECTOR_INDEX_TICKER = "^CRSLDX"

# --- Retry / concurrency ---
DOWNLOAD_RETRIES = 3
DOWNLOAD_BACKOFF_SECONDS = [1, 3, 8]
DEFAULT_WORKERS = 8
DEFAULT_PER_REQUEST_DELAY = 0.3

# --- Match history / weekly report ---
MATCH_HISTORY_CSV = "match_history.csv"
MATCH_HISTORY_FIELDS = [
    "scan_run_date", "symbol", "sector", "week_date", "close_at_match",
    "ema10_at_match", "rs_vs_sector_at_match", "monthly_confirmed", "buy_tag",
]
DEFAULT_REPORT_LOOKBACK_WEEKS = 8

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _download_with_retry(ticker: str, period: str, label: str) -> Optional[pd.DataFrame]:
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

    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True, timeout=15)
        if hist is not None and not hist.empty:
            return hist
    except Exception as e:
        last_err = f"{last_err} | Ticker().history() raised: {e}"

    print(f"[WARN] {label}: all download attempts failed — {last_err}")
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

    if len(weekly) > 0:
        last_bar_date = weekly.index[-1]
        days_since = (pd.Timestamp.now().normalize() - last_bar_date).days
        if 0 <= days_since < 2 and pd.Timestamp.now().dayofweek < 4:  # before Friday close
            print(f"[WARN] latest weekly bar ({last_bar_date.date()}) looks partial — "
                  f"run after Friday close for stable signals")

    # Need enough history for the MARKET_STAGE_MA_WEEKS-week MA plus lookback margin
    if len(weekly) < max(45, MARKET_STAGE_MA_WEEKS + 10):
        return None

    return weekly


def build_monthly_trend(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Implements RK's actual monthly framework: 6/20/40-month EMA stack.
    A monthly uptrend requires price above ALL THREE EMAs in proper stacked
    order (6 > 20 > 40) — not just 6M > 20M, which is what v1-v3 checked.
    The buy trigger is a pullback to EITHER the 6M or the 20M EMA (RK's own
    wording), not just proximity to the 6M EMA alone.
    """
    monthly = daily.resample("ME").agg({"close": "last"}).dropna()
    monthly["ema6_m"] = monthly["close"].ewm(span=6, adjust=False).mean()
    monthly["ema20_m"] = monthly["close"].ewm(span=20, adjust=False).mean()
    monthly["ema40_m"] = monthly["close"].ewm(span=40, adjust=False).mean()

    stack_ok = (monthly["ema6_m"] > monthly["ema20_m"]) & (monthly["ema20_m"] > monthly["ema40_m"])
    price_above_all = (
        (monthly["close"] > monthly["ema6_m"])
        & (monthly["close"] > monthly["ema20_m"])
        & (monthly["close"] > monthly["ema40_m"])
    )
    monthly["monthly_uptrend"] = stack_ok & price_above_all

    dist_to_ema6_m_pct = (monthly["close"] - monthly["ema6_m"]).abs() / monthly["close"]
    dist_to_ema20_m_pct = (monthly["close"] - monthly["ema20_m"]).abs() / monthly["close"]
    # Pullback trigger = close is near EITHER the 6M or the 20M EMA (RK's stated rule)
    monthly["dist_to_ema6_m_pct"] = dist_to_ema6_m_pct
    monthly["dist_to_ema20_m_pct"] = dist_to_ema20_m_pct
    monthly["monthly_pullback_dist_pct"] = pd.concat(
        [dist_to_ema6_m_pct, dist_to_ema20_m_pct], axis=1
    ).min(axis=1)

    return monthly[[
        "monthly_uptrend", "dist_to_ema6_m_pct", "dist_to_ema20_m_pct",
        "monthly_pullback_dist_pct", "ema6_m", "ema20_m", "ema40_m",
    ]]


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
    merged["dist_to_ema20_m_pct"] = merged["dist_to_ema20_m_pct"].fillna(999.0)
    merged["monthly_pullback_dist_pct"] = merged["monthly_pullback_dist_pct"].fillna(999.0)
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


def compute_volume_weighted_rs_series(
    stock_weekly: pd.DataFrame,
    benchmark_weekly: pd.DataFrame,
    lookback_weeks: int = RS_LOOKBACK_WEEKS,
) -> pd.Series:
    """
    Full rolling vw-RS series, one value per week, indexed by week date.
    This is what makes a true point-in-time backtest possible: for any
    historical week, you can look up what the stock's RS actually was AS OF
    that week — not today's RS retroactively applied to every past match,
    which is what a single-value RS silently does if you reuse it across a
    lookback window.
    """
    if stock_weekly is None or benchmark_weekly is None:
        return pd.Series(dtype=float)
    if len(stock_weekly) < lookback_weeks + 1 or len(benchmark_weekly) < lookback_weeks + 1:
        return pd.Series(dtype=float)

    sw = _clean_datetime_index(stock_weekly)
    bw = _clean_datetime_index(benchmark_weekly)

    combined = pd.DataFrame({
        "stock_close": sw["close"],
        "stock_vol": sw["volume"],
        "benchmark_close": bw["close"],
    }).dropna()

    if len(combined) < lookback_weeks + 1:
        return pd.Series(dtype=float)

    combined["ratio"] = combined["stock_close"] / combined["benchmark_close"]
    combined["ratio_return"] = combined["ratio"].pct_change()
    combined["vol_sma10"] = combined["stock_vol"].rolling(window=10).mean()
    combined["vol_weight"] = combined["stock_vol"] / combined["vol_sma10"].replace(0, 1)
    combined["vw_rs_component"] = combined["ratio_return"] * combined["vol_weight"]

    rs_series = (combined["vw_rs_component"].rolling(window=lookback_weeks).sum() * 100).round(2)
    return rs_series.dropna()


def compute_volume_weighted_rs(
    stock_weekly: pd.DataFrame,
    benchmark_weekly: pd.DataFrame,
    symbol: str = "?",
    verbose: bool = True,
) -> Optional[float]:
    """Latest single vw-RS value — thin wrapper over the series, for the live/non-backtest path."""
    series = compute_volume_weighted_rs_series(stock_weekly, benchmark_weekly)
    if series.empty:
        return None
    return float(series.iloc[-1])


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
# Market regime — RK's own 6/20/40-MONTH EMA stack applied to the Nifty 500,
# not an imported US Weinstein-textbook weekly MA. Same function
# (build_monthly_trend) that gates individual stocks, applied to the index.
# Regime = monthly timeframe (slow, structural). Entries = weekly timeframe
# (build_weekly / evaluate_conditions). That's the correct top-down split:
# monthly context, weekly trigger — not two unrelated indicators.
# ---------------------------------------------------------------------------

def compute_market_stage(index_daily: Optional[pd.DataFrame]) -> dict:
    """
    Market Stage 2 = Nifty 500 monthly close above ALL of its 6M/20M/40M
    EMAs, properly stacked (6M > 20M > 40M) — the exact same rule RK applies
    to individual NSE names, just run on the index itself.
    """
    if index_daily is None or len(index_daily) < 40 * 22:  # ~40 months of trading days, rough floor
        return {"stage2": False, "detail": "insufficient index history for monthly stage analysis"}

    monthly = build_monthly_trend(index_daily)
    if monthly.empty or monthly[["ema6_m", "ema20_m", "ema40_m"]].iloc[-1].isna().any():
        return {"stage2": False, "detail": "6M/20M/40M EMA stack not available yet"}

    last = monthly.iloc[-1]
    stage2 = bool(last["monthly_uptrend"])  # already encodes stack order + price-above-all

    stack_str = f"EMA6M {last['ema6_m']:.1f} / EMA20M {last['ema20_m']:.1f} / EMA40M {last['ema40_m']:.1f}"
    detail = (
        f"{MARKET_INDEX_LABEL} monthly ({last.name.strftime('%Y-%m')}): {stack_str} — "
        f"{'price above full bullish stack (Stage 2)' if stage2 else 'stack not confirmed bullish'}"
    )
    return {"stage2": stage2, "detail": detail}


# ---------------------------------------------------------------------------
# Scan logic
# ---------------------------------------------------------------------------

BASE_LOOKBACK_CAP_WEEKS = 26  # v3 fix: cap how far the base walk-back can go. Without
                               # this, a slow steady grind (small week-to-week range, but
                               # large cumulative drift over months) gets miscounted as one
                               # long "base" — and the stop-loss lookup then reaches back
                               # to an old, irrelevantly-low low. ~6 months is a sane ceiling
                               # for a single basing structure.


def _base_duration_weeks(df: pd.DataFrame, idx: int, range_pct_max: float = BASE_RANGE_COMPRESSION_PCT) -> int:
    """
    A week counts as "in base" if that week's (high-low)/close range stays
    under range_pct_max. This is a genuine consolidation measure — unlike
    proximity to a fast EMA, it doesn't get fooled by a stock that's simply
    trending steadily upward while hugging its own 5W/10W/20W EMA.
    Walk-back is capped at BASE_LOOKBACK_CAP_WEEKS (see note above).
    """
    count = 0
    i = idx
    while i >= 0 and count < BASE_LOOKBACK_CAP_WEEKS:
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

    dist_ema5 = abs(row.close - row.ema5) / row.close
    dist_ema10 = abs(row.close - row.ema10) / row.close
    dist_ema20 = abs(row.close - row.ema20) / row.close
    near_ema5 = dist_ema5 <= NEAR_EMA_PCT
    near_ema10 = dist_ema10 <= NEAR_EMA_PCT
    near_ema20 = dist_ema20 <= NEAR_EMA_PCT
    near_any = near_ema5 or near_ema10 or near_ema20

    ema_spread_pct = (max(row.ema5, row.ema10, row.ema20) - min(row.ema5, row.ema10, row.ema20)) / row.close * 100
    squeeze_ok = ema_spread_pct <= EMA_SPREAD_MAX_PCT

    ema40_sloping_up = True
    if idx >= 4:
        ema40_prev = df.iloc[idx - 4]["ema40"]
        ema40_sloping_up = bool(row.ema40 >= ema40_prev)

    uptrend_ok = bool(row.ema10 > row.ema20 > row.ema40 and row.close > row.ema40 and ema40_sloping_up)

    adx_val = row.get("adx14", float("nan"))
    adx_floor_ok = bool(pd.notna(adx_val) and adx_val >= ADX_MIN)
    adx_rising_ok = True
    if idx >= 4:
        adx_prev = df.iloc[idx - 4].get("adx14", float("nan"))
        adx_rising_ok = bool(pd.notna(adx_prev) and pd.notna(adx_val) and adx_val >= adx_prev)
    # v3: adx_rising is tracked (used later as a buy_tag conviction booster) but
    # only blocks candidacy if ADX_RISING_REQUIRED is explicitly turned back on.
    adx_ok = adx_floor_ok and (adx_rising_ok if ADX_RISING_REQUIRED else True)

    monthly_uptrend = bool(row.get("monthly_uptrend", False))
    monthly_dist = row.get("monthly_pullback_dist_pct", 999.0)
    monthly_retest = bool(pd.notna(monthly_dist) and monthly_dist <= MONTHLY_RETEST_PCT)
    monthly_confirmed = monthly_uptrend and monthly_retest

    # --- 52-week-high gate ---
    high_52w = row.get("high_52w", float("nan"))
    if pd.notna(high_52w) and high_52w > 0:
        dist_from_52w_high_pct = round((high_52w - row.close) / high_52w * 100, 2)
        near_52w_high_ok = dist_from_52w_high_pct <= MAX_PCT_OFF_52W_HIGH
        pct_detail = f"{dist_from_52w_high_pct:.1f}% off 52W high (need <= {MAX_PCT_OFF_52W_HIGH:.0f}%)"
    else:
        dist_from_52w_high_pct = None
        near_52w_high_ok = False
        pct_detail = "52W high not available yet (needs 52 weeks of history)"

    # --- Base duration gate — now range-compression based, see _base_duration_weeks ---
    base_weeks = _base_duration_weeks(df, idx)
    base_duration_ok = base_weeks >= MIN_BASE_WEEKS

    # --- Risk gate — computed here so it's a HARD filter, not a post-hoc tag ---
    start_base_idx = max(0, idx - max(base_weeks, 1))
    base_slice_for_stop = df.iloc[start_base_idx: idx + 1]
    lowest_low = base_slice_for_stop["low"].min()
    stop_loss = round(float(lowest_low) * 0.99, 2)
    risk_pct = round((row.close - stop_loss) / row.close * 100, 2)
    risk_ok = risk_pct <= MAX_RISK_PCT

    checks = {
        "near_5w_10w_or_20w_ema": (near_any,
            f"close {row.close:.2f} | dist to EMA5 {dist_ema5*100:.2f}% | dist to EMA10 {dist_ema10*100:.2f}% | dist to EMA20 {dist_ema20*100:.2f}% (need <= {NEAR_EMA_PCT*100:.0f}% to any)"),
        "ema_squeeze":         (squeeze_ok,
            f"EMA5/10/20 spread {ema_spread_pct:.2f}% of close (need <= {EMA_SPREAD_MAX_PCT:.1f}%)"),
        "uptrend":            (uptrend_ok if UPTREND_REQUIRED else True,
            f"ema10 {row.ema10:.2f} > ema20 {row.ema20:.2f} > ema40 {row.ema40:.2f} (sloping up), close {row.close:.2f} > ema40: {uptrend_ok}"),
        "adx_floor":          (adx_ok,
            (f"ADX {adx_val:.1f} (need >= {ADX_MIN}; rising vs 4w ago: {adx_rising_ok} — informational unless --require-adx-rising)" if pd.notna(adx_val) else "ADX not available")),
        "near_52w_high":      (near_52w_high_ok, pct_detail),
        "min_base_duration":  (base_duration_ok,
            f"range-compressed base held {base_weeks} week(s), weekly range <= {BASE_RANGE_COMPRESSION_PCT:.1f}% of close (need >= {MIN_BASE_WEEKS}w)"),
        "risk_within_max":   (risk_ok,
            f"stop {stop_loss} implies {risk_pct:.2f}% risk (need <= {MAX_RISK_PCT:.1f}%)"),
    }

    breakout_vol_ratio, vol_contracting = _volume_profile(df, idx, base_weeks)

    return {
        "row": row, "checks": checks, "monthly_confirmed": monthly_confirmed,
        "dist_from_52w_high_pct": dist_from_52w_high_pct, "base_weeks": base_weeks,
        "breakout_vol_ratio": breakout_vol_ratio, "vol_contracting": vol_contracting,
        "ema_spread_pct": ema_spread_pct, "stop_loss": stop_loss, "risk_pct": risk_pct,
        "adx_rising": adx_rising_ok,
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

    # --- Threat annotations (informational only — risk_pct itself is already
    # a hard gate above, these are secondary flags worth knowing about) ---
    threats = []
    if evald["vol_contracting"] is False:
        threats.append("Volume elevated through base")
    if evald["dist_from_52w_high_pct"] is not None and evald["dist_from_52w_high_pct"] > THREAT_DIST_52W_HIGH_PCT:
        threats.append(f"Deep off high ({evald['dist_from_52w_high_pct']}%)")
    if row.rsi14 is not None and row.rsi14 > THREAT_RSI_EXTENDED:
        threats.append(f"RSI slightly extended ({row.rsi14:.1f})")
    if not market_stage2:
        threats.append(f"Market ({MARKET_INDEX_LABEL}) not confirmed Stage 2")

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


def check_row(df: pd.DataFrame, idx: int, market_stage2: bool) -> Optional[ScanResult]:
    evald = evaluate_conditions(df, idx)
    if evald is None:
        return None
    if not all(passed for passed, _ in evald["checks"].values()):
        return None
    return _build_scan_result(evald, df, idx, market_stage2)


def check_row_with_reasons(df: pd.DataFrame, idx: int, market_stage2: bool):
    evald = evaluate_conditions(df, idx)
    if evald is None:
        return None, None
    checks = evald["checks"]
    if not all(passed for passed, _ in checks.values()):
        return None, checks
    return _build_scan_result(evald, df, idx, market_stage2), checks


_sector_cache_lock = threading.Lock()


def prefetch_sector_benchmarks(symbols: List[str], sector_cache: dict):
    needed = {SECTOR_BENCHMARK_MAP.get(SECTOR_MAP.get(s), FALLBACK_SECTOR_INDEX_TICKER) for s in symbols}
    needed.add(FALLBACK_SECTOR_INDEX_TICKER)
    for ticker in sorted(needed):
        fetched = _fetch_index_weekly(ticker, period="5y")
        sector_cache[ticker] = fetched


_gate_counter_lock = threading.Lock()
_weekly_cache_lock = threading.Lock()


def scan_symbol(symbol: str, sector_cache: dict, market_weekly: Optional[pd.DataFrame],
                 backtest: bool, lookback_weeks: int, market_stage2: bool,
                 rs_debug: bool = False, per_request_delay: float = 0.0, gate_counter: dict = None,
                 weekly_cache: dict = None):
    """
    Returns (results, latest_rs_vs_market_pct). rs value is returned even
    when the symbol fails other gates — it's needed to build the universe-
    wide RS percentile distribution.
    """
    if per_request_delay:
        time.sleep(per_request_delay + random.uniform(0, per_request_delay))

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

    if weekly_cache is not None:
        with _weekly_cache_lock:
            weekly_cache[symbol] = weekly

    sector = SECTOR_MAP.get(symbol)
    benchmark_ticker = SECTOR_BENCHMARK_MAP.get(sector, FALLBACK_SECTOR_INDEX_TICKER)

    if benchmark_ticker not in sector_cache or sector_cache[benchmark_ticker] is None:
        with _sector_cache_lock:
            if benchmark_ticker not in sector_cache or sector_cache[benchmark_ticker] is None:
                fetched = _fetch_index_weekly(benchmark_ticker, period="5y")
                sector_cache[benchmark_ticker] = fetched

    benchmark_weekly = sector_cache.get(benchmark_ticker)

    if benchmark_weekly is None and benchmark_ticker != FALLBACK_SECTOR_INDEX_TICKER:
        with _sector_cache_lock:
            if FALLBACK_SECTOR_INDEX_TICKER not in sector_cache or sector_cache[FALLBACK_SECTOR_INDEX_TICKER] is None:
                sector_cache[FALLBACK_SECTOR_INDEX_TICKER] = _fetch_index_weekly(FALLBACK_SECTOR_INDEX_TICKER, period="5y")
        benchmark_weekly = sector_cache.get(FALLBACK_SECTOR_INDEX_TICKER)

    vw_rs_sector = compute_volume_weighted_rs(weekly, benchmark_weekly, symbol=symbol, verbose=rs_debug)
    vw_rs_market = compute_volume_weighted_rs(weekly, market_weekly, symbol=symbol, verbose=rs_debug)

    if backtest:
        start_idx = max(1, len(weekly) - lookback_weeks)
        for i in range(start_idx, len(weekly)):
            r = check_row(weekly, i, market_stage2)
            if r:
                r.symbol = symbol
                r.sector = sector
                r.rs_vs_sector_pct = vw_rs_sector
                r.rs_vs_market_pct = vw_rs_market
                results.append(r)
    else:
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


# ---------------------------------------------------------------------------
# Backtest R-multiple stats
# ---------------------------------------------------------------------------

def compute_backtest_r_stats(results: List[ScanResult], weekly_cache: dict, forward_weeks: int = 8) -> dict:
    """
    For each matched setup, walks forward up to `forward_weeks` weekly bars
    and computes an R-multiple: -1R if any forward week's LOW undercuts the
    stop_loss, otherwise (exit_close - entry_close) / (entry_close - stop_loss)
    at the end of the window (or at the last available bar if the window
    isn't fully elapsed yet).

    NOTE: weekly-bar based — doesn't model intraweek stop precision, slippage,
    or gaps. Good for a directional sanity check on the new gates, not a
    substitute for a proper walk-forward/vectorized backtest.
    """
    r_multiples = []
    for r in results:
        weekly = weekly_cache.get(r.symbol)
        if weekly is None:
            continue
        weekly = _clean_datetime_index(weekly)
        try:
            match_date = pd.Timestamp(r.week_date)
        except Exception:
            continue

        future = weekly[weekly.index > match_date]
        if future.empty:
            continue

        window = future.iloc[:forward_weeks]
        if r.stop_loss is None:
            continue
        risk_per_share = r.close - r.stop_loss
        if risk_per_share <= 0:
            continue

        stopped_out = bool((window["low"] <= r.stop_loss).any())
        if stopped_out:
            r_multiples.append(-1.0)
            continue

        exit_close = float(window["close"].iloc[-1])
        r_mult = (exit_close - r.close) / risk_per_share
        r_multiples.append(round(float(r_mult), 2))

    if not r_multiples:
        return {"n": 0}

    wins = [x for x in r_multiples if x > 0]
    losses = [x for x in r_multiples if x <= 0]
    win_rate = len(wins) / len(r_multiples) * 100
    avg_r = sum(r_multiples) / len(r_multiples)
    avg_win_r = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss_r = (sum(losses) / len(losses)) if losses else 0.0
    expectancy = (win_rate / 100 * avg_win_r) + ((1 - win_rate / 100) * avg_loss_r)

    return {
        "n": len(r_multiples),
        "win_rate_pct": round(win_rate, 1),
        "avg_r": round(avg_r, 2),
        "avg_win_r": round(avg_win_r, 2),
        "avg_loss_r": round(avg_loss_r, 2),
        "expectancy_r": round(expectancy, 2),
    }


# ---------------------------------------------------------------------------
# Telegram & Formatting
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
        rs_mkt = f"{r.rs_vs_market_pct:+.1f}" if r.rs_vs_market_pct is not None else "n/a"
        rs_sec = f"{r.rs_vs_sector_pct:+.1f}" if r.rs_vs_sector_pct is not None else "n/a"
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
            f"  Dist: 5W={dist_5:.1f}% | 10W={dist_10:.1f}% | 20W={dist_20:.1f}% | EMA spread: {r.ema_spread_pct:.2f}%\n"
            f"  RSI: {rsi_txt} | ADX: {adx_txt}\n"
            f"  RS vs {MARKET_INDEX_LABEL}: {rs_mkt} | RS vs Sector: {rs_sec}\n"
            f"  {off_high_txt} | {base_txt} ({tightness_txt})\n"
            f"  🎯 *{sl_txt}* | {vol_txt}{threats_txt}\n"
        )

    lines = [f"*EMA Squeeze Scan v2*"]
    lines.append(f"Market: {market_stage_detail}")
    if rs_cutoff is not None:
        lines.append(f"RS gate: top {100 - RS_PERCENTILE_MIN:.0f}% of universe (vw-RS >= {rs_cutoff:+.1f})")
    lines.append("")

    if buy_setups:
        lines.append(f"✅ *BUY SETUPS* — {len(buy_setups)} (all gates + volume-confirmed)\n")
        for r in buy_setups:
            lines.append(_fmt_entry(r, star=True))
    else:
        lines.append("✅ *BUY SETUPS* — none this week. No stock cleared every gate; that's expected most weeks.\n")

    if include_watchlist:
        if watchlist:
            lines.append(f"👀 *Watchlist* (passed technicals, NOT volume-confirmed — {len(watchlist)})\n")
            for r in watchlist:
                lines.append(_fmt_entry(r, star=False))
        else:
            lines.append("👀 *Watchlist* — empty.\n")

    return "\n".join(lines)


def _ensure_history_schema(csv_path: str = MATCH_HISTORY_CSV):
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        current_fields = reader.fieldnames or []
        rows = list(reader)

    if current_fields == MATCH_HISTORY_FIELDS:
        return

    migrated = [{field: row.get(field, "") for field in MATCH_HISTORY_FIELDS} for row in rows]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MATCH_HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(migrated)


def log_matches_to_history(results: List[ScanResult], csv_path: str = MATCH_HISTORY_CSV):
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


def generate_weekly_report(lookback_weeks: int = DEFAULT_REPORT_LOOKBACK_WEEKS,
                           csv_path: str = MATCH_HISTORY_CSV, workers: int = DEFAULT_WORKERS) -> str:
    if not os.path.exists(csv_path):
        return f"*Weekly Performance Report*\nNo match history found at `{csv_path}` yet."

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return "*Weekly Performance Report*\nMatch history is empty."

    unique_symbols = sorted({r["symbol"] for r in rows})
    latest_prices = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_symbol = {executor.submit(get_latest_close, sym): sym for sym in unique_symbols}
        for future in as_completed(future_to_symbol):
            sym = future_to_symbol[future]
            try:
                latest_prices[sym] = future.result()
            except Exception:
                latest_prices[sym] = None

    all_enriched = []
    for r in rows:
        current = latest_prices.get(r["symbol"])
        entry = float(r["close_at_match"])
        pct_return = round((current / entry - 1) * 100, 2) if current else None
        weeks_held = round((pd.Timestamp.now().normalize() - pd.to_datetime(r["week_date"])).days / 7, 1)
        all_enriched.append({**r, "current_price": current, "pct_return": pct_return, "weeks_held": weeks_held})

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(weeks=lookback_weeks)
    enriched = [e for e in all_enriched if pd.to_datetime(e["week_date"]) >= cutoff]
    valid = [e for e in enriched if e["pct_return"] is not None]

    if not valid:
        return f"*Weekly Performance Report*\nNo valid tracked returns in the last {lookback_weeks} weeks."

    valid.sort(key=lambda x: x["pct_return"], reverse=True)
    win_rate = round(len([e for e in valid if e["pct_return"] > 0]) / len(valid) * 100, 1)
    avg_return = round(sum(e["pct_return"] for e in valid) / len(valid), 2)
    median_return = round(pd.Series([e["pct_return"] for e in valid]).median(), 2)

    winners = [e for e in valid if e["pct_return"] > 0]
    losers = [e for e in valid if e["pct_return"] <= 0]
    losers.sort(key=lambda x: x["pct_return"])

    top_winners = winners[:5]
    top_losers = losers[:5]

    lines = [
        f"*Weekly Performance Digest* (searched last {lookback_weeks}w)\n",
        f"📊 *Summary:* {len(valid)} matches tracked",
        f"• *Win Rate:* {win_rate}% _({len(winners)} wins / {len(losers)} losses)_",
        f"• *Avg Return:* {avg_return:+.2f}% | *Median:* {median_return:+.2f}%\n",
        f"🚀 *Top Performers:*",
    ]
    for e in top_winners:
        curr_str = f"₹{e['current_price']:.2f}" if e['current_price'] else "n/a"
        lines.append(f"  • *{e['symbol']}* ({e.get('sector', 'N/A')}) — *{e['pct_return']:+.2f}%* (₹{e['close_at_match']} $\\rightarrow$ {curr_str})")

    if top_losers:
        lines.append(f"\n📉 *Notable Laggards:*")
        for e in top_losers:
            curr_str = f"₹{e['current_price']:.2f}" if e['current_price'] else "n/a"
            lines.append(f"  • *{e['symbol']}* ({e.get('sector', 'N/A')}) — *{e['pct_return']:+.2f}%* (₹{e['close_at_match']} $\\rightarrow$ {curr_str})")

    return "\n".join(lines)


def explain_symbol(symbol: str, market_weekly: Optional[pd.DataFrame], market_stage2: bool):
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
        benchmark_weekly = _fetch_index_weekly(FALLBACK_SECTOR_INDEX_TICKER, period="5y")
    vw_rs_sector = compute_volume_weighted_rs(weekly, benchmark_weekly, symbol=symbol, verbose=True)
    vw_rs_market = compute_volume_weighted_rs(weekly, market_weekly, symbol=symbol, verbose=True)

    print(f"\n=== {symbol} ({sector}) — week of {row.name.date()} — close {row.close:.2f} ===")
    all_pass = True
    for name, (passed, detail) in checks.items():
        mark = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{mark}] {name}: {detail}")
    print(f"  [INFO] monthly_confirmed: {evald['monthly_confirmed']}")
    print(f"  [INFO] VW-RS vs {benchmark_ticker}: {vw_rs_sector}")
    print(f"  [INFO] VW-RS vs {MARKET_INDEX_TICKER}: {vw_rs_market}")
    print(f"  [INFO] market stage2: {market_stage2}")
    print(f"  => OVERALL (technical gates only, RS-percentile applied separately): {'MATCH' if all_pass else 'no match'}\n")


def main():
    parser = argparse.ArgumentParser(description="EMA Squeeze Base weekly scanner v2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--lookback-weeks", type=int, default=5)
    parser.add_argument("--forward-weeks", type=int, default=8, help="Forward window for backtest R-multiple stats")
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
    parser.add_argument("--include-watchlist", action="store_true",
                         help="Also show technically-qualified but NOT volume-confirmed setups")
    parser.add_argument("--ignore-market-stage", action="store_true",
                         help="No-op when --require-market-stage is off (the v3 default). Kept for compatibility.")
    parser.add_argument("--require-market-stage", action="store_true",
                         help="v3 default is OFF (informational only). Pass this to restore the v2 behavior "
                              "of suppressing buy_tag scan-wide when Nifty 500 isn't confirmed Stage 2.")
    parser.add_argument("--require-adx-rising", action="store_true",
                         help="v3 default is OFF. Pass this to require ADX rising vs 4w ago as a hard "
                              "candidacy gate again (on top of the ADX_MIN floor), not just an informational flag.")
    args = parser.parse_args()

    global REQUIRE_MARKET_STAGE2, ADX_RISING_REQUIRED
    if args.require_market_stage:
        REQUIRE_MARKET_STAGE2 = True
    if args.require_adx_rising:
        ADX_RISING_REQUIRED = True

    market_weekly = _fetch_index_weekly(MARKET_INDEX_TICKER, period="5y")
    market_index_daily = _download_with_retry(MARKET_INDEX_TICKER, period="10y", label=MARKET_INDEX_TICKER)
    if market_index_daily is not None:
        if isinstance(market_index_daily.columns, pd.MultiIndex):
            market_index_daily.columns = market_index_daily.columns.get_level_values(0)
        market_index_daily.columns = [c.lower() for c in market_index_daily.columns]
    stage_info = compute_market_stage(market_index_daily)
    market_stage2_effective = stage_info["stage2"] or args.ignore_market_stage or not REQUIRE_MARKET_STAGE2
    print(f"[MARKET] {stage_info['detail']} -> stage2={stage_info['stage2']}"
          f"{' (override active, gate not enforced)' if args.ignore_market_stage else ''}")

    if args.explain:
        explain_symbol(args.explain.upper(), market_weekly, market_stage2_effective)
        return

    if args.weekly_report:
        message = generate_weekly_report(lookback_weeks=args.report_lookback_weeks, workers=args.workers)
        print(message)
        if not args.dry_run:
            send_telegram_message(message)
        return

    symbols = SYMBOLS[: args.limit] if args.limit else SYMBOLS
    print(f"Scanning {len(symbols)} symbols with {args.workers} parallel workers...")

    all_candidates: List[ScanResult] = []
    rs_pool: List[float] = []
    sector_cache: dict = {}
    gate_counter: dict = {} if args.gate_debug else None
    weekly_cache: dict = {} if args.backtest else None

    prefetch_sector_benchmarks(symbols, sector_cache)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_symbol = {
            executor.submit(
                scan_symbol, symbol, sector_cache, market_weekly, args.backtest, args.lookback_weeks,
                market_stage2_effective, args.rs_debug, args.delay, gate_counter, weekly_cache,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                results, rs_value = future.result()
            except Exception as e:
                print(f"[WARN] scan_symbol failed for {symbol}: {e}")
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

    # --- RS percentile gate — v3: this now only decides CONVICTION (buy_tag),
    # not candidacy. The full technically-qualified pool stays in all_candidates
    # (visible via --include-watchlist) so a strict RS bar can no longer zero
    # out the run entirely — it only trims which candidates earn buy_tag. ---
    rs_cutoff = None
    if rs_pool:
        rs_cutoff = float(np.percentile(rs_pool, RS_PERCENTILE_MIN))
        for r in all_candidates:
            rs_ok = r.rs_vs_market_pct is not None and r.rs_vs_market_pct >= rs_cutoff
            if not rs_ok and r.buy_tag:
                r.buy_tag = False
                r.threats.append(f"RS below top-{100 - RS_PERCENTILE_MIN:.0f}% cutoff ({rs_cutoff:+.1f})")
        print(f"[RS] universe n={len(rs_pool)} | top-{100 - RS_PERCENTILE_MIN:.0f}% cutoff = {rs_cutoff:+.2f}")
    else:
        print("[RS] no RS values computed — RS gate skipped this run")

    buy_setups = [r for r in all_candidates if r.buy_tag]
    watchlist = [r for r in all_candidates if not r.buy_tag]

    print(f"\n{len(buy_setups)} BUY SETUP(s), {len(watchlist)} watchlist candidate(s) after RS percentile gate.")

    message = format_results_message(
        buy_setups, watchlist, stage_info["detail"], rs_cutoff, args.include_watchlist
    )
    print(message)

    if args.backtest:
        stats = compute_backtest_r_stats(all_candidates, weekly_cache, forward_weeks=args.forward_weeks)
        print("\n=== BACKTEST R-MULTIPLE STATS (weekly-close approximation) ===")
        if stats.get("n", 0) == 0:
            print("  No trackable matches (need forward weekly bars after the match date).")
        else:
            print(f"  n = {stats['n']}")
            print(f"  Win rate: {stats['win_rate_pct']}%")
            print(f"  Avg R: {stats['avg_r']:+.2f}")
            print(f"  Avg win R: {stats['avg_win_r']:+.2f} | Avg loss R: {stats['avg_loss_r']:+.2f}")
            print(f"  Expectancy per trade: {stats['expectancy_r']:+.2f}R")

    to_log = [r for r in buy_setups]
    if to_log and not args.backtest and not args.no_log_history:
        log_matches_to_history(to_log)

    if args.with_weekly_report:
        report = generate_weekly_report(lookback_weeks=args.report_lookback_weeks, workers=args.workers)
        print("\n" + report)
        message = message + "\n\n" + ("─" * 20) + "\n\n" + report

    if not args.dry_run and not args.backtest:
        send_telegram_message(message)


if __name__ == "__main__":
    main()
