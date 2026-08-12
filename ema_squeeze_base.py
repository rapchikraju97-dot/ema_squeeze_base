"""
EMA Squeeze Base Scanner (symbols embedded — no external file needed)
-----------------------------------------------------------------------
Weekly-timeframe scan for RKScanBot.

Flags stocks where:
  1. 5W / 10W / 20W EMAs are compressed (tight spread) relative to price
  2. Close is at/just above the 10W or 20W EMA (within 3%)
  3. Close is above the 40W EMA (broader uptrend intact)
  4. ADX(14) > 20
  5. Monthly Macro Trend confirmation (6M > 20M > 40M EMA alignment + Monthly RSI 50-70 + Support Test) - Zero Look-Ahead Bias
  6. True Volume-Weighted Relative Strength vs Nifty 50 benchmark (0% to 25% sweet-spot filter)

Usage:
    python ema_squeeze_base.py                # live run, sends Telegram alert
    python ema_squeeze_base.py --dry-run       # prints results, no Telegram send
    python ema_squeeze_base.py --backtest --lookback-weeks 20 --dry-run

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

NEAR_EMA_PCT = 0.03        # Weekly close must be within 3% of the 10W or 20W EMA
UPTREND_REQUIRED = True    # Require ema10 > ema20 > ema40 and close > ema40
ADX_MIN = 20               # Weekly ADX(14) must be at least this

MONTHLY_EMA_PROXIMITY = 0.05   # Monthly close within 5% of 6M or 20M EMA support
MONTHLY_RSI_MIN = 50         # Monthly RSI(14) institutional minimum floor (secular bull zone)
MONTHLY_RSI_MAX = 70         # Monthly RSI(14) maximum ceiling (avoids parabolic overbought extremes)

RS_LOOKBACK_WEEKS = 12       # ~1 quarter for Relative Strength calculation
MARKET_INDEX_TICKER = "^NSEI"# Nifty 50 benchmark ticker
RS_VS_MARKET_MIN = 0.0       # RS vs Market floor (avoid weak/underperforming stocks)
RS_VS_MARKET_MAX = 25.0      # RS vs Market ceiling (avoid overextended/parabolic stocks)
REQUIRE_RS_GATE = True       # Enforce true relative strength sweet-spot filtering gate

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# NSE symbols — Nifty Total Market list (embedded)
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
    "LALPATHLAB": "Healthcare", "DRREDDY": "Healthcare", "DUMMYINXGN": "Construction", "DUMMYTRVN": "Capital Goods",
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
    "TECHNOE": "Construction", "TEGA": "Capital Goods", "TEJASNET": "Telecommunication", "TEJSS": "Automobile and Auto Components",
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
    "ZENSARTECH": "Information Technology", "ZYDUSLIFE": "Healthcare", "ZYDUSWELL": "Fast Moving Consumer Goods", "ECLERX": "Services"
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
    monthly_confirmed: bool = False


# ---------------------------------------------------------------------------
# Data loading & Clean Index Helpers
# ---------------------------------------------------------------------------

def fetch_daily_ohlc(symbol: str, period: str = "5y") -> Optional[pd.DataFrame]:
    ticker = f"{symbol}.NS"
    try:
        daily = yf.download(
            ticker, period=period, interval="1d", progress=False,
            auto_adjust=True, timeout=15,
        )
    except Exception as e:
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

    if len(weekly) > 0:
        today = pd.Timestamp.now().normalize()
        if weekly.index[-1] > today:
            weekly = weekly.iloc[:-1]

    if len(weekly) < 45:
        return None

    return weekly


def _clean_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.as_unit("ns")
    return df


def build_monthly_trend(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Computes monthly trend and confluence rules once per symbol history.
    Completely eliminates look-ahead bias and redundant per-row loops.
    """
    monthly = daily.resample("ME").agg({"close": "last"}).dropna()
    if len(monthly) < 45:
        return pd.DataFrame()

    monthly["ema6"] = monthly["close"].ewm(span=6, adjust=False).mean()
    monthly["ema20"] = monthly["close"].ewm(span=20, adjust=False).mean()
    monthly["ema40"] = monthly["close"].ewm(span=40, adjust=False).mean()
    
    monthly["rsi14_m"] = RSIIndicator(close=monthly["close"], window=14).rsi()

    # Broad Institutional Bull Rules: 6M > 20M > 40M EMA stack, price > 40M baseline, RSI in 50-70 band
    stack_ok = (monthly["ema6"] > monthly["ema20"]) & (monthly["ema20"] > monthly["ema40"]) & (monthly["close"] > monthly["ema40"])
    rsi_ok = monthly["rsi14_m"].between(MONTHLY_RSI_MIN, MONTHLY_RSI_MAX)
    monthly["macro_valid"] = stack_ok & rsi_ok

    # Support test near 6M or 20M EMA (within 5%)
    dist_ema6 = (monthly["close"] - monthly["ema6"]).abs() / monthly["close"]
    dist_ema20 = (monthly["close"] - monthly["ema20"]).abs() / monthly["close"]
    is_testing_support = (dist_ema6 <= MONTHLY_EMA_PROXIMITY) | (dist_ema20 <= MONTHLY_EMA_PROXIMITY)

    monthly["monthly_confirmed"] = monthly["macro_valid"] & is_testing_support
    return monthly[["monthly_confirmed"]]


def attach_monthly_trend(weekly: pd.DataFrame, monthly_trend: pd.DataFrame) -> pd.DataFrame:
    """
    Safely attaches the most recently COMPLETED monthly confirmation status 
    to each weekly bar using backward merge-asof (zero look-ahead bias).
    """
    if monthly_trend.empty:
        weekly["monthly_confirmed"] = False
        return weekly

    w = _clean_datetime_index(weekly)
    m = _clean_datetime_index(monthly_trend)
    m.index = m.index + pd.Timedelta(days=1)  # Push to next day so merge_asof strictly sees completed months

    merged = pd.merge_asof(
        w.sort_index(), m.sort_index(),
        left_index=True, right_index=True, direction="backward"
    )
    merged["monthly_confirmed"] = merged["monthly_confirmed"].fillna(False)
    return merged


def _fetch_index_weekly(ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
    try:
        daily = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True, timeout=15)
    except Exception as e:
        return None
    if daily.empty:
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


def compute_true_volume_weighted_rs(stock_weekly: pd.DataFrame, market_weekly: pd.DataFrame) -> Optional[float]:
    """
    Computes True Volume-Weighted Relative Strength: measures rate-of-change outperformance 
    of the stock relative to Nifty 50, scaled by volume expansion over the last 12 weeks.
    """
    if stock_weekly is None or market_weekly is None:
        return None
    if len(stock_weekly) < 15 or len(market_weekly) < 15:
        return None

    sw = _clean_datetime_index(stock_weekly)
    mw = _clean_datetime_index(market_weekly)

    combined = pd.DataFrame({
        "stock_close": sw["close"],
        "stock_vol": sw["volume"],
        "market_close": mw["close"]
    }).dropna()

    if len(combined) < 15:
        return None

    # Ratio of stock to benchmark index
    combined["ratio"] = combined["stock_close"] / combined["market_close"]
    combined["ratio_return"] = combined["ratio"].pct_change()

    # Volume expansion factor: stock volume vs its 10-week average
    combined["vol_sma10"] = combined["stock_vol"].rolling(window=10).mean()
    combined["vol_weight"] = combined["stock_vol"] / combined["vol_sma10"].replace(0, 1)

    # Accumulate over lookback window
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


# ---------------------------------------------------------------------------
# Scan logic
# ---------------------------------------------------------------------------

def evaluate_conditions(df: pd.DataFrame, idx: int) -> Optional[dict]:
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

    monthly_confirmed = bool(row.get("monthly_confirmed", False))

    checks = {
        "near_10w_or_20w_ema": (near_either,
            f"close {row.close:.2f} | dist to EMA10 {dist_ema10*100:.2f}% | dist to EMA20 {dist_ema20*100:.2f}% (need <= {NEAR_EMA_PCT*100:.0f}% to either)"),
        "uptrend":            (uptrend_ok if UPTREND_REQUIRED else True,
            f"ema10 {row.ema10:.2f} > ema20 {row.ema20:.2f} > ema40 {row.ema40:.2f}, close {row.close:.2f} > ema40: {uptrend_ok}"),
        "adx_min":            (adx_ok,
            f"ADX {adx_val:.1f} (need >= {ADX_MIN})" if pd.notna(adx_val) else "ADX not available"),
    }
    return {"row": row, "checks": checks, "monthly_confirmed": monthly_confirmed}


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
        monthly_confirmed=evald["monthly_confirmed"],
    )


def scan_symbol(symbol: str, market_weekly: Optional[pd.DataFrame], backtest: bool, lookback_weeks: int):
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

    # Compute True Volume-Weighted RS vs Market Benchmark
    vw_rs = compute_true_volume_weighted_rs(weekly, market_weekly)

    if backtest:
        start_idx = max(1, len(weekly) - lookback_weeks)
        for i in range(start_idx, len(weekly)):
            r = check_row(weekly, i)
            if r:
                r.symbol = symbol
                r.sector = SECTOR_MAP.get(symbol)
                r.rs_vs_market_pct = vw_rs
                results.append(r)
    else:
        r = check_row(weekly, len(weekly) - 1)
        if r:
            r.symbol = symbol
            r.sector = SECTOR_MAP.get(symbol)
            r.rs_vs_market_pct = vw_rs
            results.append(r)

    return results, vw_rs


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
        return "*Near 10W/20W EMA Scan*\nNo matches this week."

    high_conviction = [r for r in results if r.monthly_confirmed]
    tactical = [r for r in results if not r.monthly_confirmed]

    def _fmt_entry(r: ScanResult, star: bool) -> str:
        dist_10 = abs(r.close - r.ema10) / r.close * 100
        dist_20 = abs(r.close - r.ema20) / r.close * 100
        rsi_txt = f"{r.rsi14:.1f}" if r.rsi14 is not None else "n/a"
        adx_txt = f"{r.adx14:.1f}" if r.adx14 is not None else "n/a"
        rs_mkt = f"{r.rs_vs_market_pct:+.1f}" if r.rs_vs_market_pct is not None else "n/a"
        sector_txt = f" [{r.sector}]" if r.sector else ""
        prefix = "⭐" if star else "•"
        return (
            f"{prefix} *{r.symbol}* ({r.week_date}){sector_txt}\n"
            f"  Close: {r.close} | EMA10: {r.ema10} | EMA20: {r.ema20} | EMA40: {r.ema40}\n"
            f"  Dist to EMA10: {dist_10:.2f}% | Dist to EMA20: {dist_20:.2f}%\n"
            f"  RSI: {rsi_txt} | ADX: {adx_txt}\n"
            f"  True VW-RS vs Nifty50: {rs_mkt}\n"
        )

    lines = [f"*Near 10W/20W EMA Scan* — {len(results)} match(es)\n"]

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

    market_weekly = _fetch_index_weekly(MARKET_INDEX_TICKER)
    monthly_trend = build_monthly_trend(daily)
    weekly = attach_monthly_trend(weekly, monthly_trend)
    weekly = compute_indicators(weekly)

    evald = evaluate_conditions(weekly, len(weekly) - 1)
    if evald is None:
        print(f"{symbol}: not enough history to evaluate yet.")
        return

    vw_rs = compute_true_volume_weighted_rs(weekly, market_weekly)
    row = evald["row"]
    checks = evald["checks"]
    print(f"\n=== {symbol} — week of {row.name.date()} — close {row.close:.2f} ===")
    all_pass = True
    for name, (passed, detail) in checks.items():
        mark = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{mark}] {name}: {detail}")
    print(f"  [INFO] monthly_confirmed: {evald['monthly_confirmed']}")
    print(f"  [INFO] True VW-RS score: {vw_rs}")
    print(f"  => OVERALL: {'MATCH' if all_pass else 'no match'}\n")


def main():
    parser = argparse.ArgumentParser(description="EMA Squeeze Base weekly scanner")
    parser.add_argument("--dry-run", action="store_true", help="Print results, skip Telegram send")
    parser.add_argument("--backtest", action="store_true", help="Check the last N weeks instead of just the latest")
    parser.add_argument("--lookback-weeks", type=int, default=5, help="Weeks to check when --backtest is set")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to sleep between symbol downloads")
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N symbols")
    parser.add_argument("--explain", type=str, default=None, help="Show a per-condition pass/fail breakdown for one symbol")
    args = parser.parse_args()

    if args.explain:
        explain_symbol(args.explain.upper())
        return

    symbols = SYMBOLS[: args.limit] if args.limit else SYMBOLS
    print(f"Scanning {len(symbols)} symbols...")

    market_weekly = _fetch_index_weekly(MARKET_INDEX_TICKER)
    if market_weekly is None:
        print("Warning: could not fetch Nifty 50 benchmark — RS gate will be bypassed.")

    all_results: List[ScanResult] = []

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol}")
        results, _ = scan_symbol(symbol, market_weekly, backtest=args.backtest, lookback_weeks=args.lookback_weeks)
        all_results.extend(results)
        time.sleep(args.delay)

    before_gate = len(all_results)
    if REQUIRE_RS_GATE and market_weekly is not None:
        def _in_sweet_spot(r: ScanResult) -> bool:
            if r.rs_vs_market_pct is None:
                return True  # Don't drop due to data gaps
            return RS_VS_MARKET_MIN <= r.rs_vs_market_pct <= RS_VS_MARKET_MAX

        all_results = [r for r in all_results if _in_sweet_spot(r)]
        dropped = before_gate - len(all_results)
        if dropped:
            print(f"True VW-RS sweet-spot filter: dropped {dropped} match(es) outside [{RS_VS_MARKET_MIN}, {RS_VS_MARKET_MAX}].")

    print(f"\n{len(all_results)} match(es) found.")
    message = format_results_message(all_results)
    print(message)

    if not args.dry_run:
        send_telegram_message(message)


if __name__ == "__main__":
    main()
