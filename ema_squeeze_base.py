"""
EMA Squeeze Base Scanner with Monthly Confluence & Volume-Weighted RS (RKScanBot)
-----------------------------------------------------------------------
Weekly-timeframe scan integrated with balanced Monthly Macro Trend confirmation 
(6M > 20M > 40M EMA alignment + Monthly RSI 55-63 + Support Test), 
Volume-Weighted RS, and robust Telegram dispatch with strict character-safe 
chunking & Markdown fallback.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
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

MONTHLY_EMA_PROXIMITY = 0.05 # Monthly close within 5% of 6M or 20M EMA support
MONTHLY_RSI_MIN = 55       # Monthly RSI(14) minimum floor
MONTHLY_RSI_MAX = 63       # Monthly RSI(14) maximum ceiling for flexibility

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
    vw_rs_score: float = 0.0
    monthly_confirmed: bool = False
    sector: Optional[str] = None


# ---------------------------------------------------------------------------
# Data loading & Indicators
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


def compute_volume_weighted_rs(weekly_df: pd.DataFrame) -> float:
    """
    Computes a clean Volume-Weighted RS score scaled strictly from 0 to 100.
    Measures accumulation strength over the last 12 weeks.
    """
    if len(weekly_df) < 15:
        return 0.0

    df = weekly_df.copy()
    df["vol_sma10"] = df["volume"].rolling(window=10).mean()
    df["return"] = df["close"].pct_change()
    
    recent = df.iloc[-12:].copy()
    positive_accumulation_weeks = 0.0
    total_weeks = len(recent)
    
    for _, row in recent.iterrows():
        if row["return"] > 0 and row["volume"] > row["vol_sma10"]:
            vol_multiplier = min(row["volume"] / row["vol_sma10"], 2.0)
            positive_accumulation_weeks += (0.5 * vol_multiplier) + 0.5
        elif row["return"] > 0:
            positive_accumulation_weeks += 0.5

    score = (positive_accumulation_weeks / total_weeks) * 100
    return round(min(max(score, 0.0), 100.0), 2)


def check_monthly_confluence(daily: pd.DataFrame) -> bool:
    monthly = daily.resample("ME").agg({
        "close": "last"
    }).dropna()

    if len(monthly) > 0:
        today = pd.Timestamp.now().normalize()
        if monthly.index[-1].month == today.month and monthly.index[-1].year == today.year:
            monthly = monthly.iloc[:-1]

    if len(monthly) < 45:
        return False

    monthly["ema6"] = monthly["close"].ewm(span=6, adjust=False).mean()
    monthly["ema20"] = monthly["close"].ewm(span=20, adjust=False).mean()
    monthly["ema40"] = monthly["close"].ewm(span=40, adjust=False).mean()
    
    monthly["rsi14"] = RSIIndicator(close=monthly["close"], window=14).rsi()
    latest = monthly.iloc[-1]
    
    # Balanced Macro Rules with Flexible RSI range (55 to 63):
    # 1. Structural Stack: 6M EMA > 20M EMA > 40M EMA
    # 2. Price comfortably above the long-term 40M baseline
    # 3. Monthly RSI bounded between 55 and 63
    if not (latest["ema6"] > latest["ema20"] > latest["ema40"] and 
            latest["close"] > latest["ema40"] and 
            pd.notna(latest["rsi14"]) and 
            MONTHLY_RSI_MIN <= latest["rsi14"] <= MONTHLY_RSI_MAX):
        return False

    # Check if price is testing support near either the 6M or 20M EMA (within 5%)
    dist_ema6 = abs(latest["close"] - latest["ema6"]) / latest["close"]
    dist_ema20 = abs(latest["close"] - latest["ema20"]) / latest["close"]
    is_testing_support = (dist_ema6 <= MONTHLY_EMA_PROXIMITY) or (dist_ema20 <= MONTHLY_EMA_PROXIMITY)

    return bool(is_testing_support)


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

    return df


# ---------------------------------------------------------------------------
# Scan Logic
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
    near_either = (dist_ema10 <= NEAR_EMA_PCT) or (dist_ema20 <= NEAR_EMA_PCT)

    uptrend_ok = bool(row.ema10 > row.ema20 > row.ema40 and row.close > row.ema40)
    adx_val = row.get("adx14", float("nan"))
    adx_ok = bool(pd.notna(adx_val) and adx_val >= ADX_MIN)

    checks = {
        "near_10w_or_20w_ema": (near_either, f"dist to EMA10: {dist_ema10*100:.2f}%, EMA20: {dist_ema20*100:.2f}%"),
        "uptrend": (uptrend_ok if UPTREND_REQUIRED else True, f"Uptrend stack active: {uptrend_ok}"),
        "adx_min": (adx_ok, f"ADX: {adx_val:.1f}"),
    }
    return {"row": row, "checks": checks}


def check_row(weekly_df: pd.DataFrame, daily_df: pd.DataFrame, idx: int, symbol: str) -> Optional[ScanResult]:
    evald = evaluate_conditions(weekly_df, idx)
    if evald is None:
        return None

    row = evald["row"]
    checks = evald["checks"]
    if not all(passed for passed, _ in checks.values()):
        return None

    dist_ema10 = abs(row.close - row.ema10) / row.close
    dist_ema20 = abs(row.close - row.ema20) / row.close
    
    monthly_confirmed = check_monthly_confluence(daily_df)
    vwrs_score = compute_volume_weighted_rs(weekly_df)

    return ScanResult(
        symbol=symbol,
        close=round(row.close, 2),
        ema5=round(row.ema5, 2),
        ema10=round(row.ema10, 2),
        ema20=round(row.ema20, 2),
        ema40=round(row.ema40, 2),
        rsi14=round(row.rsi14, 2) if pd.notna(row.get("rsi14")) else 0.0,
        adx14=round(row.adx14, 2) if pd.notna(row.get("adx14")) else 0.0,
        pdi14=round(row.pdi14, 2) if pd.notna(row.get("pdi14")) else 0.0,
        ndi14=round(row.ndi14, 2) if pd.notna(row.get("ndi14")) else 0.0,
        compression_pct=round(min(dist_ema10, dist_ema20) * 100, 2),
        week_date=str(row.name.date()),
        vw_rs_score=vwrs_score,
        monthly_confirmed=monthly_confirmed,
        sector=SECTOR_MAP.get(symbol, "General")
    )


def scan_symbol(symbol: str, backtest: bool, lookback_weeks: int) -> List[ScanResult]:
    daily = fetch_daily_ohlc(symbol)
    if daily is None:
        return []

    weekly = build_weekly(daily)
    if weekly is None:
        return []

    weekly = compute_indicators(weekly)
    results = []

    if backtest:
        start_idx = max(1, len(weekly) - lookback_weeks)
        for i in range(start_idx, len(weekly)):
            r = check_row(weekly, daily, i, symbol)
            if r:
                results.append(r)
    else:
        r = check_row(weekly, daily, len(weekly) - 1, symbol)
        if r:
            results.append(r)

    return results


# ---------------------------------------------------------------------------
# Telegram Formatting & Dispatch
# ---------------------------------------------------------------------------

def _split_message_into_chunks(text: str, max_chars: int = 3800) -> List[str]:
    """Strictly split message text into chunks safely under Telegram's per-message character limit."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current_chunk = ""
    lines = text.split("\n")
    
    for line in lines:
        if len(line) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            for i in range(0, len(line), max_chars):
                chunks.append(line[i:i + max_chars])
            continue

        candidate = (current_chunk + "\n" + line) if current_chunk else line
        if len(candidate) > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            current_chunk = candidate
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return [c for c in chunks if c]


def format_results_message(results: List[ScanResult]) -> str:
    if not results:
        return "*EMA Squeeze Base Scan*\nNo matches this week."

    high_conviction = [r for r in results if r.monthly_confirmed]
    tactical = [r for r in results if not r.monthly_confirmed]

    lines = [f"*EMA Squeeze Base Scan* — {len(results)} total match(es)\n"]

    if high_conviction:
        lines.append(f"🔥 *High-Conviction Tier (Weekly + Monthly Confluence)*: {len(high_conviction)}")
        for r in high_conviction:
            lines.append(
                f"⭐ *{r.symbol}* ({r.week_date}) [{r.sector}]\n"
                f"  Close: {r.close} | VW-RS: {r.vw_rs_score} | Squeeze: {r.compression_pct}%\n"
                f"  RSI: {r.rsi14:.1f} | ADX: {r.adx14:.1f}\n"
            )
        lines.append("")

    if tactical:
        lines.append(f"📊 *Tactical Tier (Weekly Setup Only)*: {len(tactical)}")
        for r in tactical:
            lines.append(
                f"• *{r.symbol}* ({r.week_date}) [{r.sector}] — Close: {r.close} | VW-RS: {r.vw_rs_score}"
            )

    return "\n".join(lines)


def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing. Printing output locally:")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = _split_message_into_chunks(text)
    
    for i, chunk in enumerate(chunks, 1):
        payload = {
            "chat_id": TELEGRAM_CHAT_ID, 
            "text": chunk, 
            "parse_mode": "Markdown"
        }
        try:
            resp = requests.post(url, data=payload, timeout=10)
            
            if resp.status_code == 400:
                print(f"Markdown parse warning on chunk {i}. Retrying as plain text...")
                payload.pop("parse_mode")
                resp = requests.post(url, data=payload, timeout=10)

            resp.raise_for_status()
            print(f"Telegram chunk {i}/{len(chunks)} sent OK.")
        except Exception as e:
            body = getattr(e, "response", None)
            body_text = body.text if body is not None else ""
            print(f"Telegram send failed on chunk {i}/{len(chunks)}: {e} | Response: {body_text}")


# ---------------------------------------------------------------------------
# Main Execution Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EMA Squeeze Base Weekly + Monthly Confluence Scanner")
    parser.add_argument("--dry-run", action="store_true", help="Print results to console, skip Telegram send")
    parser.add_argument("--backtest", action="store_true", help="Check historical lookback periods")
    parser.add_argument("--lookback-weeks", type=int, default=5, help="Lookback window size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of symbols to scan for testing")
    args = parser.parse_args()

    symbols = SYMBOLS[: args.limit] if args.limit else SYMBOLS
    print(f"Scanning {len(symbols)} symbols with Monthly RSI (55-63) Confluence & VW-RS...")

    all_results: List[ScanResult] = []
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Checking {symbol}...")
        results = scan_symbol(symbol, backtest=args.backtest, lookback_weeks=args.lookback_weeks)
        all_results.extend(results)
        time.sleep(0.3)

    print(f"\nScan complete. Total matches found: {len(all_results)}")
    message = format_results_message(all_results)
    print("\n" + message)

    if not args.dry_run:
        send_telegram_message(message)


if __name__ == "__main__":
    main()
