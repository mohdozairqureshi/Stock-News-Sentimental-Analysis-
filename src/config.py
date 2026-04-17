from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "report"
FIGURES_DIR = REPORT_DIR / "figures"
ASSETS_DIR = BASE_DIR / "assets"

NEWS_FILE = RAW_DATA_DIR / "Combined_News_DJIA.csv"
MARKET_FILE = RAW_DATA_DIR / "upload_DJIA_table.csv"
YAHOO_HISTORY_FILE = PROCESSED_DATA_DIR / "yahoo_market_history.csv"
FEATURES_FILE = PROCESSED_DATA_DIR / "engineered_features.csv"
MODEL_FILE = MODELS_DIR / "stock_direction_model.joblib"
METRICS_FILE = PROCESSED_DATA_DIR / "model_comparison.csv"
MODEL_INFO_FILE = MODELS_DIR / "model_info.json"
BEST_REPORT_FILE = REPORT_DIR / "best_model_report.txt"
PROJECT_REPORT_FILE = REPORT_DIR / "Final_Project_Report.md"

DEFAULT_MARKET_TICKER = "^DJI"
DEFAULT_LIVE_TICKER = "SPY"
MAX_LIVE_NEWS_ITEMS = 5
