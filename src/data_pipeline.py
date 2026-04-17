import re

import pandas as pd
import yfinance as yf

from src.config import (
    DEFAULT_MARKET_TICKER,
    FEATURES_FILE,
    NEWS_FILE,
    PROCESSED_DATA_DIR,
    YAHOO_HISTORY_FILE,
)


HEADLINE_COLUMNS = [f"Top{i}" for i in range(1, 26)]


def load_news_data() -> pd.DataFrame:
    if not NEWS_FILE.exists():
        raise FileNotFoundError(
            f"News dataset not found: {NEWS_FILE}. Download Combined_News_DJIA.csv into data/raw."
        )

    df = pd.read_csv(NEWS_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def combine_headlines(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()
    working_df[HEADLINE_COLUMNS] = working_df[HEADLINE_COLUMNS].fillna("")
    working_df[HEADLINE_COLUMNS] = working_df[HEADLINE_COLUMNS].apply(
        lambda column: column.map(clean_headline_text)
    )
    working_df["combined_headlines"] = working_df[HEADLINE_COLUMNS].agg(" ".join, axis=1)
    return working_df[["Date", "Label", "combined_headlines"]]


def save_features(df: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_FILE, index=False)


def clean_headline_text(text: str) -> str:
    clean_text = text if isinstance(text, str) else ""
    clean_text = re.sub(r"^b[\"']", "", clean_text)
    clean_text = re.sub(r"[\"']$", "", clean_text)
    clean_text = clean_text.replace("\\n", " ")
    clean_text = clean_text.replace("\\", "")
    clean_text = clean_text.replace("&amp;", "&")
    clean_text = re.sub(r"\s+", " ", clean_text)
    return clean_text.strip()


def fetch_market_history(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    ticker: str = DEFAULT_MARKET_TICKER,
) -> pd.DataFrame:
    start = (pd.to_datetime(start_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end = (pd.to_datetime(end_date) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    history = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if history.empty:
        raise ValueError(f"No Yahoo Finance history returned for {ticker}.")

    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)

    history = history.reset_index()
    history["Date"] = pd.to_datetime(history["Date"]).dt.tz_localize(None)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(YAHOO_HISTORY_FILE, index=False)
    return history


def engineer_market_features(history: pd.DataFrame) -> pd.DataFrame:
    market_df = history.copy().sort_values("Date").reset_index(drop=True)
    market_df["return_1d"] = market_df["Close"].pct_change()
    market_df["return_3d"] = market_df["Close"].pct_change(periods=3)
    market_df["ma_5_ratio"] = market_df["Close"] / market_df["Close"].rolling(5).mean() - 1
    market_df["ma_10_ratio"] = market_df["Close"] / market_df["Close"].rolling(10).mean() - 1
    market_df["volatility_5"] = market_df["Close"].pct_change().rolling(5).std()
    market_df["volume_change"] = market_df["Volume"].pct_change()
    market_df["range_pct"] = (market_df["High"] - market_df["Low"]) / market_df["Close"]
    market_df["Label"] = (market_df["Close"].shift(-1) > market_df["Close"]).astype(int)

    feature_columns = [
        "Date",
        "Label",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "return_1d",
        "return_3d",
        "ma_5_ratio",
        "ma_10_ratio",
        "volatility_5",
        "volume_change",
        "range_pct",
    ]

    market_df = market_df[feature_columns].dropna().reset_index(drop=True)
    return market_df


def merge_news_and_market(news_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(news_df[["Date", "combined_headlines"]], market_df, on="Date", how="inner")
    return merged_df.sort_values("Date").reset_index(drop=True)
