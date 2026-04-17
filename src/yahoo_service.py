import pandas as pd
import yfinance as yf

from src.config import DEFAULT_LIVE_TICKER, MAX_LIVE_NEWS_ITEMS
from src.data_pipeline import clean_headline_text, engineer_market_features


def fetch_recent_news(ticker: str = DEFAULT_LIVE_TICKER, limit: int = MAX_LIVE_NEWS_ITEMS) -> list[dict]:
    items = []
    raw_news = yf.Ticker(ticker).news or []

    for entry in raw_news:
        content = entry.get("content", {})
        title = content.get("title")
        pub_date = content.get("pubDate") or content.get("displayTime")
        if not title:
            continue
        items.append(
            {
                "title": clean_headline_text(title),
                "publisher": content.get("provider", {}).get("displayName", "Yahoo Finance"),
                "published_at": pub_date,
            }
        )
        if len(items) >= limit:
            break

    return items


def fetch_live_feature_row(ticker: str = DEFAULT_LIVE_TICKER) -> dict:
    history = yf.download(
        ticker,
        period="3mo",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if history.empty:
        raise ValueError(f"No Yahoo Finance live market data returned for {ticker}.")

    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)

    history = history.reset_index()
    history["Date"] = pd.to_datetime(history["Date"]).dt.tz_localize(None)
    engineered = engineer_market_features(history)
    return engineered.iloc[-1].to_dict()
