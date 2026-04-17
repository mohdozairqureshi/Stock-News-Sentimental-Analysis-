import base64
import json

import joblib
import nltk
import pandas as pd
import streamlit as st

from src.config import (
    ASSETS_DIR,
    DEFAULT_LIVE_TICKER,
    MAX_LIVE_NEWS_ITEMS,
    MODEL_FILE,
    MODEL_INFO_FILE,
)
from src.sentiment import build_sentiment_analyzer, score_text
from src.yahoo_service import fetch_live_feature_row, fetch_recent_news


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def load_background_image() -> str:
    background_path = ASSETS_DIR / "market_background.svg"
    encoded = base64.b64encode(background_path.read_bytes()).decode("utf-8")
    return encoded


def load_model_info() -> dict:
    if MODEL_INFO_FILE.exists():
        return json.loads(MODEL_INFO_FILE.read_text(encoding="utf-8"))
    return {"best_model": "Not trained", "accuracy": None}


def predict_headline_bundle(text: str, live_features: dict):
    nltk.download("vader_lexicon", quiet=True)
    analyzer = build_sentiment_analyzer()
    scores = score_text(text, analyzer)

    model_row = {
        "combined_headlines": text,
        "neg": scores["neg"],
        "neu": scores["neu"],
        "pos": scores["pos"],
        "compound": scores["compound"],
        "return_1d": live_features["return_1d"],
        "return_3d": live_features["return_3d"],
        "ma_5_ratio": live_features["ma_5_ratio"],
        "ma_10_ratio": live_features["ma_10_ratio"],
        "volatility_5": live_features["volatility_5"],
        "volume_change": live_features["volume_change"],
        "range_pct": live_features["range_pct"],
    }

    model_input = pd.DataFrame([model_row])

    model = load_model()
    prediction = model.predict(model_input)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(model_input)[0][1])
    return int(prediction), scores, probability


st.set_page_config(page_title="Stock Trend Predictor", page_icon="📈", layout="wide")

background_base64 = load_background_image()
app_css = """
<style>
.stApp {
    background:
        linear-gradient(rgba(4, 13, 23, 0.64), rgba(4, 13, 23, 0.82)),
        url("data:image/svg+xml;base64,__BACKGROUND__") center/cover fixed no-repeat;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}
[data-testid="stToolbar"] {
    right: 1rem;
}
[data-testid="stSidebar"] {
    background: rgba(7, 18, 28, 0.72);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero {
    padding: 1.5rem 1.7rem;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(255, 230, 163, 0.16) 0%, rgba(71, 241, 188, 0.10) 100%);
    color: #f5f8f7;
    border: 1px solid rgba(255, 255, 255, 0.14);
    backdrop-filter: blur(10px);
    box-shadow: 0 20px 60px rgba(2, 8, 16, 0.34);
    margin-bottom: 1.1rem;
}
.hero h1 {
    margin-bottom: 0.3rem;
    letter-spacing: 0.02em;
}
.hero p {
    color: #d8efe9;
    font-size: 1.02rem;
}
.glass-card {
    background: rgba(6, 20, 31, 0.58);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 24px;
    padding: 1rem 1.1rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 16px 50px rgba(1, 9, 18, 0.24);
}
.glass-card h3 {
    color: #fff3ce;
    margin-top: 0;
}
.news-pill {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    margin-bottom: 0.55rem;
    border-radius: 999px;
    background: rgba(103, 247, 177, 0.15);
    color: #d6fff0;
    font-size: 0.82rem;
    border: 1px solid rgba(103, 247, 177, 0.18);
}
.news-item {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.07);
    padding: 0.85rem 0.95rem;
    border-radius: 18px;
    color: #eff8f5;
    margin-bottom: 0.6rem;
}
.news-item small {
    color: #b9d4cf;
}
div[data-testid="stMetric"] {
    background: rgba(8, 25, 39, 0.54);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 18px;
    padding: 0.7rem 0.85rem;
    box-shadow: 0 10px 35px rgba(1, 9, 18, 0.2);
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #f7eecf;
}
.stTextInput > div > div,
.stTextArea textarea,
.stSelectbox [data-baseweb="select"] > div {
    background: rgba(9, 24, 36, 0.75);
    color: #f6faf8;
    border-radius: 16px;
}
.stButton button {
    background: linear-gradient(135deg, #ffcf6e 0%, #48e7b6 100%);
    color: #0b1721;
    border: none;
    border-radius: 999px;
    padding: 0.6rem 1.4rem;
    font-weight: 700;
    box-shadow: 0 12px 30px rgba(72, 231, 182, 0.22);
}
.stButton button:hover {
    color: #09131d;
    border: none;
}
.caption-note {
    color: #d5e7e0;
    font-size: 0.9rem;
}
</style>
""".replace("__BACKGROUND__", background_base64)

st.markdown(
    app_css,
    unsafe_allow_html=True,
)

model_info = load_model_info()

st.markdown(
    """
    <div class="hero">
        <h1>Stock Trend Prediction from News Headlines</h1>
        <p>Academic demo with Yahoo Finance data, sentiment scoring, and machine learning to estimate whether the market may move Up or Down.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
col1.metric("Best Model", model_info.get("best_model", "Unknown"))
accuracy_value = model_info.get("accuracy")
col2.metric("Baseline Accuracy", f"{accuracy_value:.2%}" if accuracy_value is not None else "N/A")
col3.metric("Live News Limit", str(MAX_LIVE_NEWS_ITEMS))

st.write("Use up to 5 recent Yahoo Finance news articles or paste your own headlines.")

ticker = st.text_input("Ticker for live Yahoo Finance news", value=DEFAULT_LIVE_TICKER)

news_items = []
headline_default = ""
fetch_error = None

try:
    news_items = fetch_recent_news(ticker=ticker, limit=MAX_LIVE_NEWS_ITEMS)
    headline_default = "\n".join(item["title"] for item in news_items)
except Exception as exc:
    fetch_error = str(exc)

sample_headlines = {
    "Positive sample": "Markets rally after strong quarterly earnings and cooling inflation data.",
    "Negative sample": "Stocks fall sharply as recession fears grow and global trade tensions rise.",
    "Neutral sample": "Investors await central bank decision while markets remain range-bound.",
}

selected_sample = st.selectbox(
    "Quick test sample",
    options=["Yahoo latest", "Custom"] + list(sample_headlines.keys()),
)

headline_text = st.text_area(
    "Headlines",
    value=(
        headline_default
        if selected_sample == "Yahoo latest"
        else ""
        if selected_sample == "Custom"
        else sample_headlines[selected_sample]
    ),
    placeholder="Example: Central bank signals rate cut as inflation cools and markets rally",
    height=200,
)

if fetch_error:
    st.warning(f"Yahoo Finance news could not be loaded: {fetch_error}")
elif news_items:
    st.markdown('<div class="glass-card"><h3>Latest Yahoo Finance Headlines</h3>', unsafe_allow_html=True)
    st.markdown(f'<div class="news-pill">Ticker: {ticker}</div>', unsafe_allow_html=True)
    for index, item in enumerate(news_items, start=1):
        publisher = item.get("publisher", "Yahoo Finance")
        published_at = item.get("published_at", "Recent")
        st.markdown(
            f"""
            <div class="news-item">
                <strong>{index}. {item['title']}</strong><br/>
                <small>{publisher} | {published_at}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

if st.button("Predict Trend"):
    if not MODEL_FILE.exists():
        st.error("Train the model first by running src/train.py.")
    elif not headline_text.strip():
        st.warning("Please enter at least one news headline.")
    else:
        live_features = fetch_live_feature_row(ticker=ticker)
        prediction, scores, probability = predict_headline_bundle(headline_text, live_features)
        label = "Up" if prediction == 1 else "Down"
        sentiment_tone = (
            "Positive"
            if scores["compound"] > 0.05
            else "Negative"
            if scores["compound"] < -0.05
            else "Neutral"
        )

        result_col, score_col = st.columns([1.2, 1])
        with result_col:
            if label == "Up":
                st.success(f"Predicted market direction: {label}")
            else:
                st.error(f"Predicted market direction: {label}")

            if probability is not None:
                st.metric("Confidence for Up", f"{probability:.2%}")
            st.metric("Headline sentiment", sentiment_tone)
            st.caption(f"Live price features pulled from Yahoo Finance for `{ticker}`.")

        with score_col:
            st.write("Sentiment scores")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Positive", f"{scores['pos']:.3f}")
            metric_cols[1].metric("Neutral", f"{scores['neu']:.3f}")
            metric_cols[2].metric("Negative", f"{scores['neg']:.3f}")
            metric_cols[3].metric("Compound", f"{scores['compound']:.3f}")

st.markdown(
    '<p class="caption-note">For academic use only. This project demonstrates a research workflow and does not provide financial advice.</p>',
    unsafe_allow_html=True,
)
