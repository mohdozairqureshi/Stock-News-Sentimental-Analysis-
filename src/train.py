import json
from pathlib import Path

import joblib
import nltk
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import BEST_REPORT_FILE, DEFAULT_MARKET_TICKER, METRICS_FILE, MODEL_FILE, MODEL_INFO_FILE
from src.data_pipeline import (
    combine_headlines,
    engineer_market_features,
    fetch_market_history,
    load_news_data,
    merge_news_and_market,
    save_features,
)
from src.features import to_dense
from src.sentiment import build_sentiment_analyzer, score_text


def ensure_nltk_resources() -> None:
    nltk.download("vader_lexicon", quiet=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = build_sentiment_analyzer()
    sentiment_scores = df["combined_headlines"].apply(lambda text: score_text(text, analyzer))
    score_df = pd.DataFrame(sentiment_scores.tolist())
    feature_df = pd.concat([df.reset_index(drop=True), score_df], axis=1)
    save_features(feature_df)
    return feature_df


NUMERIC_FEATURES = [
    "neg",
    "neu",
    "pos",
    "compound",
    "return_1d",
    "return_3d",
    "ma_5_ratio",
    "ma_10_ratio",
    "volatility_5",
    "volume_change",
    "range_pct",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=1500, stop_words="english"), "combined_headlines"),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def build_candidate_pipelines() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("features", build_preprocessor()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("features", build_preprocessor()),
                ("to_dense", FunctionTransformer(to_dense, accept_sparse=True)),
                ("model", RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)),
            ]
        ),
        "Linear SVM": Pipeline(
            steps=[
                ("features", build_preprocessor()),
                ("model", SVC(kernel="linear", probability=True, random_state=42)),
            ]
        ),
    }


def save_model_report(report_text: str) -> None:
    BEST_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    BEST_REPORT_FILE.write_text(report_text, encoding="utf-8")


def save_model_info(model_name: str, accuracy: float) -> None:
    MODEL_INFO_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_INFO_FILE.write_text(
        json.dumps(
            {
                "best_model": model_name,
                "accuracy": round(accuracy, 4),
                "market_ticker": DEFAULT_MARKET_TICKER,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def split_train_test(feature_df: pd.DataFrame):
    ordered_df = feature_df.sort_values("Date").reset_index(drop=True)
    split_index = int(len(ordered_df) * 0.8)
    train_df = ordered_df.iloc[:split_index].copy()
    test_df = ordered_df.iloc[split_index:].copy()
    return train_df, test_df


def train_model() -> None:
    ensure_nltk_resources()

    raw_df = load_news_data()
    combined_df = combine_headlines(raw_df)
    market_history = fetch_market_history(combined_df["Date"].min(), combined_df["Date"].max())
    market_features = engineer_market_features(market_history)
    merged_df = merge_news_and_market(combined_df, market_features)
    feature_df = build_features(merged_df)
    train_df, test_df = split_train_test(feature_df)

    X_train = train_df[["combined_headlines", *NUMERIC_FEATURES]]
    y_train = train_df["Label"]
    X_test = test_df[["combined_headlines", *NUMERIC_FEATURES]]
    y_test = test_df["Label"]

    model_rows = []
    best_model_name = None
    best_accuracy = -1.0
    best_pipeline = None
    best_report_text = ""

    for model_name, pipeline in build_candidate_pipelines().items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        report_text = classification_report(y_test, predictions, zero_division=0)

        model_rows.append(
            {
                "model": model_name,
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
            }
        )

        print(f"\n=== {model_name} ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(report_text)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_pipeline = pipeline
            best_report_text = report_text

    metrics_df = pd.DataFrame(model_rows).sort_values(by="accuracy", ascending=False)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_FILE, index=False)

    Path(MODEL_FILE).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_FILE)
    save_model_report(best_report_text)
    save_model_info(best_model_name, best_accuracy)

    print("\n=== Best Model Summary ===")
    print(metrics_df.to_string(index=False))
    print(
        f"\nTrain dates: {train_df['Date'].min().date()} to {train_df['Date'].max().date()} | "
        f"Test dates: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}"
    )
    print(f"\nSaved best model ({best_model_name}) to: {MODEL_FILE}")
    print(f"Saved model comparison to: {METRICS_FILE}")


if __name__ == "__main__":
    train_model()
