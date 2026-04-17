import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from src.config import FEATURES_FILE, FIGURES_DIR, METRICS_FILE
from src.data_pipeline import combine_headlines, load_news_data
from src.train import build_features, ensure_nltk_resources


sns.set_theme(style="whitegrid")


def load_feature_data() -> pd.DataFrame:
    if FEATURES_FILE.exists():
        df = pd.read_csv(FEATURES_FILE)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    ensure_nltk_resources()
    raw_df = load_news_data()
    combined_df = combine_headlines(raw_df)
    return build_features(combined_df)


def save_figure(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_label_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Label", hue="Label", palette=["#c0392b", "#1f7a4d"], legend=False)
    plt.title("Market Direction Label Distribution")
    plt.xlabel("Label (0 = Down, 1 = Up)")
    plt.ylabel("Number of Days")
    save_figure("label_distribution.png")


def plot_compound_by_label(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="Label", y="compound", hue="Label", palette=["#c0392b", "#1f7a4d"], legend=False)
    plt.title("Compound Sentiment by Market Label")
    plt.xlabel("Label (0 = Down, 1 = Up)")
    plt.ylabel("Compound Sentiment Score")
    save_figure("compound_by_label.png")


def plot_sentiment_correlation(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    corr = df[["neg", "neu", "pos", "compound", "Label"]].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="RdYlGn", fmt=".2f")
    plt.title("Sentiment Feature Correlation")
    save_figure("sentiment_correlation.png")


def plot_top_words(df: pd.DataFrame) -> None:
    vectorizer = CountVectorizer(stop_words="english", max_features=15)
    matrix = vectorizer.fit_transform(df["combined_headlines"])
    counts = matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    top_words = pd.DataFrame({"word": words, "count": counts}).sort_values("count", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_words, x="count", y="word", hue="word", dodge=False, palette="viridis", legend=False)
    plt.title("Most Frequent Words in Headlines")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    save_figure("top_words.png")


def plot_model_comparison() -> None:
    if not METRICS_FILE.exists():
        return

    metrics_df = pd.read_csv(METRICS_FILE)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=metrics_df, x="accuracy", y="model", hue="model", dodge=False, palette="mako", legend=False)
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Accuracy")
    plt.ylabel("Model")
    save_figure("model_comparison.png")


def plot_price_feature_correlation(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 6))
    corr_columns = [
        "return_1d",
        "return_3d",
        "ma_5_ratio",
        "ma_10_ratio",
        "volatility_5",
        "volume_change",
        "range_pct",
        "Label",
    ]
    corr = df[corr_columns].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Yahoo Price Feature Correlation")
    save_figure("price_feature_correlation.png")


def build_analysis() -> None:
    df = load_feature_data()
    plot_label_distribution(df)
    plot_compound_by_label(df)
    plot_sentiment_correlation(df)
    plot_price_feature_correlation(df)
    plot_top_words(df)
    plot_model_comparison()
    print(f"Saved analysis figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    build_analysis()
