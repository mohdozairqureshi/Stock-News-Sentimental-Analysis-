from nltk.sentiment import SentimentIntensityAnalyzer


def build_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def score_text(text: str, analyzer: SentimentIntensityAnalyzer) -> dict:
    clean_text = text if isinstance(text, str) else ""
    return analyzer.polarity_scores(clean_text)
