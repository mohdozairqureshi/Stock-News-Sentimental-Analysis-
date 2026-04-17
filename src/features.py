def get_text_column(df):
    return df["combined_headlines"]


def get_sentiment_columns(df):
    return df[["neg", "neu", "pos", "compound"]]


def to_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def get_numeric_columns(df):
    numeric_columns = [
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
    return df[numeric_columns]
