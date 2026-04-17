# Final Year Project

## Title

Stock Market Trend Prediction Using News Headline Sentiment Analysis and Machine Learning

## Scope

This project predicts whether the market trend will move up or down based on:

- financial news headlines
- sentiment analysis
- machine learning classification

## Initial setup

- virtual environment created in `.venv`
- dependencies listed in `requirements.txt`
- raw datasets downloaded in `data/raw`
- baseline training code added in `src/`

## Run commands

From the project root:

```powershell
.\.venv\Scripts\activate
python -m src.train
python -m src.eda
streamlit run src/app.py
```

## Current workflow

- historical financial headlines from the project dataset
- Yahoo Finance API for market price history and technical features
- up to 5 recent Yahoo Finance news articles in the live Streamlit demo
- model comparison across Logistic Regression, Random Forest, and Linear SVM
- analysis charts generated in `report/figures`

## Important note

This project predicts market direction using headline sentiment and text features. It is intended for academic demonstration only.
