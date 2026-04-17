# STOCK PREDICTION USING NEWS SENTIMENTS

## A PRECISE  INTRO 

Stock Market Trend Prediction Using News Headline Sentiment Analysis and Machine Learning

## Abstract

The stock market is influenced by many factors, including investor psychology, macroeconomic events, and corporate announcements. News headlines often shape short-term sentiment and can affect market direction. This project presents a machine learning system that predicts the next-day market direction as `Up` or `Down` by combining sentiment extracted from financial news headlines with market data fetched from Yahoo Finance. Historical headline data is cleaned and transformed into textual and sentiment-based features, while Yahoo Finance price history is used to generate technical indicators and the target label. Multiple machine learning models are trained and compared, including Logistic Regression, Random Forest, and Linear Support Vector Machine. The best-performing model is deployed through a Streamlit application that fetches up to five recent Yahoo Finance news articles for live prediction. The project demonstrates how natural language processing and financial data analysis can be combined for academic stock trend prediction, while also highlighting the limitations of such systems in real-world financial decision making.

## 1. Introduction

Financial markets react rapidly to the flow of information. News relating to company earnings, inflation, interest rates, regulation, geopolitical conflict, and investor confidence can influence short-term market movements. With the growth of online finance portals and APIs, it has become possible to collect both market data and financial news in an automated way.

Machine learning provides a practical way to analyze large text collections and structured market data together. Instead of predicting the exact future stock price, which is highly difficult and unstable, this project focuses on a simpler and more defendable task: predicting whether the market will move up or down on the next trading day.

The project combines three ideas:

1. financial headline collection and cleaning
2. sentiment analysis using VADER
3. classification using machine learning models

The final system is implemented as a working Streamlit web application that can demonstrate live predictions using the latest Yahoo Finance news headlines.

## 2. Problem Statement

Investors and analysts often examine financial news to understand market behavior, but manual interpretation is slow and subjective. There is a need for a system that can automatically process news headlines, combine them with historical market signals, and estimate the probable next-day direction of the market.

This project addresses the following problem:

To design and implement a machine learning model that predicts whether the market will go `Up` or `Down` on the next trading day using financial news headline sentiment and Yahoo Finance market data.

## 3. Objectives

The major objectives of the project are:

- to collect and preprocess historical financial news headlines
- to compute sentiment scores from headlines using natural language processing
- to fetch market price history from Yahoo Finance and create market-based features
- to train and compare multiple machine learning classification models
- to deploy the best model in a user-friendly web application
- to present the complete system with charts, evaluation metrics, and live demonstration capability

## 4. Scope of the Project

This project is developed for academic and demonstration purposes. It predicts short-term market direction and does not attempt exact stock price forecasting. The scope includes:

- historical news headline processing
- Yahoo Finance market history integration
- sentiment analysis
- machine learning classification
- Streamlit-based deployment

The scope does not include:

- high-frequency trading
- portfolio optimization
- investment advice
- guaranteed financial forecasting

## 5. Literature Survey

News sentiment has been widely studied as an indicator of market behavior. Many researchers have shown that investor sentiment, media tone, and macroeconomic news can influence short-term returns and volatility. Traditional approaches used bag-of-words models and lexicon-based sentiment analysis, while newer approaches use deep learning and transformer-based finance language models.

For an academic project with limited development time, lexicon-based sentiment analysis remains practical because it is simple to implement, fast to explain, and computationally efficient. Likewise, classical machine learning models such as Logistic Regression, Support Vector Machine, and Random Forest remain appropriate baselines for binary trend classification.

This project builds on those ideas by combining:

- text vectorization through TF-IDF
- VADER sentiment analysis
- Yahoo Finance price-based technical indicators
- classical supervised machine learning algorithms

## 6. Dataset Description

The project uses two data sources:

### 6.1 Historical News Headline Dataset

The historical headline dataset is based on `Combined_News_DJIA.csv`, which contains multiple top financial headlines for each trading date. These daily headlines are combined into one text field after cleaning unwanted formatting artifacts.

Key fields used:

- `Date`
- `Top1` to `Top25`

### 6.2 Yahoo Finance Market Data

Historical market price data is downloaded using the `yfinance` Python library. In the current model, the market ticker used for training is `^DJI` (Dow Jones Industrial Average). Yahoo Finance data is used to generate:

- opening price
- high price
- low price
- closing price
- trading volume
- daily returns
- moving average ratios
- short-term volatility
- price range percentage

The target label is generated from Yahoo Finance closing price data:

- `1` if next day close > current day close
- `0` otherwise

This makes the training target more consistent and realistic than relying only on sentiment-based assumptions.

## 7. System Architecture

The project workflow is as follows:

1. collect historical news headlines
2. clean and combine daily headlines
3. fetch Yahoo Finance historical market data
4. engineer technical market features
5. compute VADER sentiment scores from combined headlines
6. merge text features, sentiment features, and price features by date
7. train multiple machine learning classifiers
8. compare models and save the best model
9. use the saved model in a Streamlit app for live prediction

## 8. Methodology

### 8.1 Data Preprocessing

The historical headline data contains noisy formatting such as byte-string markers and escaped characters. These issues are cleaned before model training. Missing values in headline columns are replaced with empty strings, and the 25 daily headlines are concatenated into a single text input.

### 8.2 Sentiment Analysis

VADER (Valence Aware Dictionary and sEntiment Reasoner) is used to calculate sentiment scores for the combined headlines. Four values are generated:

- positive score
- negative score
- neutral score
- compound score

These values help convert text emotion into numerical form for machine learning.

### 8.3 Price Feature Engineering

From Yahoo Finance historical data, the following technical indicators are created:

- one-day return
- three-day return
- five-day moving average ratio
- ten-day moving average ratio
- five-day volatility
- volume change
- intraday price range percentage

These features give the model more market context in addition to sentiment.

### 8.4 Text Vectorization

The combined daily headlines are transformed into numerical text features using TF-IDF (Term Frequency-Inverse Document Frequency). This method captures the importance of words across the headline collection.

### 8.5 Model Training

Three classification models are trained and tested:

- Logistic Regression
- Random Forest
- Linear Support Vector Machine

A time-based split is used to preserve the chronological structure of market data. Approximately 80% of the earlier data is used for training and the remaining 20% is used for testing.

## 9. Tools and Technologies Used

- Python
- Pandas
- NumPy
- scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Streamlit
- yfinance
- Joblib

## 10. Implementation Details

The project is organized into modular Python files:

- `src/data_pipeline.py` handles data cleaning, Yahoo Finance download, and feature engineering
- `src/sentiment.py` handles VADER sentiment scoring
- `src/train.py` trains and compares machine learning models
- `src/eda.py` creates report figures and analysis charts
- `src/yahoo_service.py` fetches live Yahoo Finance news and current market features
- `src/app.py` runs the Streamlit user interface

The trained model is saved in:

- `models/stock_direction_model.joblib`

The model metadata is stored in:

- `models/model_info.json`

The model comparison results are stored in:

- `data/processed/model_comparison.csv`

## 11. Experimental Results

The latest trained models produced the following results:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.5126 | 0.5115 | 0.7761 | 0.6166 |
| Random Forest | 0.5025 | 0.5040 | 0.9502 | 0.6586 |
| Linear SVM | 0.5226 | 0.5203 | 0.7015 | 0.5975 |

The best-performing model is:

- `Linear SVM`
- Accuracy: `0.5226`
- Market ticker used for training: `^DJI`

Although the improvement over random guessing is modest, the result is acceptable for an academic project because stock trend prediction is an inherently difficult problem influenced by many external factors. The project successfully demonstrates end-to-end integration of sentiment analysis, market features, and machine learning deployment.

## 12. Generated Analysis Charts

The following charts were generated for report and presentation use:

- `report/figures/label_distribution.png`
- `report/figures/compound_by_label.png`
- `report/figures/sentiment_correlation.png`
- `report/figures/price_feature_correlation.png`
- `report/figures/top_words.png`
- `report/figures/model_comparison.png`

These figures support the analysis of class distribution, sentiment behavior, feature relationships, word frequency, and classifier performance.

## 13. Streamlit Application

A Streamlit interface is developed to demonstrate the project in an interactive way. The app:

- accepts a market ticker from the user
- fetches up to five recent Yahoo Finance headlines
- allows manual or live headline input
- computes sentiment scores in real time
- fetches current market features from Yahoo Finance
- predicts whether the market may move `Up` or `Down`

This makes the project suitable for final demonstration and viva presentation.

## 14. Advantages of the Proposed System

- simple and interpretable architecture
- combines textual and numerical features
- uses Yahoo Finance for practical data access
- includes a working graphical user interface
- easy to demonstrate in an academic setting
- modular code structure for future improvement

## 15. Limitations

The project has the following limitations:

- stock market prediction is highly uncertain and noisy
- news sentiment alone cannot explain all market movements
- Yahoo Finance news availability may vary by ticker
- the model accuracy is still limited
- the system does not use deep learning or finance-specific transformer models
- external macroeconomic variables are not explicitly included

## 16. Future Scope

The project can be improved in the future by:

- using company-specific historical news datasets
- adding transformer-based models such as FinBERT
- including more technical indicators and macroeconomic variables
- testing more tickers and sectors
- performing hyperparameter tuning
- adding confidence calibration and better explainability
- deploying the app on a cloud platform

## 17. Conclusion

This project demonstrates a complete machine learning workflow for stock market direction prediction using news headline sentiment analysis and Yahoo Finance data. Historical financial headlines are cleaned and converted into text and sentiment features, while Yahoo Finance market data provides target labels and technical signals. Multiple machine learning models are trained and evaluated, and the best model is deployed in a live Streamlit application. The resulting system is suitable for an engineering final year project because it is practical, explainable, and easy to demonstrate. Even though prediction accuracy remains limited due to the complexity of financial markets, the project successfully shows how sentiment analysis and machine learning can be integrated into a meaningful financial analytics application.

## 18. References

1. Yahoo Finance data accessed through the `yfinance` Python library.
2. Hutto, C.J. and Gilbert, E.E. VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
3. scikit-learn documentation for Logistic Regression, Random Forest, Support Vector Machine, TF-IDF, and preprocessing tools.
4. Streamlit documentation for Python web application deployment.

