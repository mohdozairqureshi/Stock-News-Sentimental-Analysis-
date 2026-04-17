📈 Stock News Sentiment Analysis

A machine learning project that analyzes financial news to estimate stock sentiment and provide predictive insights for market trends.

🚀 Overview

This project leverages Natural Language Processing (NLP) and Machine Learning to extract sentiment from news articles related to stock tickers. The goal is to explore whether news sentiment can serve as a signal for stock movement.

Instead of just building a model, the focus was on end-to-end pipeline development — from data collection to deployment.

💡 Key Highlights
Built a complete news → sentiment → prediction pipeline
Integrated real-time data using Yahoo Finance API
Compared multiple ML models to select the best performer
Deployed an interactive web app using Streamlit
Achieved consistent model performance (~50–60%) on noisy real-world data
🧠 Approach
Data Collection
Fetched financial news articles using ticker symbols via Yahoo Finance API
Text Processing
Applied NLP techniques:
TF-IDF vectorization
Tokenization & stopword removal (NLTK)
Positive/Negative word analysis
Model Selection

Tested multiple algorithms:

Logistic Regression
Random Forest (Selected)
Linear SVM

Random Forest was chosen due to:

Better generalization on noisy text data
Higher recall (important for capturing sentiment signals)
📊 Model Performance
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	51.26%	51.15%	77.61%	61.66%
Random Forest	50.25%	50.40%	95.02%	65.86%
Linear SVM	52.26%	52.03%	70.15%	59.75%

Note: Financial sentiment prediction is inherently noisy; results reflect real-world complexity rather than overfitted benchmarks.

🛠️ Tech Stack

Languages

Python (ML & backend)
HTML/CSS (frontend)

Libraries & Tools

pandas, numpy
scikit-learn
nltk
matplotlib, seaborn
streamlit
joblib
yfinance
codex(code review and analysis) 


🌐 Deployment
Deployed using Streamlit for an interactive user interface
Users can input stock tickers and view sentiment analysis results in real time



📌 Features
Real-time news fetching
Sentiment classification of articles
Model comparison and evaluation
Simple UI for non-technical users


⚠️ Limitations
Moderate accuracy due to unpredictable nature of financial markets
Relies solely on news sentiment (no technical indicators)
Limited dataset size






🔮 Future Improvements
Incorporate deep learning models (LSTM, BERT)
Combine sentiment with historical stock data
Improve dataset quality and size
Add portfolio-level analysis





📎 How to Run
git clone <repo-link>
cd project-folder
pip install -r requirements.txt
streamlit run app.py







#######This is made for academic purposes,this model is still prone to mistakes between real values and variability#############
