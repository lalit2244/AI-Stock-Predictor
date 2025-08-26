# AI-Stock-Predictor
A comprehensive stock analysis application built with Streamlit featuring AI-powered predictions, multi-stock comparison, portfolio analysis, and real-time news sentiment analysis.
Features

AI-Powered Stock Predictions: Uses Random Forest machine learning models for price predictions
Multi-Stock Comparison: Compare performance metrics across multiple stocks
Portfolio Analysis: Build and analyze custom portfolios with correlation analysis
Technical Analysis: RSI, MACD, Moving Averages, and volatility indicators
News Sentiment Analysis: Real-time news analysis with sentiment scoring
Interactive Charts: Advanced Plotly visualizations with candlestick charts
Market Overview: Major indices, cryptocurrency, and economic indicators
Multiple Themes: Dark mode, light mode, colorful, and professional themes

Installation
Prerequisites

Python 3.8 or higher
pip package manager

Setup

Clone the repository:

bashgit clone https://github.com/yourusername/ultimate-stock-analysis.git
cd ultimate-stock-analysis

Create a virtual environment:

bashpython -m venv stock_analysis_env

Activate the virtual environment:


Windows: stock_analysis_env\Scripts\activate
macOS/Linux: source stock_analysis_env/bin/activate


Install required packages:

bashpip install -r requirements.txt
Usage
Run the Streamlit application:
bashstreamlit run app.py
The application will open in your default web browser at http://localhost:8501
Application Structure
Analysis Modes

Single Stock Analysis

Comprehensive analysis of individual stocks
AI price predictions with confidence intervals
Technical indicators (RSI, MACD, Moving Averages)
Real-time news sentiment analysis


Multi-Stock Comparison

Side-by-side comparison of multiple stocks
Performance metrics and risk analysis
Correlation analysis between stocks
Visual comparison charts


Portfolio Analysis

Custom portfolio creation with weight allocation
Portfolio performance tracking
Diversification analysis
Risk-return optimization



Stock Categories

Tech Giants (AAPL, MSFT, GOOGL, AMZN, META, NFLX, NVDA, TSLA)
Financial (JPM, BAC, WFC, GS, V, MA)
Healthcare (JNJ, PFE, UNH, MRNA, ABT)
Consumer (KO, PEP, NKE, MCD, DIS)

Technical Details
Machine Learning

Random Forest Regressor: For stock price predictions
Feature Engineering: Technical indicators as input features
Model Validation: Train/test split with accuracy scoring

Technical Analysis

RSI: Relative Strength Index for momentum analysis
MACD: Moving Average Convergence Divergence
Moving Averages: 5, 10, 20, and 50-day periods
Volatility: Rolling standard deviation analysis

Data Sources

Stock Data: Yahoo Finance API via yfinance
News Data: Yahoo Finance news feed
Market Indices: Real-time data for major indices

Limitations and Disclaimers

Educational Purpose: This application is for educational and research purposes only
Not Financial Advice: Do not use for actual investment decisions
Data Accuracy: Stock data depends on Yahoo Finance API availability
Prediction Accuracy: AI predictions are experimental and not guaranteed
Market Risk: All investments carry risk of loss

Contributing

Fork the repository
Create a feature branch (git checkout -b feature/new-feature)
Commit your changes (git commit -am 'Add new feature')
Push to the branch (git push origin feature/new-feature)
Create a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.
Troubleshooting
Common Issues

Module Import Errors: Ensure all requirements are installed
Data Loading Issues: Check internet connection for API access
Performance Issues: Reduce the number of stocks in comparison mode
Chart Display Problems: Update your browser or try a different one

Performance Tips

Use shorter time periods for faster loading
Limit portfolio analysis to 5-6 stocks maximum
Close unused browser tabs to improve performance

Future Enhancements

Integration with more data sources
Advanced ML models (LSTM, Transformer)
Real-time streaming data
Mobile-responsive design
Export functionality for reports
More sophisticated sentiment analysis

Contact
For questions, issues, or contributions, please open an issue on GitHub.

Disclaimer: This software is provided "as is" without warranty of any kind. Past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.
