#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
import time
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🚀 Ultimate Stock Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
def load_theme(theme):
    if theme == "Dark Mode 🌙":
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3a 50%, #2d1b69 100%);
            color: white;
        }
        .metric-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            margin: 10px 0;
        }
        .comparison-card {
            background: linear-gradient(45deg, rgba(0,255,0,0.1), rgba(0,100,255,0.1));
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #00ff88;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)

# Available stocks with categories
STOCK_CATEGORIES = {
    "🍎 Tech Giants": {
        "Apple": "AAPL",
        "Microsoft": "MSFT", 
        "Google": "GOOGL",
        "Amazon": "AMZN",
        "Meta": "META",
        "Netflix": "NFLX",
        "Nvidia": "NVDA",
        "Tesla": "TSLA"
    },
    "🏦 Financial": {
        "JPMorgan": "JPM",
        "Bank of America": "BAC",
        "Wells Fargo": "WFC",
        "Goldman Sachs": "GS",
        "Visa": "V",
        "Mastercard": "MA"
    },
    "🏥 Healthcare": {
        "Johnson & Johnson": "JNJ",
        "Pfizer": "PFE",
        "UnitedHealth": "UNH",
        "Moderna": "MRNA",
        "Abbott": "ABT"
    },
    "🛒 Consumer": {
        "Coca-Cola": "KO",
        "PepsiCo": "PEP",
        "Nike": "NKE",
        "McDonald's": "MCD",
        "Disney": "DIS"
    }
}

# Flatten all stocks for easy access
ALL_STOCKS = {}
for category, stocks in STOCK_CATEGORIES.items():
    ALL_STOCKS.update(stocks)

def get_stock_data(symbols, period="1y", max_retries=3):
    """Enhanced stock data fetching with robust error handling and retries"""
    if isinstance(symbols, str):
        symbols = [symbols]
    
    data = {}
    
    for symbol in symbols:
        success = False
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Add delay between retries
                if attempt > 0:
                    time.sleep(1)
                
                # Try different approaches
                if attempt == 0:
                    # Standard approach
                    stock = yf.Ticker(symbol)
                    hist_data = stock.history(period=period, timeout=10)
                    info = stock.info
                elif attempt == 1:
                    # Try with different session
                    session = requests.Session()
                    session.headers.update({
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    stock = yf.Ticker(symbol, session=session)
                    hist_data = stock.history(period=period, timeout=15)
                    info = stock.info
                else:
                    # Last attempt with minimal data
                    stock = yf.Ticker(symbol)
                    hist_data = stock.history(period="3mo", timeout=20)  # Shorter period
                    info = {}  # Skip info if problematic
                
                # Validate data
                if hist_data is not None and not hist_data.empty and len(hist_data) > 10:
                    # Clean the data
                    hist_data = hist_data.dropna()
                    if len(hist_data) > 5:  # Minimum viable data
                        data[symbol] = {
                            'history': hist_data,
                            'info': info if info else {}
                        }
                        success = True
                        break
                
            except Exception as e:
                last_error = str(e)
                continue
        
        if not success:
            st.warning(f"Could not load data for {symbol} after {max_retries} attempts. Error: {last_error}")
    
    return data

def get_stock_data_fallback(symbol):
    """Fallback method using alternative data source or cached data"""
    try:
        # Simple fallback - create synthetic data for demo purposes
        # In production, you might use alternative APIs like Alpha Vantage, IEX, etc.
        
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        # Create realistic stock price simulation
        np.random.seed(hash(symbol) % 1000)  # Consistent seed per symbol
        
        base_price = 150  # Starting price
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        synthetic_data = pd.DataFrame({
            'Open': prices[:-1],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'Close': prices[1:],
            'Volume': [np.random.randint(1000000, 5000000) for _ in range(252)]
        }, index=dates)
        
        return synthetic_data
    
    except:
        return None

def get_news_sentiment(symbol, limit=5):
    """Enhanced news and sentiment analysis with better error handling"""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news[:limit]
        
        news_data = []
        for article in news:
            title = article.get('title', '')
            
            # Enhanced keyword lists
            positive_words = [
                'gains', 'up', 'rise', 'bull', 'bullish', 'positive', 'growth', 'strong', 
                'beat', 'beats', 'surge', 'soar', 'rally', 'jump', 'climbs', 'boost',
                'outperform', 'upgrade', 'buy', 'higher', 'record', 'profit', 'earnings',
                'revenue', 'success', 'breakthrough', 'optimistic', 'confident'
            ]
            
            negative_words = [
                'falls', 'fall', 'down', 'drop', 'drops', 'bear', 'bearish', 'negative', 
                'decline', 'weak', 'weakness', 'miss', 'misses', 'plunge', 'crash',
                'tumble', 'slide', 'slump', 'underperform', 'downgrade', 'sell',
                'lower', 'loss', 'losses', 'concern', 'worry', 'risk', 'warning'
            ]
            
            title_lower = title.lower()
            pos_count = sum(1 for word in positive_words if word in title_lower)
            neg_count = sum(1 for word in negative_words if word in title_lower)
            
            if pos_count > neg_count:
                sentiment = "Positive 📈"
                sentiment_score = 0.7
            elif neg_count > pos_count:
                sentiment = "Negative 📉"
                sentiment_score = 0.3
            else:
                sentiment = "Neutral ➡️"
                sentiment_score = 0.5
            
            news_data.append({
                'title': title,
                'url': article.get('link', ''),
                'published': datetime.fromtimestamp(article.get('providerPublishTime', 0)),
                'sentiment': sentiment,
                'sentiment_score': sentiment_score
            })
        
        return news_data
    except Exception as e:
        # Return mock news data if real data unavailable
        return [{
            'title': f'Mock news for {symbol} - Market analysis ongoing',
            'url': '#',
            'published': datetime.now(),
            'sentiment': 'Neutral ➡️',
            'sentiment_score': 0.5
        }]

def advanced_technical_analysis(data):
    """Calculate advanced technical indicators with error handling"""
    try:
        df = data.copy()
        
        # Ensure we have enough data
        if len(df) < 50:
            st.warning("Limited data available for technical analysis")
        
        # Moving Averages
        df['MA_5'] = df['Close'].rolling(window=min(5, len(df)//10)).mean()
        df['MA_10'] = df['Close'].rolling(window=min(10, len(df)//5)).mean()
        df['MA_20'] = df['Close'].rolling(window=min(20, len(df)//3)).mean()
        df['MA_50'] = df['Close'].rolling(window=min(50, len(df)//2)).mean()
        
        # RSI
        if len(df) >= 14:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        if len(df) >= 26:
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=min(10, len(df)//5)).std()
        
        return df
    
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")
        return data

def predict_stock_price_advanced(data):
    """Enhanced prediction with better error handling"""
    if data is None or len(data) < 30:
        return None, None, None
    
    try:
        # Technical analysis
        df = advanced_technical_analysis(data)
        df = df.dropna()
        
        if len(df) < 20:
            return None, None, None
        
        # Prepare features - only use what's available
        potential_features = ['MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD', 'Volatility', 'Volume']
        available_features = []
        
        for feature in potential_features:
            if feature in df.columns and not df[feature].isna().all() and df[feature].nunique() > 1:
                available_features.append(feature)
        
        if len(available_features) < 2:
            # Use simple price-based features as fallback
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_MA'] = df['Close'].rolling(5).mean()
            available_features = ['Price_Change', 'Price_MA', 'Volume']
            df = df.dropna()
        
        if len(df) < 10:
            return None, None, None
        
        X = df[available_features].values
        y = df['Close'].values
        
        # Simple train-test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Random Forest Model
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        
        # Prediction
        if len(X) > 0:
            rf_pred = rf_model.predict(X[-1].reshape(1, -1))[0]
            
            # Calculate confidence based on recent prediction accuracy
            if len(X_test) > 0:
                test_pred = rf_model.predict(X_test)
                mse = np.mean((test_pred - y_test) ** 2)
                rf_confidence = max(10, min(85, 100 - (mse / np.mean(y_test)) * 100))
            else:
                rf_confidence = 50  # Default confidence
            
            return rf_pred, rf_confidence, {'rf_prediction': rf_pred, 'rf_confidence': rf_confidence}
        
        return None, None, None
        
    except Exception as e:
        st.warning(f"Prediction model error: {str(e)}")
        return None, None, None

def compare_stocks(stock_data):
    """Enhanced stock comparison with better error handling"""
    if len(stock_data) < 2:
        return None
    
    comparison_results = {}
    
    for symbol, data in stock_data.items():
        try:
            hist = data['history']
            
            if hist is None or hist.empty or len(hist) < 2:
                continue
            
            # Calculate metrics
            current_price = hist['Close'].iloc[-1]
            start_price = hist['Close'].iloc[0]
            total_return = ((current_price - start_price) / start_price) * 100
            
            # Volatility
            daily_returns = hist['Close'].pct_change().dropna()
            if len(daily_returns) > 1:
                volatility = daily_returns.std() * np.sqrt(252) * 100
                
                # Sharpe Ratio
                risk_free_rate = 0.02  # 2% annual risk-free rate
                excess_returns = daily_returns.mean() - (risk_free_rate/252)
                sharpe_ratio = (excess_returns / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
                
                # Max Drawdown
                cumulative = (1 + daily_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min() * 100
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            comparison_results[symbol] = {
                'current_price': current_price,
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_volume': hist['Volume'].mean() if 'Volume' in hist.columns else 0
            }
            
        except Exception as e:
            st.warning(f"Error analyzing {symbol}: {str(e)}")
            continue
    
    return comparison_results if comparison_results else None

def main():
    # Enhanced Header
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px;">
    <h1 style="color: white; margin: 0;">🚀 Ultimate Stock Analysis Platform</h1>
    <h3 style="color: white; margin: 5px 0;">AI-Powered Stock Prediction, Comparison & News Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar
    st.sidebar.markdown("## 🎛️ Advanced Control Panel")
    
    # Theme selector
    theme = st.sidebar.selectbox(
        "🎨 Choose Theme:",
        ["Dark Mode 🌙", "Light Mode ☀️", "Colorful 🌈", "Professional 💼"]
    )
    load_theme(theme)
    
    # Analysis Mode
    analysis_mode = st.sidebar.selectbox(
        "📊 Analysis Mode:",
        ["Single Stock Analysis", "Multi-Stock Comparison", "Portfolio Analysis"]
    )
    
    if analysis_mode == "Single Stock Analysis":
        # Single stock analysis
        st.sidebar.markdown("### 📈 Stock Selection")
        
        category = st.sidebar.selectbox("Choose Category:", list(STOCK_CATEGORIES.keys()))
        selected_stock_name = st.sidebar.selectbox(
            "Choose Stock:", 
            list(STOCK_CATEGORIES[category].keys())
        )
        symbol = STOCK_CATEGORIES[category][selected_stock_name]
        
        # Time period
        period = st.sidebar.selectbox(
            "⏰ Time Period:",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=3
        )
        
        # Analysis options
        st.sidebar.markdown("### 🔍 Analysis Options")
        show_predictions = st.sidebar.checkbox("🤖 AI Predictions", True)
        show_technical = st.sidebar.checkbox("📊 Technical Analysis", True)
        show_news = st.sidebar.checkbox("📰 News & Sentiment", True)
        use_fallback = st.sidebar.checkbox("📊 Use Demo Data if Needed", True)
        
        # Main analysis
        st.markdown(f"## 📊 {selected_stock_name} ({symbol}) - Comprehensive Analysis")
        
        # Fetch data with progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Loading stock data...")
        progress_bar.progress(25)
        
        stock_data = get_stock_data([symbol], period)
        progress_bar.progress(75)
        
        # Fallback if primary data source fails
        if symbol not in stock_data and use_fallback:
            status_text.text("🔄 Primary data source failed, using fallback...")
            fallback_data = get_stock_data_fallback(symbol)
            if fallback_data is not None:
                stock_data[symbol] = {
                    'history': fallback_data,
                    'info': {'shortName': selected_stock_name}
                }
                st.info("📊 Using simulated data for demonstration purposes")
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        if symbol in stock_data:
            data = stock_data[symbol]['history']
            info = stock_data[symbol]['info']
            
            # Validate data
            if data is None or data.empty:
                st.error("❌ No valid data available for this stock")
                return
            
            # Current metrics
            try:
                current_price = data['Close'].iloc[-1]
                if len(data) >= 2:
                    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                    change_percent = (price_change / data['Close'].iloc[-2]) * 100
                else:
                    price_change = 0
                    change_percent = 0
                
                # Enhanced metrics display
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("💰 Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({change_percent:+.1f}%)")
                
                with col2:
                    st.metric("📈 Period High", f"${data['High'].max():.2f}")
                
                with col3:
                    st.metric("📉 Period Low", f"${data['Low'].min():.2f}")
                
                with col4:
                    market_cap = info.get('marketCap', 0) if info else 0
                    if market_cap > 1e12:
                        market_cap_display = f"${market_cap/1e12:.1f}T"
                    elif market_cap > 1e9:
                        market_cap_display = f"${market_cap/1e9:.1f}B"
                    elif market_cap > 1e6:
                        market_cap_display = f"${market_cap/1e6:.1f}M"
                    else:
                        market_cap_display = "N/A" if market_cap == 0 else f"${market_cap:,.0f}"
                    st.metric("🏢 Market Cap", market_cap_display)
                
                with col5:
                    avg_volume = data['Volume'].mean() if 'Volume' in data.columns else 0
                    st.metric("📊 Avg Volume", f"{avg_volume:,.0f}")
                
                # Advanced Chart
                st.markdown("### 📈 Advanced Price Chart")
                
                fig = go.Figure()
                
                # Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=f"{selected_stock_name}"
                ))
                
                # Add moving averages if available
                df_tech = advanced_technical_analysis(data)
                if 'MA_20' in df_tech.columns and not df_tech['MA_20'].isna().all():
                    fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MA_20'], 
                                           name='MA 20', line=dict(color='orange', width=2)))
                if 'MA_50' in df_tech.columns and not df_tech['MA_50'].isna().all():
                    fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MA_50'], 
                                           name='MA 50', line=dict(color='red', width=2)))
                
                fig.update_layout(
                    title=f"📈 {selected_stock_name} - Advanced Chart",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=600,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error displaying metrics: {str(e)}")
                return
            
            # AI Predictions
            if show_predictions:
                st.markdown("### 🤖 AI-Powered Predictions")
                
                with st.spinner("🧠 Running AI prediction models..."):
                    prediction, confidence, model_info = predict_stock_price_advanced(data.copy())
                
                if prediction and confidence:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h3>🎯 AI Prediction</h3>
                        <h2 style="color: #00ff00;">${prediction:.2f}</h2>
                        <p>Random Forest Model</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                        <h3>🎲 Confidence</h3>
                        <h2 style="color: #ffd700;">{confidence:.1f}%</h2>
                        <p>Model Accuracy</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        price_diff = prediction - current_price
                        change_pct = (price_diff / current_price) * 100
                        color = "#00ff00" if price_diff > 0 else "#ff4757"
                        direction = "🚀 BULLISH" if price_diff > 0 else "🐻 BEARISH"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                        <h3>📊 Signal</h3>
                        <h2 style="color: {color};">{direction}</h2>
                        <p>{change_pct:+.2f}% Expected</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Not enough data for reliable AI prediction")
            
            # Technical Analysis
            if show_technical:
                st.markdown("### 📊 Advanced Technical Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI Chart
                    if 'RSI' in df_tech.columns and not df_tech['RSI'].isna().all():
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=df_tech.index, y=df_tech['RSI'], 
                                                   name='RSI', line=dict(color='purple')))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                        fig_rsi.update_layout(title="📈 RSI Indicator", template="plotly_dark")
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    else:
                        st.info("RSI data not available with current dataset")
                
                with col2:
                    # MACD Chart
                    if 'MACD' in df_tech.columns and not df_tech['MACD'].isna().all():
                        fig_macd = go.Figure()
                        fig_macd.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD'], 
                                                    name='MACD', line=dict(color='blue')))
                        if 'MACD_Signal' in df_tech.columns:
                            fig_macd.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD_Signal'], 
                                                        name='Signal', line=dict(color='red')))
                        fig_macd.update_layout(title="📊 MACD Indicator", template="plotly_dark")
                        st.plotly_chart(fig_macd, use_container_width=True)
                    else:
                        st.info("MACD data not available with current dataset")
                
                # Technical Summary
                st.markdown("#### 📋 Technical Indicators Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'RSI' in df_tech.columns and not df_tech['RSI'].isna().all():
                        current_rsi = df_tech['RSI'].iloc[-1]
                        rsi_signal = "🔴 Overbought" if current_rsi > 70 else "🟢 Oversold" if current_rsi < 30 else "🟡 Neutral"
                        st.metric("RSI", f"{current_rsi:.1f}", rsi_signal)
                    else:
                        st.metric("RSI", "N/A", "Insufficient data")
                
                with col2:
                    if 'MACD' in df_tech.columns and not df_tech['MACD'].isna().all():
                        current_macd = df_tech['MACD'].iloc[-1]
                        macd_signal = "🟢 Bullish" if current_macd > 0 else "🔴 Bearish"
                        st.metric("MACD", f"{current_macd:.3f}", macd_signal)
                    else:
                        st.metric("MACD", "N/A", "Insufficient data")
                
                with col3:
                    if 'Volatility' in df_tech.columns and not df_tech['Volatility'].isna().all():
                        volatility = df_tech['Volatility'].iloc[-1]
                        vol_level = "🔴 High" if volatility > df_tech['Volatility'].quantile(0.8) else "🟡 Medium" if volatility > df_tech['Volatility'].quantile(0.4) else "🟢 Low"
                        st.metric("Volatility", f"{volatility:.2f}", vol_level)
                    else:
                        st.metric("Volatility", "N/A", "Calculating...")
                
                with col4:
                    if 'MA_20' in df_tech.columns and not df_tech['MA_20'].isna().all():
                        ma_20 = df_tech['MA_20'].iloc[-1]
                        trend = "🟢 Uptrend" if current_price > ma_20 else "🔴 Downtrend"
                        st.metric("Trend vs MA20", f"{((current_price/ma_20-1)*100):+.1f}%", trend)
                    else:
                        st.metric("Trend", "N/A", "Calculating...")
            
            # News and Sentiment Analysis
            if show_news:
                st.markdown("### 📰 Latest News & Sentiment Analysis")
                
                with st.spinner("📰 Fetching latest news and analyzing sentiment..."):
                    news_data = get_news_sentiment(symbol)
                
                if news_data:
                    # Overall sentiment
                    avg_sentiment = np.mean([news['sentiment_score'] for news in news_data])
                    overall_sentiment = "📈 Positive" if avg_sentiment > 0.6 else "📉 Negative" if avg_sentiment < 0.4 else "➡️ Neutral"
                    
                    st.markdown(f"#### Overall News Sentiment: {overall_sentiment} ({avg_sentiment:.2f}/1.0)")
                    
                    # News articles
                    for news in news_data:
                        with st.expander(f"📰 {news['title'][:80]}..." if len(news['title']) > 80 else f"📰 {news['title']}"):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**Published:** {news['published'].strftime('%Y-%m-%d %H:%M')}")
                                if news['url'] and news['url'] != '#':
                                    st.markdown(f"[📖 Read Article]({news['url']})")
                            with col2:
                                st.markdown(f"**Sentiment:** {news['sentiment']}")
                                st.progress(news['sentiment_score'])
                else:
                    st.info("📰 No recent news available for this stock")
        else:
            st.error("❌ Could not load stock data. Please check your internet connection or try a different stock.")
    
    elif analysis_mode == "Multi-Stock Comparison":
        # Multi-stock comparison with enhanced error handling
        st.sidebar.markdown("### 📊 Stock Comparison")
        
        # Select multiple stocks
        selected_stocks = []
        for category, stocks in STOCK_CATEGORIES.items():
            selected = st.sidebar.multiselect(f"{category}:", list(stocks.keys()), key=category)
            for stock_name in selected:
                selected_stocks.append((stock_name, stocks[stock_name]))
        
        if len(selected_stocks) < 2:
            st.warning("Please select at least 2 stocks for comparison")
            return
        
        # Time period
        period = st.sidebar.selectbox("⏰ Comparison Period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
        use_fallback_comparison = st.sidebar.checkbox("📊 Use Demo Data if API Fails", True)
        
        st.markdown(f"## 🔍 Comparing {len(selected_stocks)} Stocks")
        
        # Fetch data for all selected stocks
        symbols = [stock[1] for stock in selected_stocks]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f"📊 Loading data for {len(symbols)} stocks...")
        comparison_data = get_stock_data(symbols, period)
        progress_bar.progress(50)
        
        # Use fallback for any failed stocks
        if len(comparison_data) < len(symbols) and use_fallback_comparison:
            status_text.text("📊 Using fallback data for missing stocks...")
            for symbol in symbols:
                if symbol not in comparison_data:
                    fallback_data = get_stock_data_fallback(symbol)
                    if fallback_data is not None:
                        comparison_data[symbol] = {
                            'history': fallback_data,
                            'info': {'shortName': next(name for name, sym in selected_stocks if sym == symbol)}
                        }
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        if comparison_data and len(comparison_data) >= 2:
            # Stock comparison metrics
            comparison_results = compare_stocks(comparison_data)
            
            if comparison_results and len(comparison_results) >= 2:
                # Display successful stocks
                successful_stocks = list(comparison_results.keys())
                stock_names = [next(name for name, sym in selected_stocks if sym == stock) for stock in successful_stocks]
                
                if len(comparison_results) < len(selected_stocks):
                    failed_stocks = [sym for _, sym in selected_stocks if sym not in successful_stocks]
                    st.warning(f"⚠️ Could not load data for: {', '.join(failed_stocks)}")
                
                # Comparison table
                st.markdown("### 📊 Performance Comparison")
                
                df_comparison = pd.DataFrame(comparison_results).T
                df_comparison.index = stock_names
                
                # Format the dataframe for display
                df_display = df_comparison.copy()
                df_display['current_price'] = df_display['current_price'].apply(lambda x: f"${x:.2f}")
                df_display['total_return'] = df_display['total_return'].apply(lambda x: f"{x:+.2f}%")
                df_display['volatility'] = df_display['volatility'].apply(lambda x: f"{x:.2f}%")
                df_display['sharpe_ratio'] = df_display['sharpe_ratio'].apply(lambda x: f"{x:.3f}")
                df_display['max_drawdown'] = df_display['max_drawdown'].apply(lambda x: f"{x:.2f}%")
                df_display['avg_volume'] = df_display['avg_volume'].apply(lambda x: f"{x:,.0f}")
                
                df_display.columns = ['Current Price', 'Total Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Avg Volume']
                
                st.dataframe(df_display, use_container_width=True)
                
                # Comparison charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Return comparison
                    fig_returns = go.Figure()
                    returns = df_comparison['total_return'].values
                    colors = ['green' if r > 0 else 'red' for r in returns]
                    
                    fig_returns.add_trace(go.Bar(
                        x=stock_names,
                        y=returns,
                        marker_color=colors,
                        name='Total Return'
                    ))
                    fig_returns.update_layout(title="📊 Total Return Comparison", yaxis_title="Return (%)", template="plotly_dark")
                    st.plotly_chart(fig_returns, use_container_width=True)
                
                with col2:
                    # Risk vs Return scatter
                    fig_risk = go.Figure()
                    fig_risk.add_trace(go.Scatter(
                        x=df_comparison['volatility'],
                        y=df_comparison['total_return'],
                        mode='markers+text',
                        text=stock_names,
                        textposition='top center',
                        marker=dict(size=15, color=df_comparison['total_return'], colorscale='RdYlGn'),
                        name='Risk vs Return'
                    ))
                    fig_risk.update_layout(title="🎯 Risk vs Return", xaxis_title="Volatility (%)", yaxis_title="Return (%)", template="plotly_dark")
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                # Price comparison chart
                st.markdown("### 📈 Price Performance Comparison")
                
                fig_price_comp = go.Figure()
                
                for stock_name, symbol in selected_stocks:
                    if symbol in comparison_data:
                        hist = comparison_data[symbol]['history']
                        if hist is not None and not hist.empty:
                            # Normalize to percentage change from start
                            normalized_prices = ((hist['Close'] / hist['Close'].iloc[0]) - 1) * 100
                            fig_price_comp.add_trace(go.Scatter(
                                x=hist.index,
                                y=normalized_prices,
                                name=stock_name,
                                line=dict(width=3)
                            ))
                
                fig_price_comp.update_layout(
                    title="📈 Normalized Price Performance (%)",
                    xaxis_title="Date",
                    yaxis_title="% Change from Start",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig_price_comp, use_container_width=True)
                
                # Best performers
                st.markdown("### 🏆 Top Performers")
                col1, col2, col3 = st.columns(3)
                
                try:
                    with col1:
                        best_return = df_comparison['total_return'].idxmax()
                        best_return_val = df_comparison.loc[best_return, 'total_return']
                        st.markdown(f"""
                        <div class="comparison-card">
                        <h4>🥇 Best Return</h4>
                        <h3>{best_return}</h3>
                        <p>{best_return_val:+.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        best_sharpe = df_comparison['sharpe_ratio'].idxmax()
                        best_sharpe_val = df_comparison.loc[best_sharpe, 'sharpe_ratio']
                        st.markdown(f"""
                        <div class="comparison-card">
                        <h4>⚡ Best Risk-Adjusted</h4>
                        <h3>{best_sharpe}</h3>
                        <p>{best_sharpe_val:.3f} Sharpe</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        lowest_vol = df_comparison['volatility'].idxmin()
                        lowest_vol_val = df_comparison.loc[lowest_vol, 'volatility']
                        st.markdown(f"""
                        <div class="comparison-card">
                        <h4>🛡️ Lowest Risk</h4>
                        <h3>{lowest_vol}</h3>
                        <p>{lowest_vol_val:.2f}% Vol</p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning("Could not determine top performers due to insufficient data")
            else:
                st.error("❌ Could not analyze the selected stocks for comparison")
        else:
            st.error("❌ Could not load enough stock data for comparison. Please try different stocks or check your connection.")
    
    elif analysis_mode == "Portfolio Analysis":
        st.sidebar.markdown("### 💼 Portfolio Builder")
        
        st.markdown("## 💼 Custom Portfolio Analysis")
        st.info("📝 Build your custom portfolio by selecting stocks and their weights")
        
        # Portfolio creation with error handling
        portfolio_stocks = []
        portfolio_weights = []
        
        # Add stocks to portfolio
        num_stocks = st.sidebar.number_input("Number of stocks in portfolio:", min_value=2, max_value=10, value=3)
        
        total_weight = 0
        for i in range(num_stocks):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Stock selection
                all_stock_names = list(ALL_STOCKS.keys())
                selected_stock = st.selectbox(f"Stock {i+1}:", all_stock_names, key=f"stock_{i}")
                portfolio_stocks.append(ALL_STOCKS[selected_stock])
            
            with col2:
                # Weight allocation
                weight = st.number_input(f"Weight % for {selected_stock}:", min_value=0.0, max_value=100.0, value=33.33, step=0.01, key=f"weight_{i}")
                portfolio_weights.append(weight)
                total_weight += weight
        
        st.markdown(f"**Total Portfolio Weight: {total_weight:.2f}%**")
        
        if abs(total_weight - 100) > 0.01:
            st.error("⚠️ Portfolio weights must sum to 100%!")
        else:
            # Analyze portfolio
            period = st.sidebar.selectbox("⏰ Analysis Period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
            use_fallback_portfolio = st.sidebar.checkbox("📊 Use Demo Data if API Fails", True)
            
            with st.spinner("📊 Analyzing your portfolio..."):
                portfolio_data = get_stock_data(portfolio_stocks, period)
            
            # Use fallback for missing stocks
            if len(portfolio_data) < len(portfolio_stocks) and use_fallback_portfolio:
                for symbol in portfolio_stocks:
                    if symbol not in portfolio_data:
                        fallback_data = get_stock_data_fallback(symbol)
                        if fallback_data is not None:
                            portfolio_data[symbol] = {
                                'history': fallback_data,
                                'info': {}
                            }
            
            if portfolio_data and len(portfolio_data) >= 2:
                # Calculate portfolio performance
                try:
                    all_dates = None
                    for symbol in portfolio_stocks:
                        if symbol in portfolio_data and portfolio_data[symbol]['history'] is not None:
                            dates = portfolio_data[symbol]['history'].index
                            if all_dates is None:
                                all_dates = dates
                            else:
                                all_dates = all_dates.intersection(dates)
                    
                    if all_dates is not None and len(all_dates) > 10:
                        # Calculate portfolio value over time
                        portfolio_history = pd.DataFrame(index=all_dates)
                        
                        for i, symbol in enumerate(portfolio_stocks):
                            if symbol in portfolio_data and portfolio_data[symbol]['history'] is not None:
                                weight = portfolio_weights[i] / 100
                                stock_prices = portfolio_data[symbol]['history']['Close']
                                stock_prices = stock_prices.reindex(all_dates, method='ffill')
                                
                                # Normalize to percentage of portfolio
                                portfolio_history[symbol] = (stock_prices / stock_prices.iloc[0]) * weight
                        
                        # Total portfolio value
                        portfolio_history['Total'] = portfolio_history.sum(axis=1)
                        
                        # Portfolio metrics
                        portfolio_return = (portfolio_history['Total'].iloc[-1] - 1) * 100
                        portfolio_returns = portfolio_history['Total'].pct_change().dropna()
                        portfolio_vol = portfolio_returns.std() * np.sqrt(252) * 100
                        
                        # Display portfolio summary
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("💼 Portfolio Return", f"{portfolio_return:+.2f}%")
                        
                        with col2:
                            st.metric("📊 Portfolio Volatility", f"{portfolio_vol:.2f}%")
                        
                        with col3:
                            sharpe = (portfolio_return - 2) / portfolio_vol if portfolio_vol > 0 else 0
                            st.metric("⚡ Sharpe Ratio", f"{sharpe:.3f}")
                        
                        with col4:
                            current_value = portfolio_history['Total'].iloc[-1]
                            st.metric("💰 Portfolio Value", f"{current_value:.3f}x")
                        
                        # Portfolio performance chart
                        fig_portfolio = go.Figure()
                        
                        # Add individual stock performance
                        for i, symbol in enumerate(portfolio_stocks):
                            if symbol in portfolio_history.columns:
                                stock_name = next(name for name, sym in ALL_STOCKS.items() if sym == symbol)
                                normalized_performance = portfolio_history[symbol] * (100/portfolio_weights[i])
                                fig_portfolio.add_trace(go.Scatter(
                                    x=all_dates,
                                    y=normalized_performance,
                                    name=f"{stock_name} ({portfolio_weights[i]:.1f}%)",
                                    line=dict(width=2, dash='dot')
                                ))
                        
                        # Add total portfolio performance
                        fig_portfolio.add_trace(go.Scatter(
                            x=all_dates,
                            y=portfolio_history['Total'],
                            name='Total Portfolio',
                            line=dict(width=4, color='gold')
                        ))
                        
                        fig_portfolio.update_layout(
                            title="💼 Portfolio Performance Analysis",
                            xaxis_title="Date",
                            yaxis_title="Normalized Value",
                            template="plotly_dark",
                            height=600
                        )
                        st.plotly_chart(fig_portfolio, use_container_width=True)
                        
                        # Portfolio allocation pie chart
                        stock_names_for_pie = [next(name for name, sym in ALL_STOCKS.items() if sym == symbol) for symbol in portfolio_stocks]
                        
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=stock_names_for_pie,
                            values=portfolio_weights,
                            hole=.3,
                            textinfo='label+percent',
                            textfont_size=12
                        )])
                        
                        fig_pie.update_layout(
                            title="🥧 Portfolio Allocation",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Correlation analysis
                        st.markdown("### 🔗 Stock Correlation Analysis")
                        
                        correlation_data = pd.DataFrame()
                        for symbol in portfolio_stocks:
                            if symbol in portfolio_data and portfolio_data[symbol]['history'] is not None:
                                stock_name = next(name for name, sym in ALL_STOCKS.items() if sym == symbol)
                                returns = portfolio_data[symbol]['history']['Close'].pct_change()
                                correlation_data[stock_name] = returns
                        
                        if len(correlation_data.columns) > 1:
                            corr_matrix = correlation_data.corr()
                            
                            fig_corr = go.Figure(data=go.Heatmap(
                                z=corr_matrix.values,
                                x=corr_matrix.columns,
                                y=corr_matrix.index,
                                colorscale='RdYlBu',
                                text=np.round(corr_matrix.values, 2),
                                texttemplate="%{text}",
                                textfont={"size":10}
                            ))
                            
                            fig_corr.update_layout(
                                title="🔗 Stock Correlation Matrix",
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                            # Diversification insights
                            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()
                            
                            if avg_correlation > 0.7:
                                diversification = "🔴 Low Diversification"
                                div_color = "red"
                            elif avg_correlation > 0.4:
                                diversification = "🟡 Medium Diversification" 
                                div_color = "orange"
                            else:
                                diversification = "🟢 High Diversification"
                                div_color = "green"
                            
                            st.markdown(f"""
                            <div class="metric-card">
                            <h4>📊 Portfolio Diversification Analysis</h4>
                            <h3 style="color: {div_color};">{diversification}</h3>
                            <p>Average correlation: {avg_correlation:.3f}</p>
                            <small>Lower correlation = better diversification</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("❌ No overlapping dates found for portfolio analysis")
                        
                except Exception as e:
                    st.error(f"❌ Error in portfolio analysis: {str(e)}")
            else:
                st.error("❌ Could not load enough portfolio data for analysis")
    
    # Enhanced sidebar with troubleshooting
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Advanced Features")
    
    if st.sidebar.button("📊 Market Overview"):
        try:
            st.markdown("### 🌍 Market Overview Dashboard")
            
            # Major indices
            indices = {
                "S&P 500": "^GSPC",
                "Dow Jones": "^DJI", 
                "NASDAQ": "^IXIC",
                "Russell 2000": "^RUT"
            }
            
            st.markdown("#### 📈 Major Market Indices")
            col1, col2, col3, col4 = st.columns(4)
            cols = [col1, col2, col3, col4]
            
            for i, (name, symbol) in enumerate(indices.items()):
                with cols[i]:
                    try:
                        index_data = yf.Ticker(symbol).history(period="2d")
                        if len(index_data) >= 2:
                            current = index_data['Close'][-1]
                            previous = index_data['Close'][-2]
                            change = current - previous
                            change_pct = (change / previous) * 100
                            st.metric(name, f"{current:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
                        else:
                            st.metric(name, "N/A", "Insufficient data")
                    except:
                        st.metric(name, "N/A", "Data unavailable")
        except Exception as e:
            st.error("Market overview temporarily unavailable")
    
    # Troubleshooting section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Troubleshooting")
    st.sidebar.markdown("""
    **If you see data loading errors:**
    - ✅ Enable "Use Demo Data" option
    - 🔄 Try a different time period (1mo or 3mo)
    - 📶 Check your internet connection  
    - 🔄 Refresh the page
    - 📧 Try different stocks
    
    **Demo Mode:** Uses simulated data for demonstration when real data is unavailable.
    """)
    
    # Footer with app info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 15px;">
    <h3 style="color: white; margin: 0;">🚀 Ultimate Stock Analysis Platform</h3>
    <p style="color: white; margin: 5px 0;">
    🤖 AI Predictions | 📊 Multi-Stock Comparison | 📰 News Sentiment | 💼 Portfolio Analysis
    </p>
    <p style="color: white; margin: 5px 0; font-size: 12px;">
    Built with Python, Streamlit, AI/ML | ⚠️ Educational purposes only - Not financial advice
    </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


# In[ ]:




