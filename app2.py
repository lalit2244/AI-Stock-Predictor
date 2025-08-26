#!/usr/bin/env python
# coding: utf-8

# In[7]:


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

def get_stock_data(symbols, period="1y"):
    """Enhanced stock data fetching with error handling"""
    if isinstance(symbols, str):
        symbols = [symbols]
    
    data = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            hist_data = stock.history(period=period)
            info = stock.info
            
            if not hist_data.empty:
                data[symbol] = {
                    'history': hist_data,
                    'info': info
                }
        except Exception as e:
            st.warning(f"Could not load data for {symbol}")
    
    return data

def get_news_sentiment(symbol, limit=5):
    """Get news and basic sentiment analysis"""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news[:limit]
        
        news_data = []
        for article in news:
            title = article.get('title', '')
            
            # Simple keyword-based sentiment
            positive_words = ['gains', 'up', 'rise', 'bull', 'positive', 'growth', 'strong', 'beat']
            negative_words = ['falls', 'down', 'drop', 'bear', 'negative', 'decline', 'weak', 'miss']
            
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
    except Exception:
        return []

def advanced_technical_analysis(data):
    """Calculate advanced technical indicators"""
    df = data.copy()
    
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    return df

def predict_stock_price_advanced(data):
    """Enhanced prediction using Random Forest"""
    if data is None or len(data) < 50:
        return None, None, None
    
    try:
        # Technical analysis
        df = advanced_technical_analysis(data)
        df = df.dropna()
        
        if len(df) < 30:
            return None, None, None
        
        # Prepare features
        features = ['MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD', 'Volatility', 'Volume']
        available_features = [f for f in features if f in df.columns and not df[f].isna().all()]
        
        if len(available_features) < 3:
            return None, None, None
        
        X = df[available_features].values
        y = df['Close'].values
        
        # Random Forest Model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        rf_pred = rf_model.predict(X[-1].reshape(1, -1))[0]
        rf_score = rf_model.score(X_test, y_test)
        rf_confidence = min(max(rf_score * 100, 0), 95)
        
        model_info = {
            'rf_prediction': rf_pred,
            'rf_confidence': rf_confidence
        }
        
        return rf_pred, rf_confidence, model_info
        
    except Exception as e:
        return None, None, None

def compare_stocks(stock_data):
    """Advanced stock comparison analysis"""
    if len(stock_data) < 2:
        return None
    
    comparison_results = {}
    
    for symbol, data in stock_data.items():
        hist = data['history']
        
        # Calculate metrics
        current_price = hist['Close'][-1]
        start_price = hist['Close'][0]
        total_return = ((current_price - start_price) / start_price) * 100
        
        # Volatility
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Sharpe Ratio
        excess_returns = daily_returns.mean() - (0.02/252)
        sharpe_ratio = (excess_returns / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        comparison_results[symbol] = {
            'current_price': current_price,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_volume': hist['Volume'].mean()
        }
    
    return comparison_results

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
        
        # Main analysis
        st.markdown(f"## 📊 {selected_stock_name} ({symbol}) - Comprehensive Analysis")
        
        # Fetch data
        with st.spinner("🔄 Loading comprehensive stock data..."):
            stock_data = get_stock_data([symbol], period)
        
        if symbol in stock_data:
            data = stock_data[symbol]['history']
            info = stock_data[symbol]['info']
            
            # Current metrics
            current_price = data['Close'][-1]
            price_change = data['Close'][-1] - data['Close'][-2]
            change_percent = (price_change / data['Close'][-2]) * 100
            
            # Enhanced metrics display
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("💰 Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({change_percent:+.1f}%)")
            
            with col2:
                st.metric("📈 52W High", f"${data['High'].max():.2f}")
            
            with col3:
                st.metric("📉 52W Low", f"${data['Low'].min():.2f}")
            
            with col4:
                market_cap = info.get('marketCap', 0)
                if market_cap > 1e12:
                    market_cap_display = f"${market_cap/1e12:.1f}T"
                elif market_cap > 1e9:
                    market_cap_display = f"${market_cap/1e9:.1f}B"
                elif market_cap > 1e6:
                    market_cap_display = f"${market_cap/1e6:.1f}M"
                else:
                    market_cap_display = f"${market_cap:,.0f}"
                st.metric("🏢 Market Cap", market_cap_display)
            
            with col5:
                st.metric("📊 Volume", f"{data['Volume'][-1]:,.0f}")
            
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
            
            # Add moving averages
            df_tech = advanced_technical_analysis(data)
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MA_20'], name='MA 20', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MA_50'], name='MA 50', line=dict(color='red')))
            
            fig.update_layout(
                title=f"📈 {selected_stock_name} - Advanced Chart",
                xaxis_title="Date",
                height=600,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
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
                    if 'RSI' in df_tech.columns:
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=df_tech.index, y=df_tech['RSI'], name='RSI', line=dict(color='purple')))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                        fig_rsi.update_layout(title="📈 RSI Indicator", template="plotly_dark")
                        st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    # MACD Chart
                    if 'MACD' in df_tech.columns:
                        fig_macd = go.Figure()
                        fig_macd.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD'], name='MACD', line=dict(color='blue')))
                        fig_macd.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD_Signal'], name='Signal', line=dict(color='red')))
                        fig_macd.update_layout(title="📊 MACD Indicator", template="plotly_dark")
                        st.plotly_chart(fig_macd, use_container_width=True)
                
                # Technical Summary
                current_rsi = df_tech['RSI'].iloc[-1] if 'RSI' in df_tech.columns and not df_tech['RSI'].isna().all() else None
                current_macd = df_tech['MACD'].iloc[-1] if 'MACD' in df_tech.columns and not df_tech['MACD'].isna().all() else None
                
                st.markdown("#### 📋 Technical Indicators Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if current_rsi:
                        rsi_signal = "🔴 Overbought" if current_rsi > 70 else "🟢 Oversold" if current_rsi < 30 else "🟡 Neutral"
                        st.metric("RSI", f"{current_rsi:.1f}", rsi_signal)
                    else:
                        st.metric("RSI", "N/A", "Calculating...")
                
                with col2:
                    if current_macd:
                        macd_signal = "🟢 Bullish" if current_macd > 0 else "🔴 Bearish"
                        st.metric("MACD", f"{current_macd:.3f}", macd_signal)
                    else:
                        st.metric("MACD", "N/A", "Calculating...")
                
                with col3:
                    volatility = df_tech['Volatility'].iloc[-1] if 'Volatility' in df_tech.columns and not df_tech['Volatility'].isna().all() else None
                    if volatility:
                        vol_level = "🔴 High" if volatility > df_tech['Volatility'].quantile(0.8) else "🟡 Medium" if volatility > df_tech['Volatility'].quantile(0.4) else "🟢 Low"
                        st.metric("Volatility", f"{volatility:.2f}", vol_level)
                    else:
                        st.metric("Volatility", "N/A", "Calculating...")
                
                with col4:
                    ma_20 = df_tech['MA_20'].iloc[-1] if 'MA_20' in df_tech.columns and not df_tech['MA_20'].isna().all() else None
                    if ma_20 and current_price:
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
                                if news['url']:
                                    st.markdown(f"[📖 Read Article]({news['url']})")
                            with col2:
                                st.markdown(f"**Sentiment:** {news['sentiment']}")
                                st.progress(news['sentiment_score'])
                else:
                    st.info("📰 No recent news available for this stock")
        else:
            st.error("❌ Could not load stock data. Please try again!")
    
    elif analysis_mode == "Multi-Stock Comparison":
        # Multi-stock comparison
        st.sidebar.markdown("### 📊 Stock Comparison")
        
        # Select multiple stocks
        selected_stocks = []
        for category, stocks in STOCK_CATEGORIES.items():
            selected = st.sidebar.multiselect(f"{category}:", list(stocks.keys()), key=category)
            for stock_name in selected:
                selected_stocks.append((stock_name, stocks[stock_name]))
        
        if len(selected_stocks) == 0:
            st.warning("Please select at least 2 stocks for comparison")
            return
        
        # Time period
        period = st.sidebar.selectbox("⏰ Comparison Period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
        
        st.markdown(f"## 🔍 Comparing {len(selected_stocks)} Stocks")
        
        # Fetch data for all selected stocks
        symbols = [stock[1] for stock in selected_stocks]
        with st.spinner(f"📊 Loading data for {len(symbols)} stocks..."):
            comparison_data = get_stock_data(symbols, period)
        
        if comparison_data and len(comparison_data) >= 2:
            # Stock comparison metrics
            comparison_results = compare_stocks(comparison_data)
            
            if comparison_results:
                # Comparison table
                st.markdown("### 📊 Performance Comparison")
                
                df_comparison = pd.DataFrame(comparison_results).T
                df_comparison.index = [next(name for name, sym in selected_stocks if sym == idx) for idx in df_comparison.index]
                
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
                    stock_names = list(df_comparison.index)
                    returns = df_comparison['total_return'].values
                    colors = ['green' if r > 0 else 'red' for r in returns]
                    
                    fig_returns.add_trace(go.Bar(
                        x=stock_names,
                        y=returns,
                        marker_color=colors,
                        name='Total Return'
                    ))
                    fig_returns.update_layout(title="📊 Total Return Comparison", yaxis_title="Return (%)")
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
                    fig_risk.update_layout(title="🎯 Risk vs Return", xaxis_title="Volatility (%)", yaxis_title="Return (%)")
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                # Price comparison chart
                st.markdown("### 📈 Price Performance Comparison")
                
                fig_price_comp = go.Figure()
                
                for stock_name, symbol in selected_stocks:
                    if symbol in comparison_data:
                        hist = comparison_data[symbol]['history']
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
        else:
            st.error("❌ Could not load enough stock data for comparison")
    
    elif analysis_mode == "Portfolio Analysis":
        st.sidebar.markdown("### 💼 Portfolio Builder")
        
        st.markdown("## 💼 Custom Portfolio Analysis")
        st.info("📝 Build your custom portfolio by selecting stocks and their weights")
        
        # Portfolio creation
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
            
            with st.spinner("📊 Analyzing your portfolio..."):
                portfolio_data = get_stock_data(portfolio_stocks, period)
            
            if portfolio_data and len(portfolio_data) == len(portfolio_stocks):
                # Calculate portfolio performance
                all_dates = None
                for symbol in portfolio_stocks:
                    if symbol in portfolio_data:
                        dates = portfolio_data[symbol]['history'].index
                        if all_dates is None:
                            all_dates = dates
                        else:
                            all_dates = all_dates.intersection(dates)
                
                if all_dates is not None and len(all_dates) > 0:
                    # Calculate portfolio value over time
                    portfolio_history = pd.DataFrame(index=all_dates)
                    
                    for i, symbol in enumerate(portfolio_stocks):
                        if symbol in portfolio_data:
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
                        if symbol in portfolio_data:
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
            else:
                st.error("❌ Could not load portfolio data")
    
    # Advanced Features Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Advanced Features")
    
    if st.sidebar.button("📊 Market Overview"):
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
                except:
                    st.metric(name, "N/A", "Data unavailable")
    
    if st.sidebar.button("💰 Crypto Dashboard"):
        st.markdown("### ₿ Cryptocurrency Dashboard")
        
        crypto_symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "DOT-USD"]
        crypto_names = ["Bitcoin", "Ethereum", "Binance Coin", "Cardano", "Polkadot"]
        
        crypto_data = {}
        for symbol in crypto_symbols:
            try:
                data = yf.Ticker(symbol).history(period="7d")
                if not data.empty:
                    crypto_data[symbol] = data
            except:
                continue
        
        if crypto_data:
            cols = st.columns(len(crypto_data))
            for i, (symbol, data) in enumerate(crypto_data.items()):
                with cols[i]:
                    current = data['Close'][-1]
                    change = data['Close'][-1] - data['Close'][0]
                    change_pct = (change / data['Close'][0]) * 100
                    name = crypto_names[crypto_symbols.index(symbol)]
                    st.metric(name, f"${current:,.2f}", f"{change_pct:+.2f}%")
    
    if st.sidebar.button("📈 Economic Indicators"):
        st.markdown("### 📊 Economic Indicators Dashboard")
        
        st.info("📈 Key Economic Metrics")
        
        # Treasury yields
        treasury_symbols = {
            "10-Year Treasury": "^TNX",
            "2-Year Treasury": "^IRX", 
            "30-Year Treasury": "^TYX"
        }
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for i, (name, symbol) in enumerate(treasury_symbols.items()):
            with cols[i]:
                try:
                    treasury_data = yf.Ticker(symbol).history(period="5d")
                    if not treasury_data.empty:
                        current_yield = treasury_data['Close'][-1]
                        st.metric(name, f"{current_yield:.2f}%")
                except:
                    st.metric(name, "N/A")
    
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
    
    # Advanced sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 🎯 Features Included:
    - 🤖 **AI Predictions** (Random Forest + Technical Analysis)
    - 📊 **Multi-Stock Comparison**
    - 💼 **Portfolio Analysis** 
    - 📰 **News Sentiment Analysis**
    - 📈 **Advanced Technical Analysis**
    - 🔗 **Correlation Analysis**
    - 🌍 **Market Overview**
    - ₿ **Cryptocurrency Dashboard**
    - 📊 **Economic Indicators**
    - 🎨 **Multiple Themes**
    """)

if __name__ == "__main__":
    main()


# In[8]:


def get_news_sentiment(symbol, limit=5):
    """Enhanced news and sentiment analysis with better keyword detection"""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news[:limit]
        
        news_data = []
        for article in news:
            title = article.get('title', '')
            
            # Expanded keyword lists for better detection
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
            
            # More sophisticated sentiment scoring
            title_lower = title.lower()
            
            # Count weighted keywords (some words are stronger indicators)
            strong_positive = ['surge', 'soar', 'rally', 'jump', 'breakthrough', 'record']
            strong_negative = ['plunge', 'crash', 'tumble', 'slump', 'warning']
            
            pos_score = 0
            neg_score = 0
            
            # Regular positive/negative words worth 1 point each
            pos_score += sum(1 for word in positive_words if word in title_lower)
            neg_score += sum(1 for word in negative_words if word in title_lower)
            
            # Strong words worth 2 points each
            pos_score += sum(2 for word in strong_positive if word in title_lower)
            neg_score += sum(2 for word in strong_negative if word in title_lower)
            
            # Calculate sentiment with more nuanced scoring
            total_score = pos_score + neg_score
            
            if total_score == 0:
                sentiment = "Neutral ➡️"
                sentiment_score = 0.5
            else:
                sentiment_ratio = pos_score / total_score
                
                if sentiment_ratio >= 0.7:
                    sentiment = "Very Positive 🚀"
                    sentiment_score = 0.9
                elif sentiment_ratio >= 0.6:
                    sentiment = "Positive 📈"
                    sentiment_score = 0.7
                elif sentiment_ratio >= 0.4:
                    sentiment = "Neutral ➡️"
                    sentiment_score = 0.5
                elif sentiment_ratio >= 0.3:
                    sentiment = "Negative 📉"
                    sentiment_score = 0.3
                else:
                    sentiment = "Very Negative 💥"
                    sentiment_score = 0.1
            
            news_data.append({
                'title': title,
                'url': article.get('link', ''),
                'published': datetime.fromtimestamp(article.get('providerPublishTime', 0)),
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'pos_score': pos_score,
                'neg_score': neg_score
            })
        
        return news_data
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return []

# Alternative: Add a debug mode to see what's happening
def get_news_sentiment_debug(symbol, limit=5):
    """Debug version to see why sentiment is always neutral"""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news[:limit]
        
        print(f"Found {len(news)} news articles for {symbol}")
        
        for i, article in enumerate(news):
            title = article.get('title', '')
            print(f"Article {i+1}: {title}")
            
            # Check for keywords
            positive_words = ['gains', 'up', 'rise', 'bull', 'positive', 'growth', 'strong', 'beat']
            negative_words = ['falls', 'down', 'drop', 'bear', 'negative', 'decline', 'weak', 'miss']
            
            title_lower = title.lower()
            pos_matches = [word for word in positive_words if word in title_lower]
            neg_matches = [word for word in negative_words if word in title_lower]
            
            print(f"  Positive matches: {pos_matches}")
            print(f"  Negative matches: {neg_matches}")
            print("---")
        
        # Return normal sentiment analysis
        return get_news_sentiment(symbol, limit)
        
    except Exception as e:
        print(f"Debug error: {e}")
        return []


# In[ ]:




