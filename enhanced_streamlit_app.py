
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 🎨 Page Configuration
st.set_page_config(
    page_title="🚀 Multi-Stock AI Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🌟 Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stock-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 🏆 Main Header
st.markdown("""
<div class="main-header">
    <h1>🚀 AI Stock Prediction Super App</h1>
    <p>Analyze ANY Stock with Artificial Intelligence! Built by a Future Data Scientist! 🧑‍💻</p>
</div>
""", unsafe_allow_html=True)

# 📊 Sidebar for Stock Selection
st.sidebar.header("🎯 Choose Your Stock to Analyze!")

# Popular stock options
stock_options = {
    "🍎 Apple Inc.": "AAPL",
    "🔍 Google (Alphabet)": "GOOGL", 
    "⚡ Tesla": "TSLA",
    "💻 Microsoft": "MSFT",
    "🛒 Amazon": "AMZN",
    "📘 Meta (Facebook)": "META",
    "💎 NVIDIA": "NVDA",
    "🏦 JPMorgan Chase": "JPM",
    "🥤 Coca-Cola": "KO",
    "🎮 Netflix": "NFLX"
}

# Stock selector
selected_stock_name = st.sidebar.selectbox(
    "Pick a stock to analyze:",
    options=list(stock_options.keys()),
    index=0  # Default to Apple
)
selected_stock = stock_options[selected_stock_name]

# Date range selector
st.sidebar.header("📅 Choose Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365*2),  # 2 years ago
        max_value=datetime.now()
    )
with col2:
    end_date = st.date_input(
        "End Date", 
        value=datetime.now(),
        max_value=datetime.now()
    )

# Prediction settings
st.sidebar.header("🔮 Prediction Settings")
prediction_days = st.sidebar.slider("Days to predict into future:", 1, 30, 5)
sequence_length = st.sidebar.slider("AI memory length (days):", 30, 120, 60)

# 🚀 Main Analysis Section
if st.sidebar.button("🚀 Analyze This Stock!", type="primary"):
    with st.spinner(f"🤖 AI is analyzing {selected_stock_name}..."):
        try:
            # Download stock data
            stock_data = yf.download(selected_stock, start=start_date, end=end_date)
            
            if stock_data.empty:
                st.error("❌ No data found for this stock! Try different dates.")
            else:
                # Display basic stock info
                st.success(f"✅ Successfully loaded {len(stock_data)} days of {selected_stock_name} data!")
                
                # Stock overview
                st.markdown(f"""
                <div class="stock-card">
                    <h3>📊 {selected_stock_name} ({selected_stock}) Analysis</h3>
                    <p><strong>Period:</strong> {start_date} to {end_date}</p>
                    <p><strong>Total Days:</strong> {len(stock_data)} trading days</p>
                    <p><strong>Latest Price:</strong> ${stock_data['Close'].iloc[-1]:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Stock Charts", "🤖 AI Predictions", "📊 Performance", "🔍 Technical Analysis"])
                
                with tab1:
                    st.header("📈 Interactive Stock Price Charts")
                    
                    # Price chart with volume
                    fig = go.Figure()
                    
                    # Candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=stock_data.index,
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        name="Stock Price"
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_stock_name} Stock Price (Candlestick)",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        template="plotly_white",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume chart
                    fig_volume = px.bar(
                        x=stock_data.index, 
                        y=stock_data['Volume'],
                        title=f"{selected_stock_name} Trading Volume",
                        labels={'y': 'Volume', 'x': 'Date'}
                    )
                    fig_volume.update_layout(template="plotly_white", height=400)
                    st.plotly_chart(fig_volume, use_container_width=True)
                
                with tab2:
                    st.header("🤖 AI-Powered Price Predictions")
                    
                    # Prepare data for AI
                    prices = stock_data['Close'].values.reshape(-1, 1)
                    scaler = MinMaxScaler()
                    scaled_prices = scaler.fit_transform(prices)
                    
                    # Create sequences for training
                    def create_sequences(data, seq_length):
                        X, y = [], []
                        for i in range(seq_length, len(data)):
                            X.append(data[i-seq_length:i, 0])
                            y.append(data[i, 0])
                        return np.array(X), np.array(y)
                    
                    if len(scaled_prices) > sequence_length:
                        X, y = create_sequences(scaled_prices, sequence_length)
                        X = X.reshape(X.shape[0], X.shape[1], 1)
                        
                        # Build AI model
                        model = tf.keras.Sequential([
                            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                            tf.keras.layers.Dropout(0.2),
                            tf.keras.layers.LSTM(50, return_sequences=False),
                            tf.keras.layers.Dropout(0.2),
                            tf.keras.layers.Dense(25),
                            tf.keras.layers.Dense(1)
                        ])
                        
                        model.compile(optimizer='adam', loss='mean_squared_error')
                        
                        # Train the model
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        with st.spinner("🧠 Training AI model..."):
                            # Simple training (fewer epochs for demo)
                            history = model.fit(X, y, epochs=20, batch_size=32, verbose=0)
                            progress_bar.progress(100)
                            status_text.text("✅ AI training completed!")
                        
                        # Make predictions
                        train_predict = model.predict(X)
                        train_predict = scaler.inverse_transform(train_predict)
                        
                        # Future predictions
                        last_sequence = scaled_prices[-sequence_length:]
                        future_predictions = []
                        
                        for _ in range(prediction_days):
                            next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
                            future_predictions.append(next_pred[0, 0])
                            last_sequence = np.append(last_sequence[1:], next_pred)
                        
                        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
                        
                        # Create future dates
                        last_date = stock_data.index[-1]
                        future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
                        
                        # Plot predictions
                        fig_pred = go.Figure()
                        
                        # Historical prices
                        fig_pred.add_trace(go.Scatter(
                            x=stock_data.index,
                            y=stock_data['Close'],
                            mode='lines',
                            name='Historical Prices',
                            line=dict(color='blue')
                        ))
                        
                        # Future predictions
                        fig_pred.add_trace(go.Scatter(
                            x=future_dates,
                            y=future_predictions.flatten(),
                            mode='lines+markers',
                            name=f'AI Predictions ({prediction_days} days)',
                            line=dict(color='red', dash='dash'),
                            marker=dict(size=8)
                        ))
                        
                        fig_pred.update_layout(
                            title=f"🔮 AI Predictions for {selected_stock_name}",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            template="plotly_white",
                            height=600
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Prediction summary
                        current_price = stock_data['Close'].iloc[-1]
                        future_price = future_predictions[-1][0]
                        price_change = future_price - current_price
                        price_change_pct = (price_change / current_price) * 100
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>📊 Current Price</h4>
                                <h2>${current_price:.2f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>🔮 Predicted Price</h4>
                                <h2>${future_price:.2f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            color = "green" if price_change > 0 else "red"
                            arrow = "↗️" if price_change > 0 else "↘️"
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>💰 Price Change</h4>
                                <h2 style="color: {color}">{arrow} ${abs(price_change):.2f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>📈 Percentage</h4>
                                <h2 style="color: {color}">{price_change_pct:+.2f}%</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Prediction confidence
                        if price_change_pct > 5:
                            sentiment = "🚀 Strong Upward Trend Predicted!"
                            sentiment_color = "green"
                        elif price_change_pct < -5:
                            sentiment = "📉 Downward Trend Predicted"
                            sentiment_color = "red"
                        else:
                            sentiment = "➡️ Stable Price Movement Expected"
                            sentiment_color = "orange"
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>🤖 AI Analysis Result</h3>
                            <h2 style="color: {sentiment_color}">{sentiment}</h2>
                            <p>Based on {sequence_length} days of historical data analysis</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    else:
                        st.warning(f"⚠️ Need at least {sequence_length} days of data for AI analysis. Try selecting a longer date range!")
                
                with tab3:
                    st.header("📊 Stock Performance Metrics")
                    
                    # Calculate metrics
                    returns = stock_data['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                    avg_return = returns.mean() * 252  # Annualized return
                    
                    # Price statistics
                    price_stats = {
                        "Highest Price": stock_data['High'].max(),
                        "Lowest Price": stock_data['Low'].min(),
                        "Average Price": stock_data['Close'].mean(),
                        "Price Range": stock_data['High'].max() - stock_data['Low'].min()
                    }
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("💰 Price Statistics")
                        for metric, value in price_stats.items():
                            st.metric(metric, f"${value:.2f}")
                    
                    with col2:
                        st.subheader("📈 Performance Metrics")
                        st.metric("Annual Return", f"{avg_return:.2%}")
                        st.metric("Volatility", f"{volatility:.2%}")
                        st.metric("Total Days", len(stock_data))
                        
                    # Moving averages
                    stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
                    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
                    
                    fig_ma = go.Figure()
                    fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Price', line=dict(color='blue')))
                    fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA20'], name='20-day MA', line=dict(color='orange')))
                    fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA50'], name='50-day MA', line=dict(color='red')))
                    
                    fig_ma.update_layout(
                        title=f"{selected_stock_name} with Moving Averages",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        template="plotly_white",
                        height=500
                    )
                    
                    st.plotly_chart(fig_ma, use_container_width=True)
                
                with tab4:
                    st.header("🔍 Advanced Technical Analysis")
                    
                    # RSI calculation
                    def calculate_rsi(prices, window=14):
                        delta = prices.diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        return rsi
                    
                    stock_data['RSI'] = calculate_rsi(stock_data['Close'])
                    
                    # Bollinger Bands
                    stock_data['BB_Middle'] = stock_data['Close'].rolling(window=20).mean()
                    bb_std = stock_data['Close'].rolling(window=20).std()
                    stock_data['BB_Upper'] = stock_data['BB_Middle'] + (bb_std * 2)
                    stock_data['BB_Lower'] = stock_data['BB_Middle'] - (bb_std * 2)
                    
                    # Plot Bollinger Bands
                    fig_bb = go.Figure()
                    fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Price', line=dict(color='blue')))
                    fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BB_Upper'], name='Upper Band', line=dict(color='red', dash='dash')))
                    fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BB_Lower'], name='Lower Band', line=dict(color='green', dash='dash')))
                    fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BB_Middle'], name='Middle Band', line=dict(color='orange')))
                    
                    fig_bb.update_layout(
                        title=f"{selected_stock_name} Bollinger Bands",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        template="plotly_white",
                        height=500
                    )
                    
                    st.plotly_chart(fig_bb, use_container_width=True)
                    
                    # RSI plot
                    fig_rsi = px.line(x=stock_data.index, y=stock_data['RSI'], title=f"{selected_stock_name} RSI (Relative Strength Index)")
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                    fig_rsi.update_layout(template="plotly_white", height=400)
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # Current RSI interpretation
                    current_rsi = stock_data['RSI'].iloc[-1]
                    if current_rsi > 70:
                        rsi_signal = "⚠️ OVERBOUGHT - Stock might be due for a pullback"
                        rsi_color = "red"
                    elif current_rsi < 30:
                        rsi_signal = "🟢 OVERSOLD - Stock might be due for a bounce"
                        rsi_color = "green"
                    else:
                        rsi_signal = "⚖️ NEUTRAL - Stock is in normal trading range"
                        rsi_color = "orange"
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>📊 Technical Analysis Signal</h3>
                        <h4>Current RSI: {current_rsi:.1f}</h4>
                        <h3 style="color: {rsi_color}">{rsi_signal}</h3>
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Error analyzing stock: {str(e)}")
            st.info("💡 Try selecting a different stock or date range!")

else:
    # Welcome message when no stock is selected
    st.info("👆 Choose a stock from the sidebar and click 'Analyze This Stock!' to get started!")
    
    # Show popular stocks info
    st.header("🌟 Popular Stocks Available for Analysis")
    
    cols = st.columns(3)
    stock_info = [
        ("🍎 Apple (AAPL)", "Technology giant, iPhone maker"),
        ("⚡ Tesla (TSLA)", "Electric vehicles & clean energy"),
        ("🔍 Google (GOOGL)", "Search engine & cloud services"),
        ("💻 Microsoft (MSFT)", "Software & cloud computing"),
        ("🛒 Amazon (AMZN)", "E-commerce & cloud services"),
        ("📘 Meta (META)", "Social media platforms"),
        ("💎 NVIDIA (NVDA)", "AI chips & graphics cards"),
        ("🏦 JPMorgan (JPM)", "Banking & financial services"),
        ("🥤 Coca-Cola (KO)", "Beverages & consumer goods"),
        ("🎮 Netflix (NFLX)", "Streaming & entertainment")
    ]
    
    for i, (name, desc) in enumerate(stock_info):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="stock-card">
                <h4>{name}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# 🎓 Educational Footer
st.markdown("---")
st.markdown("""
### ⚠️ Educational Disclaimer
This AI-powered stock analysis tool is created for educational purposes only! 
- 📚 **Learning Focus**: Demonstrates advanced AI and data science skills
- 💡 **Not Financial Advice**: Never use for real investment decisions
- 👨‍👩‍👧‍👦 **Adult Supervision**: Always consult adults about financial matters
- 🎯 **Skill Building**: Shows readiness for STEM careers and advanced education

**Built with passion for learning and innovation!** 🚀🧑‍💻
""")
