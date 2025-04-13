# crypto_agent.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate mock crypto data
def fetch_crypto_data(symbol='BTC/USDT', timeframe='1h', limit=100):
    try:
        # Mock OHLCV data (timestamp, open, high, low, close, volume)
        np.random.seed(42)  # For reproducibility
        timestamps = pd.date_range(end='2025-04-13', periods=limit, freq='H')
        prices = np.random.normal(85000, 1000, limit).cumsum() / 100 + 85000  # Simulate price movement
        data = {
            'timestamp': timestamps,
            'open': prices,
            'high': prices + np.random.uniform(0, 500, limit),
            'low': prices - np.random.uniform(0, 500, limit),
            'close': prices + np.random.uniform(-200, 200, limit),
            'volume': np.random.uniform(10, 100, limit)
        }
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error generating mock data: {e}")
        return None

# Add features (SMA and RSI)
def add_features(df):
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df.dropna()

# Prepare data for model
def prepare_data(df):
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    X = df[['sma_10', 'sma_50', 'rsi', 'close', 'volume']]
    y = df['target']
    return X, y, df

# Train and save model
def train_model():
    data = fetch_crypto_data()
    if data is None:
        return None, None
    data = add_features(data)
    X, y, processed_data = prepare_data(data)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'crypto_model.pkl')
    return model, processed_data

# Predict latest price movement
def predict_latest():
    model = joblib.load('crypto_model.pkl')
    latest_data = fetch_crypto_data()
    if latest_data is None:
        return None, None, None
    latest_data = add_features(latest_data)
    latest_features = latest_data[['sma_10', 'sma_50', 'rsi', 'close', 'volume']].iloc[-1:]
    prediction = model.predict(latest_features)[0]
    action = "Buy" if prediction == 1 else "Sell"
    price = latest_data['close'].iloc[-1]
    return action, price, latest_data

if __name__ == "__main__":
    model, processed_data = train_model()
    if model:
        action, price, data = predict_latest()
        if action:
            print(f"Latest Price: ${price:.2f}")
            print(f"Recommended Action: {action}")