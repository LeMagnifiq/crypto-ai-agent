import pandas as pd
import joblib
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = deltas.clip(min=0)
    losses = -deltas.clip(max=0)
    avg_gain = pd.Series(gains).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(losses).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_sma(prices, period):
    return pd.Series(prices).rolling(window=period, min_periods=1).mean().iloc[-1]

class CryptoAgent:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        try:
            print("Expected model features:", self.model.feature_names_in_)
        except AttributeError:
            print("Model does not store feature names.")

    def fetch_price_history(self, coin='bitcoin', days=30):  # Reduced from 50
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "daily"  # Reduce data points
            }
            session = requests.Session()
            retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retries))
            response = session.get(url, params=params, timeout=10)  # Reduced timeout
            response.raise_for_status()
            data = response.json()
            prices = [point[1] for point in data["prices"]]
            volumes = [point[1] for point in data["total_volumes"]]
            timestamps = [point[0] for point in data["prices"]]
            # Limit to 30 points
            prices = prices[-30:]
            volumes = volumes[-30:]
            timestamps = timestamps[-30:]
            return prices, volumes, timestamps
        except requests.exceptions.RequestException as e:
            print(f"Error fetching price history for {coin}: {e}")
            return [100000.0] * 30, [1000.0] * 30, [0] * 30

    def predict(self, coin='bitcoin'):
        prices, volumes, timestamps = self.fetch_price_history(coin)
        if len(prices) < 30:
            prices = [100000.0] * (30 - len(prices)) + prices
            volumes = [1000.0] * (30 - len(volumes)) + volumes
            timestamps = [0] * (30 - len(timestamps)) + timestamps
        
        price = prices[-1]
        volume = volumes[-1]
        sma_10 = calculate_sma(prices, 10)
        sma_50 = calculate_sma(prices, min(30, len(prices)))  # Adjust for shorter data
        rsi = calculate_rsi(prices, 14)
        
        # Trend indicator
        trend = 'Neutral'
        if sma_10 > sma_50:
            trend = 'Uptrend'
        elif sma_10 < sma_50:
            trend = 'Downtrend'
        
        features = pd.DataFrame({
            'sma_10': [sma_10],
            'sma_50': [sma_50],
            'rsi': [rsi],
            'close': [price],
            'volume': [volume]
        })
        
        print(f"Features passed to model for {coin}:", features.columns.tolist())
        
        try:
            prediction = self.model.predict(features)[0]
            action = "Buy" if prediction == 1 else "Sell"
        except Exception as e:
            print(f"Prediction error for {coin}: {e}")
            action = "Hold"  # Fallback
        
        return price, action, prices, timestamps, trend