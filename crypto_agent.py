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
        # Debug: Print expected feature names
        try:
            print("Expected model features:", self.model.feature_names_in_)
        except AttributeError:
            print("Model does not store feature names.")

    def fetch_price_history(self, days=50):
        try:
            url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            params = {
                "vs_currency": "usd",  # Use usd instead of usdt
                "days": days
            }
            # Set up session with retries
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retries))
            response = session.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            prices = [point[1] for point in data["prices"]]
            volumes = [point[1] for point in data["total_volumes"]]
            return prices, volumes
        except Exception as e:
            print(f"Error fetching price history: {e}")
            return [100000.0] * 50, [1000.0] * 50  # Fallback

    def predict(self):
        prices, volumes = self.fetch_price_history()
        if len(prices) < 50:
            prices = [100000.0] * (50 - len(prices)) + prices
            volumes = [1000.0] * (50 - len(volumes)) + volumes
        
        # Calculate features
        price = prices[-1]  # Latest price as close
        volume = volumes[-1]  # Latest volume
        sma_10 = calculate_sma(prices, 10)
        sma_50 = calculate_sma(prices, 50)
        rsi = calculate_rsi(prices, 14)
        
        # Create feature DataFrame in model's expected order
        features = pd.DataFrame({
            'sma_10': [sma_10],
            'sma_50': [sma_50],
            'rsi': [rsi],
            'close': [price],
            'volume': [volume]
        })
        
        # Debug: Print features being passed
        print("Features passed to model:", features.columns.tolist())
        
        prediction = self.model.predict(features)[0]
        action = "Buy" if prediction == 1 else "Sell"
        return price, action