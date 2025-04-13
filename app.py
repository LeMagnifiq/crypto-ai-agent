from flask import Flask, render_template, request
from crypto_agent import CryptoAgent

app = Flask(__name__)
agent = CryptoAgent('crypto_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    coin = request.form.get('coin', 'bitcoin')
    alert_price = request.form.get('alert_price', '')
    try:
        price, action, prices, timestamps, trend = agent.predict(coin)
    except Exception as e:
        print(f"Error in predict for {coin}: {e}")
        price, action, prices, timestamps, trend = 0, "Error", [], [], "Unknown"
    
    coin_name = 'Bitcoin' if coin == 'bitcoin' else 'Ethereum'
    
    # Alert logic
    alert_message = None
    if alert_price.strip():
        try:
            alert_price = float(alert_price)
            if price > 0 and price < alert_price:  # Check valid price
                alert_message = f"{coin_name} price (${price:.2f}) is below your alert threshold (${alert_price:.2f})!"
        except ValueError:
            alert_message = "Please enter a valid number for the alert price."
    
    return render_template('index.html', price=price, action=action, prices=prices, 
                         timestamps=timestamps, coin=coin, coin_name=coin_name, 
                         trend=trend, alert_price=alert_price, alert_message=alert_message)

if __name__ == '__main__':
    app.run(debug=True)