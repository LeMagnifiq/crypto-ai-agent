from flask import Flask, render_template, request
from crypto_agent import CryptoAgent

app = Flask(__name__)
agent = CryptoAgent('crypto_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    coin = request.form.get('coin', 'bitcoin')  # Default to bitcoin
    price, action, prices, timestamps = agent.predict(coin)
    coin_name = 'Bitcoin' if coin == 'bitcoin' else 'Ethereum'
    return render_template('index.html', price=price, action=action, prices=prices, 
                         timestamps=timestamps, coin=coin, coin_name=coin_name)

if __name__ == '__main__':
    app.run(debug=True)