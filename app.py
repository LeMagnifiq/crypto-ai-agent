from flask import Flask, render_template
from crypto_agent import CryptoAgent

app = Flask(__name__)
agent = CryptoAgent('crypto_model.pkl')

@app.route('/')
def index():
    price, action = agent.predict()
    return render_template('index.html', price=price, action=action)

if __name__ == '__main__':
    app.run(debug=True)