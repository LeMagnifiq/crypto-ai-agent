# app.py
from flask import Flask, render_template
from crypto_agent import predict_latest

app = Flask(__name__)

@app.route('/')
def home():
    action, price, _ = predict_latest()
    if action:
        return render_template('index.html', price=price, action=action)
    else:
        return "<h1>Crypto AI Agent</h1><p>Error: Unable to fetch prediction</p>"

if __name__ == '__main__':
    app.run(debug=True)