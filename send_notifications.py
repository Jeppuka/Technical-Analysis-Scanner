import time
import requests
import os
from live_crypto_data import get_current_price, generate_hourly_plot, get_hourly_data
from classifier_methods import classify_single_array
import schedule
import matplotlib.pyplot as plt

def send_telegram_message(message):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

def send_telegram_image(image_path):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f'https://api.telegram.org/bot{token}/sendPhoto'
    with open(image_path, 'rb') as image_file:
        data = {
            'chat_id': chat_id
        }
        files = {
            'photo': image_file
        }
        response = requests.post(url, data=data, files=files)
    return response.json()

def plot_prices(prices, pattern):
    plt.figure(figsize=(10, 5))
    plt.plot(prices, marker='o', linestyle='-', color='b')
    plt.title(pattern)
    plt.xlabel("Time Steps (minutes)")
    plt.ylabel("Normalized Price")
    plt.grid(True)
    plt.savefig("pattern.jpg")


def send_notifications():
    tickers = ["ETH"]
    for ticker in tickers:
        price = get_current_price(ticker)
        message = f"The current price of {ticker} is USD: ${price}"
        image = generate_hourly_plot(ticker)
        send_telegram_message(message)
        send_telegram_image(f"{ticker}_hourly.jpg")

def normalize(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) * (1.2 - 1.01) + 1.01 for x in data]


# Run tasks every minute (60 seconds)

schedule.every().hour.at(":00").do(send_notifications)
while True:
    schedule.run_pending()
    data = get_hourly_data("ETH")['close']
    data = data[len(data)-15:]
    #print(data)
    data = normalize(data)
    #print(data)
    outcome = classify_single_array(data)
    if(outcome != "no breakout pattern"):
        send_telegram_message(f"Breakout pattern detected: {outcome}")
        plot_prices(data, outcome)
        send_telegram_image("pattern.jpg")
    else:
        print(outcome)
    time.sleep(60)

#print(send_telegram_message(telegram_token, telegram_chat_id, "plot.jpg"))





