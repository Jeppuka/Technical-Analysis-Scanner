import time
import requests
import os
from live_crypto_data import get_current_price, generate_hourly_plot
import schedule

def send_telegram_message(token, chat_id, message):
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

def send_telegram_image(token, chat_id, image_path):
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

telegram_token = os.getenv("TELEGRAM_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

def send_notifications():
    tickers = ["ETH", "BTC"]
    for ticker in tickers:
        price = get_current_price(ticker)
        message = f"The current price of {ticker} is (USD): {price}$"
        image = generate_hourly_plot(ticker)
        send_telegram_message(telegram_token, telegram_chat_id, message)
        send_telegram_image(telegram_token, telegram_chat_id, f"{ticker}_hourly.jpg")

# Run tasks every minute (60 seconds)
schedule.every().hour.at(":00").do(send_notifications)
while True:
    schedule.run_pending()
    time.sleep(1)

#print(send_telegram_message(telegram_token, telegram_chat_id, "plot.jpg"))





