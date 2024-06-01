import time
import requests
import os
from live_crypto_data import get_current_price


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



telegram_token = os.getenv("TELEGRAM_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

def run_indefinitely(interval):
    while True:
        price = get_current_price("ETH")
        message = f"The current price of Ethereum is (USD): {price} "
        send_telegram_message(telegram_token, telegram_chat_id, message)
        time.sleep(interval)

# Run tasks every minute (60 seconds)
run_indefinitely(60)




