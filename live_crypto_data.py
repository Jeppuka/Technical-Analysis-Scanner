import requests
import os
api_key = os.getenv("CRYPTO_COMPARE_API_KEY")
print(api_key)




def get_current_price(symbol): #returns the current price of a crypto
    url = f"https://min-api.cryptocompare.com/data/price?fsym={symbol}&tsyms=USD"
    response = requests.get(url)
    data = response.json()
    return data["USD"]



def get_hourly_data(symbol): #returns the 60 minute datapoints for a crypto
    url = f'https://min-api.cryptocompare.com/data/v2/histominute?fsym={symbol}&tsym=USD&limit=59'
    response = requests.get(url)
    data = response.json()
    prices = [entry['close'] for entry in data['Data']['Data']]
    return prices

def get_minute_price(symbol, period): #get a minute price for a specified number of minutes
    if(period <= 1):
        url = f'https://min-api.cryptocompare.com/data/v2/histominute?fsym={symbol}&tsym=USD&limit={1}'
    else:
        url = f'https://min-api.cryptocompare.com/data/v2/histominute?fsym={symbol}&tsym=USD&limit={period-1}'
    response = requests.get(url)
    data = response.json()
    prices = [entry['close'] for entry in data['Data']['Data']]
    return prices

print(get_minute_price("ETH", 10))

