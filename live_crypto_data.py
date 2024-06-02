import requests
import os
import matplotlib.pyplot as plt
from datetime import datetime
import mplfinance as mpf
import pandas as pd
import matplotlib.dates as mdates

api_key = os.getenv("CRYPTO_COMPARE_API_KEY")
# print(api_key) # Uncomment this line if you want to print the API key

def get_current_price(symbol): #returns the current price of a crypto
    url = f"https://min-api.cryptocompare.com/data/price?fsym={symbol}&tsyms=USD"
    response = requests.get(url)
    data = response.json()
    return data["USD"]

def get_hourly_data(symbol): #returns the 60 minute datapoints for a crypto
    url = f'https://min-api.cryptocompare.com/data/v2/histominute?fsym={symbol}&tsym=USD&limit=59'
    response = requests.get(url)
    data = response.json()

    high = [entry['high'] for entry in data['Data']['Data']]
    low = [entry['low'] for entry in data['Data']['Data']]
    open = [entry['open'] for entry in data['Data']['Data']]
    close = [entry['close'] for entry in data['Data']['Data']]

    times = [entry['time'] for entry in data['Data']['Data']]
    dates = [datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S') for time in times]
    
    prices = {'open': open, 'high': high, 'low': low, 'close': close, 'dates': dates}
    return prices

def get_minute_price(symbol, period): #get a minute price for a specified number of minutes
    if(period <= 1):
        url = f'https://min-api.cryptocompare.com/data/v2/histominute?fsym={symbol}&tsym=USD&limit={1}'
    else:
        url = f'https://min-api.cryptocompare.com/data/v2/histominute?fsym={symbol}&tsym=USD&limit={period-1}'
    response = requests.get(url)
    data = response.json()
    prices = [entry['close'] for entry in data['Data']['Data']]

    times = [entry['time'] for entry in data['Data']['Data']]
    dates = [datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S') for time in times]
    return prices, dates

# Get hourly data for Ethereum


def generate_hourly_plot(ticker):
    hourly_data = get_hourly_data(ticker)
    
    # Converting the data into a DataFrame
    df = pd.DataFrame(hourly_data)
    df['dates'] = pd.to_datetime(df['dates'])
    df.set_index('dates', inplace=True)
    
    # Set the style for the plot
    mc = mpf.make_marketcolors(up='g', down='r', wick={'up': 'g', 'down': 'r'}, edge={'up': 'g', 'down': 'r'}, volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', y_on_right=False)
    
    # Plotting the candlestick chart
    mpf.plot(df, type='candle', style=s, title=f'Hourly {ticker} Prices', ylabel='Price (USD)',
             datetime_format='%H:%M', xrotation=45, savefig=dict(fname=f'{ticker}_hourly.jpg', dpi=300))


generate_hourly_plot("ETH")
