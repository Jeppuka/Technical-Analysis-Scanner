import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler

def generate_converging_rising_wedge(length):
    # Time indices
    t = np.arange(length)
    S0 = 1

    # Create upper and lower trendlines
    upper_start = random.uniform(1.75, 2.25)
    upper_end = 2.5
    upper_trendline = np.linspace(upper_start, upper_end, length)

    lower_start = random.uniform(0, 0.5)
    lower_end = 2.2
    lower_trendline = np.linspace(lower_start, lower_end, length)

    # Simulate price movement between the trendlines
    prices = lower_trendline + (upper_trendline - lower_trendline) * np.random.uniform(0, 1, length)
    
    # Adjust high and low values to ensure they are distinct
    high = np.maximum(prices, upper_trendline - np.random.uniform(0.1, 0.5, length))
    low = np.minimum(prices, lower_trendline + np.random.uniform(0.1, 0.5, length))
    
    # Ensure high and low values are distinct
    high = np.where(high == low, high + random.uniform(0.1, 0.25), high)
    low = np.where(high == low, low - random.uniform(0.1, 0.25), low)
    
    open_ = []
    close = []
    for i in range(length):
        open_val = random.uniform(low[i], high[i])
        close_val = random.uniform(low[i], high[i])
        
        # Set close and low values to lower trendline for first value and every fifth value
        if i == 0 or (i + 1) % 5 == 0:
            close_val = low[i]
            low[i] = lower_trendline[i]

        if open_val == close_val:
            open_val = random.uniform(low[i], high[i])
            close_val = open_val + 0.2
        elif open_val > close_val:
            if (open_val - close_val) < 0.1:
                open_val = ((high[i] - low[i]) / 2) + low[i] + 0.2
                close_val = open_val - 0.2
        elif open_val < close_val:
            if (close_val - open_val) < 0.1:
                open_val = ((high[i] - low[i]) / 2) + low[i] - 0.2
                close_val = open_val + 0.2
        
        # Adjust open and close values if they are too close to the lower trendline
        if low[i] == lower_trendline[i]:
            if(close_val > open_val):
                open_val = low[i]
            else:
                close_val = low[i]

        open_.append(open_val)
        close.append(close_val)

    volume = np.random.uniform(10, 50, length)

    data = pd.DataFrame({
        'upper': upper_trendline,
        'lower': lower_trendline,
        'high': high,
        'low': low,
        'open': open_,
        'close': close,
        'volume': volume
    })
    return data


def create_graph(prices, volume_scale='linear'):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 8))

    # Define width of candlestick elements
    width = 0.6
    width2 = 0.07

    # Define up and down prices
    up = prices[prices.close >= prices.open]
    down = prices[prices.close < prices.open]

    # Define colors to use
    col1 = 'green'
    col2 = 'red'

    # Plot up prices on ax1
    ax1.bar(up.index, up.close - up.open, width, bottom=up.open, color=col1)
    ax1.bar(up.index, up.high - up.close, width2, bottom=up.close, color=col1)
    ax1.bar(up.index, up.low - up.open, width2, bottom=up.open, color=col1)

    # Plot down prices on ax1
    ax1.bar(down.index, down.close - down.open, width, bottom=down.open, color=col2)
    ax1.bar(down.index, down.high - down.open, width2, bottom=down.open, color=col2)
    ax1.bar(down.index, down.low - down.close, width2, bottom=down.close, color=col2)

    ax1.plot(prices.upper)
    ax1.plot(prices.lower)
    # Plot volume on ax2
    ax2.bar(prices.index, prices.volume, width, alpha=0.3, color='blue')
    ax2.set_ylabel('Volume')
    ax2.set_yscale(volume_scale)

    # Rotate x-axis tick labels
    ax1.set_xticks(prices.index)
    ax1.set_xticklabels(prices.index, rotation=45, ha='right')

    # Adjust the layout
    plt.tight_layout()

    # Display the charts
    plt.show()




def create_candlestick(upper, lower, prev_close):
    upper_range = upper
    lower_range = lower
    mid_price = random.uniform(upper_range - ((upper_range - lower_range) * 0.1), lower_range + ((upper_range - lower_range) * 0.1))
    
    open = prev_close
    direction = random.choice([-1, 1])
    candle_height = random.uniform(0.5, 1)
    
    if (upper_range - mid_price) == (mid_price - lower_range):  # if mid price is exactly in the middle
        mid_difference = (upper_range - mid_price)
    elif (upper_range - mid_price) < (mid_price - lower_range):  # if mid price is top half
        mid_difference = (upper_range - mid_price)
    else:  # if mid price is bottom half
        mid_difference = (mid_price - lower_range)

    close = mid_price + direction * (candle_height * mid_difference)
    
    if open > close:
        high = open + (random.uniform(0.1, 0.5)*(upper_range-open))
        low = close - (random.uniform(0.1, 0.5)*(close-lower_range))
    else:
        high = close + (random.uniform(0.1, 0.5)*(upper_range-close))
        low = open - (random.uniform(0.1, 0.5)*(open-lower_range))

    # Introduce randomness to allow high and low to exceed the trendlines
    high = high + random.uniform(0, 0.2)  # Add a small random value to potentially exceed the trendline
    low = low - random.uniform(0, 0.2)   # Subtract a small random value to potentially go below the trendline

    data = {
        'high': high,
        'low': low,
        'open': open,
        'close': close,
    }
    return data

def plot_candlestick(candlestick_data):
    fig, ax = plt.subplots(figsize=(5, 5))

    # Define width of candlestick elements
    width = 0.6
    width2 = 0.07

    # Define colors to use
    col = 'green' if candlestick_data['close'] >= candlestick_data['open'] else 'red'

    # Plot the candlestick
    ax.bar(0, candlestick_data['close'] - candlestick_data['open'], width, bottom=candlestick_data['open'], color=col)
    ax.bar(0, candlestick_data['high'] - candlestick_data['close'], width2, bottom=candlestick_data['close'], color=col)
    ax.bar(0, candlestick_data['low'] - candlestick_data['open'], width2, bottom=candlestick_data['open'], color=col)

    # Plot the upper and lower ranges
    ax.plot([0, 0], [candlestick_data['high'], candlestick_data['low']], color='black')

    # Set labels and title
    ax.set_ylabel('Price')
    ax.set_title('Candlestick')

    # Remove x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xticklabels([])

    # Display the plot
    plt.show()





def generate_converging_rising_wedge(length):
    # Time indices
    t = np.arange(length)
    S0 = 1

    # Create upper and lower trendlines
    upper_start = random.uniform(1.75, 2.25)
    upper_end = 2.5
    upper_trendline = np.linspace(upper_start, upper_end, length)

    lower_start = random.uniform(0, 0.5)
    lower_end = 2.2
    lower_trendline = np.linspace(lower_start, lower_end, length)

    # Simulate price movement between the trendlines
    prices = lower_trendline + (upper_trendline - lower_trendline) * np.random.uniform(0, 1, length)
    
    # Adjust high and low values to ensure they are distinct
    high = np.maximum(prices, upper_trendline - np.random.uniform(0.1, 0.5, length))
    low = np.minimum(prices, lower_trendline + np.random.uniform(0.1, 0.5, length))
    
    # Ensure high and low values are distinct
    high = np.where(high == low, high + random.uniform(0.1, 0.25), high)
    low = np.where(high == low, low - random.uniform(0.1, 0.25), low)
    
    candlestick_data_list = []
    prev_close = random.uniform(lower_start, upper_start)  # Initial close value

    for i in range(length):
        candlestick_data = create_candlestick(upper_trendline[i], lower_trendline[i], prev_close)
        candlestick_data['upper'] = upper_trendline[i]
        candlestick_data['lower'] = lower_trendline[i]
        prev_close = candlestick_data['close']
        candlestick_data_list.append(candlestick_data)

    return candlestick_data_list

def generate_converging_falling_wedge(length):
    # Time indices
    t = np.arange(length)
    S0 = 1

    # Create upper and lower trendlines
    upper_start = random.uniform(1.75, 2.25)
    upper_end = 0.8
    upper_trendline = np.linspace(upper_start, upper_end, length)

    lower_start = random.uniform(0.25, 0.5)
    lower_end = 0.5
    lower_trendline = np.linspace(lower_start, lower_end, length)

    # Simulate price movement between the trendlines
    prices = lower_trendline + (upper_trendline - lower_trendline) * np.random.uniform(0, 1, length)
    
    # Adjust high and low values to ensure they are distinct
    high = np.maximum(prices, upper_trendline - np.random.uniform(0.1, 0.5, length))
    low = np.minimum(prices, lower_trendline + np.random.uniform(0.1, 0.5, length))
    
    # Ensure high and low values are distinct
    high = np.where(high == low, high + random.uniform(0.1, 0.25), high)
    low = np.where(high == low, low - random.uniform(0.1, 0.25), low)
    
    candlestick_data_list = []
    prev_close = random.uniform(lower_start, upper_start)  # Initial close value

    for i in range(length):
        candlestick_data = create_candlestick(upper_trendline[i], lower_trendline[i], prev_close)
        candlestick_data['upper'] = upper_trendline[i]
        candlestick_data['lower'] = lower_trendline[i]
        prev_close = candlestick_data['close']
        candlestick_data_list.append(candlestick_data)

    return candlestick_data_list

def generate_symmetrical_wedge(length):
    # Time indices
    t = np.arange(length)
    S0 = 1

    start_val = random.uniform(1, 1.75)
    # Create upper and lower trendlines
    upper_start = 2 + start_val
    upper_end = 2.15
    upper_trendline = np.linspace(upper_start, upper_end, length)

    lower_start = 2 - start_val
    lower_end = 1.85
    lower_trendline = np.linspace(lower_start, lower_end, length)

    # Simulate price movement between the trendlines
    prices = lower_trendline + (upper_trendline - lower_trendline) * np.random.uniform(0, 1, length)
    
    # Adjust high and low values to ensure they are distinct
    high = np.maximum(prices, upper_trendline - np.random.uniform(0.1, 0.5, length))
    low = np.minimum(prices, lower_trendline + np.random.uniform(0.1, 0.5, length))
    
    # Ensure high and low values are distinct
    high = np.where(high == low, high + random.uniform(0.1, 0.25), high)
    low = np.where(high == low, low - random.uniform(0.1, 0.25), low)
    
    candlestick_data_list = []
    prev_close = random.uniform(lower_start, upper_start)  # Initial close value

    for i in range(length):
        candlestick_data = create_candlestick(upper_trendline[i], lower_trendline[i], prev_close)
        candlestick_data['upper'] = upper_trendline[i]
        candlestick_data['lower'] = lower_trendline[i]
        prev_close = candlestick_data['close']
        candlestick_data_list.append(candlestick_data)

    return candlestick_data_list

def generate_random_data(length):
    # Time indices
    t = np.arange(length)
    S0 = 1

    # Create upper and lower trendlines
    upper_start = random.uniform(1.25, 1.5)
    upper_end = upper_start
    upper_trendline = np.linspace(upper_start, upper_end, length)

    lower_start = random.uniform(0.75, 1)
    lower_end = lower_start
    lower_trendline = np.linspace(lower_start, lower_end, length)

    # Simulate price movement between the trendlines
    prices = lower_trendline + (upper_trendline - lower_trendline) * np.random.uniform(0, 1, length)
    
    # Adjust high and low values to ensure they are distinct
    high = np.maximum(prices, upper_trendline - np.random.uniform(0.1, 0.5, length))
    low = np.minimum(prices, lower_trendline + np.random.uniform(0.1, 0.5, length))
    
    # Ensure high and low values are distinct
    high = np.where(high == low, high + random.uniform(0.1, 0.25), high)
    low = np.where(high == low, low - random.uniform(0.1, 0.25), low)
    
    candlestick_data_list = []
    prev_close = random.uniform(lower_start, upper_start)  # Initial close value

    for i in range(length):
        candlestick_data = create_candlestick(upper_trendline[i], lower_trendline[i], prev_close)
        candlestick_data['upper'] = upper_trendline[i]
        candlestick_data['lower'] = lower_trendline[i]
        prev_close = candlestick_data['close']
        candlestick_data_list.append(candlestick_data)

    return candlestick_data_list

def convert_to_dataframe(candlestick_data_list):
    candlestick_data_dict = {
        'open': [candlestick['open'] for candlestick in candlestick_data_list],
        'high': [candlestick['high'] for candlestick in candlestick_data_list],
        'low': [candlestick['low'] for candlestick in candlestick_data_list],
        'close': [candlestick['close'] for candlestick in candlestick_data_list],
        'upper': [candlestick['upper'] for candlestick in candlestick_data_list],
        'lower': [candlestick['lower'] for candlestick in candlestick_data_list],
        'volume': [random.uniform(10, 50) for _ in range(len(candlestick_data_list))]
    }
    return pd.DataFrame(candlestick_data_dict)





price_columns = ['open', 'high', 'low', 'close', 'upper', 'lower']
volume_column = 'volume'


def normalize(df):
    price_columns = ['open', 'high', 'low', 'close', 'upper', 'lower']
    volume_column = 'volume'
    normalized_df = df.copy()
    first_value = df["open"].iloc[0]
    for column in price_columns:
        normalized_df[column] = df[column] / first_value
    
    # Keep the volume column as is
    normalized_df[volume_column] = df[volume_column]
    
    return normalized_df



print(normalize(convert_to_dataframe(generate_converging_rising_wedge(20))))
print(normalize(convert_to_dataframe(generate_converging_falling_wedge(20))))
print(normalize(convert_to_dataframe(generate_symmetrical_wedge(20))))
print(normalize(convert_to_dataframe(generate_random_data(20))))










