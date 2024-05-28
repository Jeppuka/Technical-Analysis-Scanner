import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def generate_symmetrical_triangle(S0, upper_initial, lower_initial, length):
    """
    Generate synthetic stock prices with a symmetrical triangle pattern where the trendlines converge at the end.
    
    Parameters:
    S0 (float): Initial stock price.
    upper_initial (float): Initial multiplier for the upper trendline.
    lower_initial (float): Initial multiplier for the lower trendline.
    length (int): Total length of the time series.
    
    Returns:
    pd.DataFrame: Time series data with a symmetrical triangle pattern as a single row.
    """
    # Time indices
    t = np.arange(length)
    
    # Create upper and lower trendlines that symmetrically converge over time
    midpoint = (upper_initial + lower_initial) / 2
    upper_trendline = S0 * (midpoint + (upper_initial - midpoint) * (1 - t / length))
    lower_trendline = S0 * (midpoint - (midpoint - lower_initial) * (1 - t / length))
    
    # Simulate price movement between the trendlines
    a, b = 0.5, 0.5  # Parameters for the beta distribution to create a U-shaped distribution
    prices = lower_trendline + (upper_trendline - lower_trendline) * np.random.beta(a, b, length)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, S0 * 0.03, length)  # 3% noise
    prices += noise
    
    # Create DataFrame with the pattern tag
    df = pd.DataFrame({'Time': t, 'Stock Price': prices})
    df = df.iloc[:-3]
    return df

# Parameters
S0 = 1  # Initial stock price
length = 18  # Length of the time series

# Create a figure with 8x8 subplots
fig, axes = plt.subplots(8, 8, figsize=(20, 20))
fig.suptitle('Synthetic Stock Prices with Symmetrical Triangle Pattern (Converging Trendlines)', fontsize=20)

# Generate and plot 64 iterations
for i in range(8):
    for j in range(8):
        rate = float(random.randrange(10, 90))
        upper_initial = 1 + rate / 100  # Initial multiplier for the upper trendline
        lower_initial = 1 - rate / 100  # Initial multiplier for the lower trendline
        df = generate_symmetrical_triangle(S0, upper_initial, lower_initial, length)
        ax = axes[i, j]
        ax.plot(df['Time'], df['Stock Price'], label='Stock Price')
        ax.plot(df['Time'], S0 * (1 + (rate / 100) * (1 - df['Time'] / length)), 'r--', label='Upper Trendline')
        ax.plot(df['Time'], S0 * (1 - (rate / 100) * (1 - df['Time'] / length)), 'g--', label='Lower Trendline')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
