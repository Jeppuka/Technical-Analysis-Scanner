import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def generate_converging_declining_wedge(S0, upper_initial, lower_initial, length):
    """
    Generate synthetic stock prices with a declining wedge pattern where the trendlines converge at the end.
    
    Parameters:
    S0 (float): Initial stock price.
    upper_initial (float): Initial multiplier for the upper trendline.
    lower_initial (float): Initial multiplier for the lower trendline.
    length (int): Total length of the time series.
    
    Returns:
    pd.DataFrame: Time series data with a declining wedge pattern as a single row.
    """
    # Time indices
    t = np.arange(length)
    
    # Create upper and lower trendlines that decline over time
    upper_trendline = S0 * (upper_initial - (upper_initial - 0.5) * (t / length))
    lower_trendline = S0 * (lower_initial - (lower_initial - 0.5) * (t / length))
    
    # Simulate price movement between the trendlines
    prices = lower_trendline + (upper_trendline - lower_trendline) * np.random.uniform(0, 1, length)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, S0 * 0.02, length)  # 2% noise
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
fig.suptitle('Synthetic Stock Prices with Declining Wedge Pattern (Converging Trendlines)', fontsize=20)

# Generate and plot 64 iterations
for i in range(8):
    for j in range(8):
        upper_initial = float(random.randrange(90, 120)) / 100  # Initial multiplier for the upper trendline
        lower_initial = float(random.randrange(50, 70)) / 100  # Initial multiplier for the lower trendline
        df = generate_converging_declining_wedge(S0, upper_initial, lower_initial, length)
        ax = axes[i, j]
        ax.plot(df['Time'], df['Stock Price'], label='Stock Price')
        ax.plot(df['Time'], S0 * (upper_initial - (upper_initial - 0.5) * (df['Time'] / length)), 'r--', label='Upper Trendline')
        ax.plot(df['Time'], S0 * (lower_initial - (lower_initial - 0.5) * (df['Time'] / length)), 'g--', label='Lower Trendline')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
