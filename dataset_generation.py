import numpy as np
import pandas as pd
import random

def generate_converging_rising_wedge(S0, upper_initial, lower_initial, length):
    """
    Generate synthetic stock prices with a rising wedge pattern where the trendlines converge at the end.
    
    Parameters:
    S0 (float): Initial stock price.
    upper_initial (float): Initial multiplier for the upper trendline.
    lower_initial (float): Initial multiplier for the lower trendline.
    length (int): Total length of the time series.
    
    Returns:
    pd.Series: Time series data with a rising wedge pattern as a single row.
    """
    # Time indices
    t = np.arange(length)
    
    # Create upper and lower trendlines
    upper_trendline = S0 * (upper_initial + (1 - upper_initial) * (t / length))
    lower_trendline = S0 * (lower_initial + (1 - lower_initial) * (t / length))
    
    # Simulate price movement between the trendlines
    prices = lower_trendline + (upper_trendline - lower_trendline) * np.random.uniform(0, 1, length)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, S0 * 0.02, length)  # 2% noise
    prices += noise
    
    # Create Series with the pattern tag
    series = pd.Series(prices, index=[f'Time_{i}' for i in range(length)])
    series = series.iloc[:-3]

    series['Pattern'] = 'rising wedge pattern'
    return series


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
    series = pd.Series(prices, index=[f'Time_{i}' for i in range(length)])
    series = series.iloc[:-3]

    series['Pattern'] = 'declining wedge pattern'
    return series

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
    
    series = pd.Series(prices, index=[f'Time_{i}' for i in range(length)])
    series = series.iloc[:-3]


    series['Pattern'] = 'symmetrical triangle pattern'
    return series

# Parameters
S0 = 1  # Initial stock price
length = 18  # Length of the time series
# List to hold all the generated series
all_series = []

# Generate and store 100 iterations of the dataset
for i in range(10000):
    converging_rising_wedge = generate_converging_rising_wedge(S0, float(random.randrange(0, 35)) / 100, float(random.randrange(55, 100)) / 100, length)
    all_series.append(converging_rising_wedge)

    converging_declining_wedge = generate_converging_declining_wedge(S0, float(random.randrange(100, 120)) / 100, float(random.randrange(50, 75)) / 100, length)
    all_series.append(converging_declining_wedge)

    rate = float(random.randrange(10, 90))
    upper_initial = 1 + rate / 100  # Initial multiplier for the upper trendline
    lower_initial = 1 - rate / 100  # Initial multiplier for the lower trendline
    symmetrical_triangle = generate_symmetrical_triangle(S0, upper_initial, lower_initial, length)
    all_series.append(symmetrical_triangle)

# Concatenate all series into a single dataframe
combined_df = pd.DataFrame(all_series)

# Display the combined dataframe
print(combined_df.head())

# Save the combined dataframe to a CSV file
combined_df.to_csv('data.csv', index=False, sep=',', encoding='utf-8')

# Split the data into training, validation, and testing sets
def split_data(df, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    assert train_frac + val_frac + test_frac == 1, "Fractions must sum to 1"
    
    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Compute split indices
    train_end = int(train_frac * len(df))
    val_end = train_end + int(val_frac * len(df))
    
    # Split the dataframe
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

train_df, val_df, test_df = split_data(combined_df)

# Save the splits to CSV files
train_df.to_csv('training.csv', index=False, sep=',', encoding='utf-8')
val_df.to_csv('validation.csv', index=False, sep=',', encoding='utf-8')
test_df.to_csv('testing.csv', index=False, sep=',', encoding='utf-8')

# Display the shapes of the splits
print(f"Training set shape: {train_df.shape}")
print(f"Validation set shape: {val_df.shape}")
print(f"Testing set shape: {test_df.shape}")
