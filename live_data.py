import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import requests
import matplotlib.pyplot as plt
# Define the same model architecture
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
def initialize_model():
    X_sample, _ = load_data('training.csv')
    input_dim = X_sample.shape[1]
    output_dim = len(np.unique(pd.read_csv('training.csv')['Pattern']))
    model = SimpleNN(input_dim, output_dim)
    model.load_state_dict(torch.load('stock_pattern_model.pth'))
    model.eval()  # Set the model to evaluation mode
    return model

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['Pattern']).values
    y = data['Pattern'].values
    return X, y

def preprocess_new_data(prices):
    # Ensure the input is a numpy array
    X_new = np.array(prices).reshape(1, -1)  # Reshape for single sample with 15 time steps
    # Convert the new data to a PyTorch tensor
    X_new_tensor = torch.tensor(X_new, dtype=torch.float32)
    return X_new_tensor

def predict(model, X_new_tensor):
    with torch.no_grad():
        outputs = model(X_new_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted

def decode_predictions(predicted, label_encoder):
    predicted_labels = label_encoder.inverse_transform(predicted.numpy())
    return predicted_labels

# Load and prepare label encoder
label_encoder = LabelEncoder()
label_encoder.fit(pd.read_csv('training.csv')['Pattern'])  # Fit the label encoder with training data labels

# Initialize the model
model = initialize_model()

# Example array of prices


#Crypto Compare API
url = "https://min-api.cryptocompare.com/data/v2/histominute?fsym=ETH&tsym=GBP&limit=14"

def get_close_prices(url):
    response = requests.get(url)
    data = response.json()
    
    # Extract the "close" prices into an array
    prices = [entry['close'] for entry in data['Data']['Data']]
    return prices


# Get the close prices
prices = get_close_prices(url)


def normalize(prices):
    start = prices[0]
    for x in range(len(prices)):
        prices[x] = prices[x]/start
    return prices
# Normalize the prices

prices = normalize(prices)
# Preprocess the array of prices
X_new_tensor = preprocess_new_data(prices)

# Get the model prediction
predicted = predict(model, X_new_tensor)

# Decode the prediction to class labels
predicted_labels = decode_predictions(predicted, label_encoder)

# Print the predictions
print("Predicted label for the given array of prices:")
print(predicted_labels)

def plot_prices(prices):
    plt.figure(figsize=(10, 5))
    plt.plot(prices, marker='o', linestyle='-', color='b')
    plt.title("Normalized Prices Over Time")
    plt.xlabel("Time Steps (minutes)")
    plt.ylabel("Normalized Price")
    plt.grid(True)
    plt.show()

plot_prices(prices)