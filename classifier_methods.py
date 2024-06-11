import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings

# Define the neural network architecture (same as the saved model)
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to load the model
def load_model(model_path, input_dim, output_dim):
    model = SimpleNN(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to classify a single array of data


def classify_single_array(array):
    first = array[0]
    array = np.array([(float(x) / float(first)) for x in array], dtype=np.float64)

    training_data = pd.read_csv('synthetic-data/training.csv')
    label_encoder = LabelEncoder()
    label_encoder.fit(training_data['Pattern'])

    # Use a sample to get input_dim and output_dim
    input_dim = training_data.drop(columns=['Pattern']).shape[1]
    output_dim = len(np.unique(training_data['Pattern']))

    model_path = 'synthetic-data/stock_pattern_model.pth'
    model = load_model(model_path, input_dim, output_dim)
    single_array =  np.array(array)
    # Ensure the input is a 2D array (1 sample, 15 features)
    single_array = np.expand_dims(single_array, axis=0)
    
    # Convert the array to a PyTorch tensor
    X_new_tensor = torch.tensor(single_array, dtype=torch.float32)
    
    # Get the model prediction
    with torch.no_grad():
        outputs = model(X_new_tensor)
        _, predicted = torch.max(outputs, 1)
    
    # Decode the prediction to a class label
    predicted_label = label_encoder.inverse_transform(predicted.numpy())[0]
    return predicted_label

