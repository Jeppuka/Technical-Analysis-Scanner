import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class StockPatternDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['Pattern']).values
    y = data['Pattern'].values
    return X, y

def prepare_dataloader(file_path, batch_size=32, shuffle=True):
    X, y = load_data(file_path)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    dataset = StockPatternDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Prepare dataloaders
train_loader = prepare_dataloader('training.csv', batch_size=32, shuffle=True)
val_loader = prepare_dataloader('validation.csv', batch_size=32, shuffle=False)
test_loader = prepare_dataloader('testing.csv', batch_size=32, shuffle=False)

# Define the neural network
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

# Initialize the model, loss function, and optimizer
# Use a sample to get input_dim and output_dim
X_sample, _ = load_data('training.csv')
input_dim = X_sample.shape[1]
output_dim = len(np.unique(pd.read_csv('training.csv')['Pattern']))

model = SimpleNN(input_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%")

# Evaluate the model on test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total}%")





new_data = pd.read_csv('new_data.csv')

# Drop any non-numeric columns (like 'Pattern' if it exists in this new data)
X_new = new_data.drop(columns=['Pattern'], errors='ignore').values

# Convert the new data to a PyTorch tensor
X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

# Get the model prediction
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = model(X_new_tensor)
    _, predicted = torch.max(outputs, 1)

# Decode the prediction to class labels
label_encoder = LabelEncoder()
label_encoder.fit(pd.read_csv('training.csv')['Pattern'])  # Fit the label encoder with training data labels
predicted_labels = label_encoder.inverse_transform(predicted.numpy())

# Print the predictions
print("Predicted labels for the new data:")
print(predicted_labels)

torch.save(model.state_dict(), 'stock_pattern_model.pth')
