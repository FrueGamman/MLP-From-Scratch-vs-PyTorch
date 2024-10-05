import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
import torch 
import torch.nn as nn
import torch.optim as optim


# fetch dataset
ecoli = fetch_ucirepo(id=39)
  
# data (as pandas dataframes)
X = ecoli.data.features
y = ecoli.data.targets
  
# metadata
print(ecoli.metadata)
  
# variable information
print(ecoli.variables)

#Filter the dataset to include only the classes cp and im, and remove the rest of the data.
# convert features and targets to a DataFrame
df = pd.concat([X, y], axis=1)

df_filtered = df[df['class'].isin(['cp' , 'im'])]
df_filtered['class'] = df_filtered['class'].map({'cp': 0, 'im': 1})
#print(df_filtered.head)

# Separate features (X) and target (y)
X_filtered = df_filtered.drop('class', axis=1)  # Drop the target column from features
y_filtered = df_filtered['class']  # Target column

print(X_filtered.head())  # Check the features
print(y_filtered.head())  # Check the target

# Normalize the features to prepare the data for the MLP models.
# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# Convert X_scaled back to a DataFrame (optional, for clarity)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_filtered.columns)

# Print the normalized data
print(X_scaled_df.head())

#Implement MLP from Scratch
#
# Define the MLP architecture using only Python and standard libraries.
# MLP Implementation
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -709, 709)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
    
    def backward(self, X, y):
        m = X.shape[0]
        y = y.values.reshape(-1, 1)  # Convert Series to NumPy array and reshape
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        return dW1, db1, dW2, db2
    
    def update_weights(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        
# Initialize MLP with Ecoli data
input_size = X_scaled.shape[1]
hidden_size = 10
output_size = 1
mlp = MLP(input_size, hidden_size, output_size)

# Perform forward pass
outputs = mlp.forward(X_scaled)
print("Initial output (before training):")
print(outputs[:5])

# Perform backpropagation
dW1, db1, dW2, db2 = mlp.backward(X_scaled, y_filtered)
print("\n")
print("Gradient for W1:")


# implement train and test MLP using PyTorch

# Convert the features and target to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)  # Features
y_tensor = torch.tensor(y_filtered.values, dtype=torch.float32).view(-1, 1)  # Targets (reshape to match output format)
class MLP_PyTorch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_PyTorch, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input to hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden to output layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification
    
    def forward(self, x):
        out = self.fc1(x)  # Linear transformation (input -> hidden)
        out = self.relu(out)  # Apply ReLU
        out = self.fc2(out)  # Linear transformation (hidden -> output)
        out = self.sigmoid(out)  # Apply Sigmoid for binary output
        return out

# Model initialization
input_size = X_tensor.shape[1]  # Number of features
hidden_size = 10  # Number of neurons in hidden layer
output_size = 1  # Binary classification (output = 1)

model = MLP_PyTorch(input_size, hidden_size, output_size)

# Define loss function (Binary Cross-Entropy)
criterion = nn.BCELoss()

# Define optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Learning rate = 0.01

# Training parameters
epochs = 1000  # Number of iterations
for epoch in range(epochs):
    # Forward pass: compute predicted outputs by passing inputs to the model
    outputs = model(X_tensor)
    
    # Compute the loss
    loss = criterion(outputs, y_tensor)
    
    # Backward pass: compute gradient of the loss with respect to model parameters
    optimizer.zero_grad()  # Zero out the gradients to avoid accumulation
    loss.backward()  # Perform backpropagation
    optimizer.step()  # Update weights
    
    # Print the loss for every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
# Evaluate the model
with torch.no_grad():
    predictions = model(X_tensor)
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes == y_tensor).float().mean()
    print(f'Accuracy: {accuracy:.4f}')