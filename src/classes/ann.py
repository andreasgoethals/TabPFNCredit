# src/classes/ann.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.utils import set_random_seed
from sklearn.base import BaseEstimator, ClassifierMixin

class NN(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_layer_size, activation, dropout_rate):
        super(NN, self).__init__()
        layers = []
        layer_sizes = [input_dim] + [hidden_layer_size] * num_hidden_layers
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(layer_sizes[-1], 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class NNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_hidden_layers=1, hidden_layer_size=100, activation='relu', dropout_rate=0.1, epochs=10, batch_size=32, learning_rate=0.001, weight_decay=0.01, early_stopping_delay=5, random_seed=0):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_delay = early_stopping_delay
        self.random_seed = random_seed
        self.model = None
        set_random_seed(self.random_seed)

    def fit(self, X, y, X_val=None, y_val=None):
        set_random_seed(self.random_seed)
        input_dim = X.shape[1]
        self.model = NN(input_dim, self.num_hidden_layers, self.hidden_layer_size, self.activation, self.dropout_rate)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = np.inf
        epochs_no_improve = 0

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                self.model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.early_stopping_delay:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor).numpy()
        return np.hstack((1 - outputs, outputs))

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba[:, 1] > threshold).astype(int)


"""
# src/classes/ann.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin




class NN(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_layer_size, activation, dropout_rate):
        super(NN, self).__init__()
        layers = []
        layer_sizes = [input_dim] + [hidden_layer_size] * num_hidden_layers
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(layer_sizes[-1], 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class NNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_hidden_layers=1, hidden_layer_size=100, activation='relu', dropout_rate=0.1, epochs=10, batch_size=32, learning_rate=0.001, weight_decay=0.01, early_stopping_delay=5):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_delay = early_stopping_delay
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        input_dim = X.shape[1]
        self.model = NN(input_dim, self.num_hidden_layers, self.hidden_layer_size, self.activation, self.dropout_rate)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = np.inf
        epochs_no_improve = 0

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                self.model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.early_stopping_delay:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor).numpy()
        return np.hstack((1 - outputs, outputs))

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba > threshold).astype(int)
"""