import numpy as np
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

class CatBoostModel:
    def __init__(self, horizon=12, random_state=42):
        self.horizon = horizon
        self.model = CatBoostRegressor(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            random_seed=random_state,
            verbose=False
        )
    
    def fit(self, X, y):
        if len(X) > 0:
            self.model.fit(X, y)
        return self
    
    def predict(self, X):
        if len(X) > 0:
            return self.model.predict(X)
        return np.array([])

class PatchTSTModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_heads=8, num_layers=3, patch_len=12, horizon=12):
        super(PatchTSTModel, self).__init__()
        self.horizon = horizon
        self.patch_len = patch_len
        
        self.input_projection = nn.Linear(patch_len, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, horizon)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        num_patches = max(1, seq_len // self.patch_len)
        actual_len = num_patches * self.patch_len
        patches = x[:, :actual_len].reshape(batch_size, num_patches, self.patch_len)
        patches = self.input_projection(patches)
        transformed = self.transformer(patches)
        global_features = transformed.mean(dim=1)
        output = self.output_layer(global_features)
        return output

class PatchTSTWrapper:
    def __init__(self, input_length=24, horizon=12, learning_rate=0.001, epochs=50):
        self.input_length = input_length
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_sequences(self, data):
        X, y = [], []
        min_len = self.input_length + self.horizon
        if len(data) < min_len:
            return np.array([]), np.array([])
        
        for i in range(len(data) - min_len + 1):
            X.append(data[i:i + self.input_length])
            y.append(data[i + self.input_length:i + self.input_length + self.horizon])
        
        if len(X) == 0:
            return np.array([]), np.array([])
        
        return np.array(X), np.array(y)
    
    def fit(self, X, y=None):
        X_seq, y_seq = self.create_sequences(X)
        
        if len(X_seq) == 0:
            return self
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        self.model = PatchTSTModel(
            input_dim=1,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            patch_len=12,
            horizon=self.horizon
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
        
        return self
    
    def predict(self, X):
        if self.model is None:
            return np.full(self.horizon, X[-1] if len(X) > 0 else 0)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if len(X_tensor.shape) == 1:
                X_tensor = X_tensor.unsqueeze(0)
            if X_tensor.shape[1] < self.input_length:
                pad = torch.zeros(X_tensor.shape[0], self.input_length - X_tensor.shape[1]).to(self.device)
                X_tensor = torch.cat([pad, X_tensor], dim=1)
            elif X_tensor.shape[1] > self.input_length:
                X_tensor = X_tensor[:, -self.input_length:]
            
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy().flatten()