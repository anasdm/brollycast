import xarray as xr
import pandas as pd
import numpy as np
import torch
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import zipfile

def prepare_era5_data(window_size=3):
    df = pd.DataFrame()
    for i in range(1,13):
         with zipfile.ZipFile(f"falmouth_weather_{i}_2025.nc", "r") as z:
             z.extractall(f"extracted_{i}")
             instant = xr.open_dataset(
    "extracted/data_stream-oper_stepType-instant.nc"
)

             accum = xr.open_dataset(
    "extracted/data_stream-oper_stepType-accum.nc"
)
             ds = xr.merge([instant, accum])

             df_new = ds.mean(dim=['latitude', 'longitude']).to_dataframe().reset_index()
             df = pd.concat([df, df_new])
    df = df.rename(columns={
    "t2m": "temperature",
    "d2m": "dewpoint",
    "msl": "pressure",
    "tcc": "cloud_cover",
    "tp": "precipitation"
})
    # Create Binary Target: 1 if precipitation in the NEXT hour > 0.01mm
    df['target'] = (df['precipitation'].shift(-1) > 0.00001).astype(int)
    
    # Drop rows where target is NaN
    df = df.dropna(subset=['target'])
    
    # Features to use
    feature_cols = ['temperature', 'dewpoint', 
                    'pressure', 'cloud_cover']
    
    # Normalise features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    df_scaled = pd.DataFrame(scaled_features, columns=feature_cols, index=df.index)
    
    # Create rolling window sequences
    X, Y = [], []
    for i in range(len(df_scaled) - window_size):
        # Grab a window of past hours
        window = df_scaled.iloc[i : i + window_size].values.flatten()
        # Grab the target for the next hour (hour 3)
        target = df['target'].iloc[i + window_size]
        
        X.append(window)
        Y.append(target)
        
    return np.array(X), np.array(Y), scaler

class WeatherDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

import torch.nn as nn

class RainNowcaster(nn.Module):
    def __init__(self, input_dim):
        super(RainNowcaster, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)  
        )
        
    def forward(self, x):
        return self.network(x)


def train_model(model, dataloader, epochs=10, lr=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            
            # Backward pass & optimiSe
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * batch_X.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_preds += (predictions == batch_Y).sum().item()
            total_samples += batch_Y.size(0)
            
        epoch_loss = total_loss / total_samples
        epoch_acc = (correct_preds / total_samples) * 100
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad(): # Disable gradient calculations f
        for batch_X, batch_Y in dataloader:
            outputs = model(batch_X)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(predictions.numpy())
            all_targets.extend(batch_Y.numpy())
            
    # 3. Standard Metrics
    print("\n=== Model Performance on Unseen Test Data ===")
    print(classification_report(all_targets, all_preds, target_names=["No Rain", "Rain"]))
    
    print("=== Confusion Matrix ===")
    print(confusion_matrix(all_targets, all_preds))



if __name__ == "__main__":

    print("Starting training pipeline...")
    

    X, Y, scaler = prepare_era5_data(window_size=3)


    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]


    train_dataset = WeatherDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = WeatherDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    input_dimension = X_train.shape[1] 
    model = RainNowcaster(input_dim=input_dimension)

    train_model(model, train_loader, epochs=15)
    

    evaluate_model(model, test_loader)


    torch.save(model.state_dict(), "rain_nowcaster.pth")
    joblib.dump(scaler, "scaler.joblib")
    print("Model and Scaler saved successfully.")

