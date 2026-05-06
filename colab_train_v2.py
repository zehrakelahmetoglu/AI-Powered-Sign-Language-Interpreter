import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

class SignLSTM(nn.Module):
    def __init__(self, input_dim=258, hidden_dim=512, num_layers=3, num_classes=226):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, T, F = x.shape
        x = self.bn_input(x.reshape(B * T, F)).reshape(B, T, F)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

class SignDataset(Dataset):
    def __init__(self, csv_path, npy_dir, label_map):
        self.df = pd.read_csv(csv_path, header=None, names=["id", "label"])
        self.npy_dir = npy_dir
        self.label_map = label_map
        self.valid_samples = [row for _, row in self.df.iterrows() 
                              if os.path.exists(os.path.join(npy_dir, f"{row['id']}.npy"))]

    def __len__(self): return len(self.valid_samples)

    def __getitem__(self, idx):
        row = self.valid_samples[idx]
        data = np.load(os.path.join(self.npy_dir, f"{row['id']}.npy"))
        label = self.label_map[str(row['label'])]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def train_v2(npy_dir, train_csv, val_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_df = pd.read_csv(train_csv, header=None)
    unique_labels = sorted(all_df[1].unique())
    label_map = {str(label): i for i, label in enumerate(unique_labels)}
    with open("label_map_v2.json", "w") as f:
        json.dump(label_map, f)
    
    model = SignLSTM(num_classes=len(unique_labels)).to(device)
    train_ds = SignDataset(train_csv, npy_dir, label_map)
    val_ds = SignDataset(val_csv, npy_dir, label_map)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0
    for epoch in range(1, 101):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); out = model(x); loss = criterion(out, y)
            loss.backward(); optimizer.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x); preds = out.argmax(dim=1)
                correct += (preds == y).sum().item(); total += y.size(0)
        acc = correct / total
        print(f"Epoch {epoch} Success: %{acc*100:.2f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "tsl_lstm_v2_best.pt")

if __name__ == "__main__":
    train_v2(npy_dir="keypoints_v2", train_csv="train_labels_full.csv", val_csv="val_labels_full.csv")
