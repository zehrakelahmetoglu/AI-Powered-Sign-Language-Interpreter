import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

# 1. Model Mimarisi (Attention Pooling ile Güçlendirilmiş)
class SignLSTM(nn.Module):
    def __init__(self, input_dim=258, hidden_dim=512, num_layers=3, num_classes=226):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        
        # Attention Mekanizması
        self.attn_fc = nn.Linear(hidden_dim * 2, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, T, F = x.shape
        # Giriş Normalizasyonu
        x = self.bn_input(x.reshape(B * T, F)).reshape(B, T, F)
        
        lstm_out, _ = self.lstm(x)  # (B, T, hidden*2)
        
        # Attention Pooling: Her bir karenin önemini hesapla
        attn_scores = self.attn_fc(lstm_out) # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1) # (B, T, 1)
        
        # Ağırlıklı toplama (Context Vector)
        context = torch.sum(lstm_out * attn_weights, dim=1) # (B, hidden*2)
        
        return self.fc(context)

# 2. Veri Yükleyici (Dataset)
class SignDataset(Dataset):
    def __init__(self, csv_path, npy_dir, label_map):
        self.df = pd.read_csv(csv_path, header=None, names=["id", "label"])
        self.npy_dir = npy_dir
        self.label_map = label_map
        
        # Dosyaların varlığını kontrol et ve sadece mevcut olanları listele
        self.valid_samples = []
        print(f"Veri kontrol ediliyor: {csv_path}")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            npy_path = os.path.join(npy_dir, f"{row['id']}.npy")
            if os.path.exists(npy_path):
                self.valid_samples.append(row)
        
        print(f"Toplam geçerli örnek sayısı: {len(self.valid_samples)}")

    def __len__(self): return len(self.valid_samples)

    def __getitem__(self, idx):
        row = self.valid_samples[idx]
        data = np.load(os.path.join(self.npy_dir, f"{row['id']}.npy"))
        label = self.label_map[str(row['label'])]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 3. Eğitim Fonksiyonu
def train_v2(npy_dir, train_csv, val_csv, epochs=100, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim başlıyor... Cihaz: {device}")
    
    # 1. Sınıf Eşleşmesini Oluştur (Hata payını sıfıra indirir)
    all_df = pd.read_csv(train_csv, header=None)
    unique_labels = sorted(all_df[1].unique())
    label_map = {str(label): i for i, label in enumerate(unique_labels)}
    
    # Label Map'i kaydet (Demo için kritik)
    with open("label_map_v2.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)
    print(f"label_map_v2.json oluşturuldu. Sınıf sayısı: {len(unique_labels)}")
    
    # 2. Model, Dataset ve Loader
    num_classes = len(unique_labels)
    model = SignLSTM(num_classes=num_classes).to(device)
    
    train_ds = SignDataset(train_csv, npy_dir, label_map)
    val_ds = SignDataset(val_csv, npy_dir, label_map)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2)
    
    # 3. Kayıp ve Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc = 0
    for epoch in range(1, epochs + 1):
        # TRAIN
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Eğitim]")
        for x, y in train_pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # VALIDATION
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Doğrulama]"):
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
