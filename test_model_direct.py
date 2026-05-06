import torch
import json
from model import SignLSTM
import numpy as np

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'tsl_lstm_best.pt'
label_map_path = 'models/label_map.json'

# Load Label Map
with open(label_map_path, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
id_to_word = {v: k for k, v in label_map.items()}

# Load Model
model = SignLSTM(input_dim=258, num_classes=226).to(device)
checkpoint = torch.load(model_path, map_location=device)
if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
    model.load_state_dict(checkpoint['model_state'])
else:
    model.load_state_dict(checkpoint)
model.eval()

print(f"Model loaded on {device}")

# Create dummy input (1 batch, 30 frames, 258 features)
# Normal data is expected to be keypoints
input_data = torch.randn(1, 30, 258).to(device)

with torch.no_grad():
    logits = model(input_data)
    prediction_id = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item() * 100
    
    word = id_to_word.get(prediction_id, f"Unknown_{prediction_id}")
    
    print("-" * 30)
    print(f"Prediction: {word}")
    print(f"Confidence: %{confidence:.2f}")
    print("-" * 30)
