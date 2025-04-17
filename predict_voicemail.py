import sys
import numpy as np
import torch
import torch.nn as nn

# === Model must match the one used in training ===
class VoiceClassifier(nn.Module):
    def __init__(self, input_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

# === Check input argument ===
if len(sys.argv) != 2:
    print("Usage: python3 predict_voicemail.py <path_to_npy_file>")
    sys.exit(1)

npy_path = sys.argv[1]

# === Load the model ===
model = VoiceClassifier()
model.load_state_dict(torch.load("voicemail_model.pth", map_location=torch.device("cpu")))
model.eval()

# === Load and preprocess the features ===
try:
    features = np.load(npy_path)
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

if len(features.shape) == 2:
    features = np.mean(features, axis=1)  # Reduce (40, time) to (40,)

if features.shape[0] != 40:
    print(f"Invalid input shape: expected 40 features, got {features.shape[0]}")
    sys.exit(1)

features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 40)

# === Predict ===
with torch.no_grad():
    output = model(features_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

label = "voicemail" if predicted_class == 1 else "human"
print(f"Prediction for '{npy_path}': {label}")
