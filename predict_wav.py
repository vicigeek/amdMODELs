import sys
import librosa
import numpy as np
import torch
import torch.nn as nn

# === Model class must match training ===
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

# === Check input ===
if len(sys.argv) != 2:
    print("Usage: python3 predict_wav.py <path_to_wav_file>")
    sys.exit(1)

wav_path = sys.argv[1]

# === Load audio and extract MFCC features ===
try:
    y, sr = librosa.load(wav_path, sr=16000)  # resample to 16kHz
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfcc, axis=1)  # (40,)
except Exception as e:
    print(f"Error processing WAV file: {e}")
    sys.exit(1)

# === Load model ===
model = VoiceClassifier()
model.load_state_dict(torch.load("voicemail_cnn_20250411_184659.pth", map_location=torch.device("cpu")))
model.eval()

# === Predict ===
features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    output = model(features_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

label = "voicemail" if predicted_class == 1 else "human"
print(f"Prediction for '{wav_path}': {label}")

