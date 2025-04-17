import sys
import librosa
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
import time

# === CONFIG ===
FIXED_TIME = 100
N_MFCC = 40
SAMPLE_RATE = 16000
LABELS = ["human", "voicemail"]

PTH_MODEL = "voicemail_cnn_attn_20250411_191701.pth"
ONNX_MODEL = "voicemail_cnn_attn_20250411_191701.onnx"

# === MODEL DEFINITIONS ===
class AttentionPool1D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)
        return pooled

class CNNVoiceWithAttention(nn.Module):
    def __init__(self, in_channels=40, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.attn_pool = AttentionPool1D(128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.attn_pool(x)
        return self.fc(x)

# === FEATURE EXTRACTOR ===
def extract_features(wav_path):
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    if mfcc.shape[1] > FIXED_TIME:
        mfcc = mfcc[:, :FIXED_TIME]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, FIXED_TIME - mfcc.shape[1])), mode='constant')
    return mfcc.astype(np.float32)

# === PYTORCH INFERENCE ===
def infer_pytorch(mfcc):
    model = CNNVoiceWithAttention()
    model.load_state_dict(torch.load(PTH_MODEL, map_location="cpu"))
    model.eval()
    x = torch.tensor(mfcc).unsqueeze(0)  # (1, 40, 100)
    start = time.time()
    with torch.no_grad():
        out = model(x)
    end = time.time()
    pred = torch.argmax(out, dim=1).item()
    print(f"[PyTorch] Prediction: {LABELS[pred]} | Inference time: {(end - start)*1000:.2f} ms")

# === ONNX INFERENCE ===
def infer_onnx(mfcc):
    session = ort.InferenceSession(ONNX_MODEL)
    input_name = session.get_inputs()[0].name
    start = time.time()
    result = session.run(None, {input_name: mfcc[np.newaxis, :]})
    end = time.time()
    pred = np.argmax(result[0], axis=1)[0]
    print(f"[ONNX  ] Prediction: {LABELS[pred]} | Inference time: {(end - start)*1000:.2f} ms")

# === MAIN ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 predict_wav_attn.py <file.wav>")
        exit(1)

    wav_path = sys.argv[1]
    mfcc = extract_features(wav_path)

    infer_pytorch(mfcc)
    infer_onnx(mfcc)

