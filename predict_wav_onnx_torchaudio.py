import sys
import torch
import torchaudio
import numpy as np
import onnxruntime as ort

# === Usage check ===
if len(sys.argv) != 2:
    print("Usage: python3 predict_wav_onnx_torchaudio.py <path_to_wav_file>")
    sys.exit(1)

wav_path = sys.argv[1]

# === Load audio using torchaudio (fast!) ===
try:
    waveform, sr = torchaudio.load(wav_path)  # shape: (1, samples)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000
except Exception as e:
    print(f"Error loading audio: {e}")
    sys.exit(1)

# === Extract MFCC ===
try:
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=40)
    mfcc = mfcc_transform(waveform)  # shape: (1, 40, time)
    features = mfcc.mean(dim=2).squeeze().numpy()  # shape: (40,)
except Exception as e:
    print(f"Error extracting MFCC: {e}")
    sys.exit(1)

# === Load ONNX model ===
try:
    session = ort.InferenceSession("voicemail_model.onnx")
    input_name = session.get_inputs()[0].name
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    sys.exit(1)

# === Prepare input and predict ===
features = features.astype(np.float32).reshape(1, -1)

output = session.run(None, {input_name: features})
pred_class = int(np.argmax(output[0], axis=1)[0])
label = "voicemail" if pred_class == 1 else "human"

print(f"Prediction for '{wav_path}': {label}")

