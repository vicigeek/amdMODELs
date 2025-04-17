import sys
import librosa
import numpy as np
import onnxruntime as ort

# === Usage check ===
if len(sys.argv) != 2:
    print("Usage: python3 predict_wav_onnx.py <path_to_wav_file>")
    sys.exit(1)

wav_file = sys.argv[1]

# === Extract MFCC from WAV ===
try:
    y, sr = librosa.load(wav_file, sr=16000)  # Resample to 16kHz
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfcc, axis=1)  # Shape: (40,)
except Exception as e:
    print(f"Error processing WAV file: {e}")
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

print(f"Prediction for '{wav_file}': {label}")

