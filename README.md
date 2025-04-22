# amdMODELs

**Answering Machine Detection AI models**

This repository contains trained deep learning models for Answering Machine Detection (AMD). These models help identify if a call is answered by a human or a machine (voicemail) and are optimized for fast CPU inference in telephony environments.

---

## 📦 Model Files

| Filename                                    | Type              | Description                                 |
|--------------------------------------------|-------------------|---------------------------------------------|
| `voicemail_cnn_20250411_184659.pth`        | PyTorch           | CNN-based model checkpoint                  |
| `voicemail_cnn_20250411_184659.onnx`       | ONNX              | Exported version of above                   |
| `voicemail_cnn_attn_20250411_191701.pth`   | PyTorch           | CNN with attention module                   |
| `voicemail_cnn_attn_20250411_191701.onnx`  | ONNX              | Attention-enabled ONNX model                |
| `voicemail_model.pth`                      | PyTorch           | Stable production-ready model               |
| `voicemail_model.onnx`                     | ONNX              | Production-ready ONNX model                 |
| `voicemail_model_20250411_184404.pth`      | PyTorch           | Older trained checkpoint                    |
| `cnn_voicemail_classifier.h5`              |                  |   Latest trained model 23-04-2025 40k data|

All models are based on v1.0 architecture trained using log-mel spectrograms with delta and delta-delta features extracted from real-world telecom data.

---

## 🧪 Usage

### Inference with ONNX Runtime (Python)
```python
python3 predict_wav_attn.py 20240514-INGROUP_504-all.wav
