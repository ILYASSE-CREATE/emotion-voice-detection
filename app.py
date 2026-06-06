import torch
import torch.nn as nn
import numpy as np
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import tempfile
import os

# ── Configuration ──────────────────────────────────────
EMOTIONS   = ['angry','calm','disgust','fearful','happy','neutral','sad','surprised']
MODEL_PATH = 'best_model_wav2vec2.pt'
MODEL_NAME = 'facebook/wav2vec2-base'
SR         = 16000
MAX_SAMPLES = SR * 4  # 4 secondes

# ── Architecture (identique à l'entraînement) ──────────
class Wav2Vec2ForSER(nn.Module):
    def __init__(self, model_name, num_classes=8, dropout=0.1):
        super().__init__()
        self.wav2vec2  = Wav2Vec2Model.from_pretrained(model_name)
        self.wav2vec2.feature_extractor._freeze_parameters()
        hidden_size    = self.wav2vec2.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        pooled  = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)


# ── Chargement du modèle ───────────────────────────────
print("Chargement du modèle...")
device    = torch.device('cpu')
model     = Wav2Vec2ForSER(MODEL_NAME, num_classes=8)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()
print(f"✅ Modèle chargé (val_acc={checkpoint['val_acc']:.2%})")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
print("✅ Feature extractor prêt")

# ── Application FastAPI ────────────────────────────────
app = FastAPI(title="Emotion Voice Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def predict_emotion(audio_path: str):
    """Charge un fichier audio et prédit l'émotion."""
    audio, _ = librosa.load(audio_path, sr=SR, mono=True)

    # Normalisation
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Padding / découpage
    if len(audio) < MAX_SAMPLES:
        pad   = MAX_SAMPLES - len(audio)
        audio = np.pad(audio, (pad // 2, pad - pad // 2))
    else:
        start = (len(audio) - MAX_SAMPLES) // 2
        audio = audio[start: start + MAX_SAMPLES]

    # Feature extraction
    inputs = feature_extractor(
        audio, sampling_rate=SR,
        return_tensors='pt', padding=False
    )

    # Prédiction
    with torch.no_grad():
        logits = model(inputs['input_values'])
        probs  = torch.softmax(logits, dim=-1)[0]
        pred   = probs.argmax().item()

    return {
        'emotion'    : EMOTIONS[pred],
        'confidence' : round(probs[pred].item() * 100, 1),
        'probabilities': {
            emo: round(probs[i].item() * 100, 1)
            for i, emo in enumerate(EMOTIONS)
        }
    }


@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Reçoit un fichier audio et retourne l'émotion prédite."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predict_emotion(tmp_path)
        return result
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
