# 🎙️ Speech Emotion Recognition (SER)
> Détection automatique des émotions dans la voix par Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Wav2Vec2-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-70.83%25-green)

---

## 📌 Présentation

Ce projet implémente un système de détection d'émotions à partir de la voix en utilisant le dataset **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song).

Plusieurs architectures ont été explorées et comparées, de CNN simple jusqu'au fine-tuning de **Wav2Vec2** (state-of-the-art).

---

## 🎯 Résultats

| Modèle | Accuracy | UAR |
|---|---|---|
| CNN de base | 50.83% | 49.22% |
| CNN + SpecAugment | 52.50% | 52.34% |
| CNN + LSTM | 50.83% | 51.17% |
| **Wav2Vec2 (fine-tuné)** | **70.83%** | **71.48%** |

### Détail par émotion — Wav2Vec2

| Émotion | Recall |
|---|---|
| disgust | 90.62% |
| surprised | 90.62% |
| neutral | 81.25% |
| angry | 75.00% |
| calm | 75.00% |
| fearful | 71.88% |
| happy | 46.88% |
| sad | 40.62% |

---

## 📂 Structure du projet

```
speech-emotion-recognition/
│
├── notebooks/
│   ├── 01_visualisation.ipynb       # Exploration & visualisation RAVDESS
│   ├── 02_preprocessing.ipynb       # Extraction features, augmentation, split
│   ├── 03_cnn.ipynb                 # CNN de base
│   ├── 04_cnn_specaugment.ipynb     # CNN + SpecAugment
│   ├── 05_cnn_lstm.ipynb            # CNN + BiLSTM + Attention
│   └── 06_wav2vec2.ipynb            # Wav2Vec2 fine-tuning ← meilleur modèle
│
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song

- 24 acteurs (12 hommes, 12 femmes)
- 1440 fichiers audio `.wav`
- 8 émotions : `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`
- Sample rate : 16 000 Hz

**Split utilisé (par acteur) :**
- Train : acteurs 1–16 (+ augmentation ×4)
- Val : acteurs 17–20 (originaux seulement)
- Test : acteurs 21–24 (originaux seulement)

---

## 🧠 Architecture finale — Wav2Vec2

```
Audio brut (signal 1D, 64000 samples)
        ↓
Wav2Vec2 feature extractor (CNN, gelé)
        ↓
Wav2Vec2 transformer (12 couches, fine-tuné)
        ↓
Mean pooling temporel
        ↓
Classifier (Linear → ReLU → Dropout → Linear)
        ↓
8 émotions
```

**Paramètres :** 94.5M total | 90.4M entraînables

---

## ⚙️ Pipeline de traitement

### Preprocessing audio
- Sample rate : 16 000 Hz
- Durée fixée : 4 secondes (padding / découpage centré)
- Normalisation amplitude
- Suppression des silences (trim top_db=20)

### Augmentation (train uniquement)
- Bruit blanc (σ=0.005)
- Décalage temporel (max 0.5s)
- Pitch shift (±2 demi-tons)
- SpecAugment (2× FrequencyMasking + 2× TimeMasking)

### Features extraites (CNN/LSTM)
- Mel-spectrogramme (128 bandes, fmax=8000 Hz)
- MFCC (40 coefficients)

---

## 🚀 Installation

```bash
git clone https://github.com/TON_USERNAME/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt
```

---

## 📓 Exécution sur Kaggle

1. Importer le dataset RAVDESS depuis Kaggle
2. Ouvrir les notebooks dans l'ordre (01 → 06)
3. Le chemin du dataset : `/kaggle/input/datasets/elouardaniilyasse/ravdess/RAVDESS`

---

## 🔧 Configuration entraînement — Wav2Vec2

```python
Modèle      : facebook/wav2vec2-base
Optimizer   : AdamW (LR transformer=1e-5, LR classifier=1e-4)
Loss        : CrossEntropyLoss (class weights + label_smoothing=0.1)
Scheduler   : CosineAnnealingLR
Batch size  : 8
Epochs      : 20 (early stopping patience=5)
```

---

## 📊 Technologies utilisées

- **Python** 3.10
- **PyTorch** 2.0
- **HuggingFace Transformers** (Wav2Vec2)
- **Librosa** (traitement audio)
- **Scikit-learn** (métriques)
- **Matplotlib / Seaborn** (visualisation)
- **Kaggle** (entraînement GPU)

---

## 👤 Auteur

**Elouardani Ilyasse**  
Projet réalisé sur Kaggle — 2025
