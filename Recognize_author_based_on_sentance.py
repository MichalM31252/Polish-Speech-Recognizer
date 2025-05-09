import os
import numpy as np
import librosa
from textgrid import TextGrid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# === CONFIG ===
DATA_DIR = "data"
SAMPLE_RATE = 16000
N_MFCC = 13
MAX_LEN = 130

# === FUNCTIONS ===

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC).T
    if len(mfcc) > MAX_LEN:
        mfcc = mfcc[:MAX_LEN]
    else:
        pad_width = MAX_LEN - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    return mfcc

def parse_textgrid(tg_path):
    tg = TextGrid.fromFile(tg_path)
    word_intervals = []
    for tier in tg.tiers:
        if tier.name.lower() in ['words', 'word']:
            for interval in tier.intervals:
                if interval.mark.strip():
                    word_intervals.append((interval.mark, interval.minTime, interval.maxTime))
    return word_intervals

def extract_prosody(word_intervals):
    word_durations = [end - start for _, start, end in word_intervals]
    total_time = word_intervals[-1][2] - word_intervals[0][1] if word_intervals else 1
    num_words = len(word_intervals)

    speaking_rate = num_words / total_time
    avg_word_duration = np.mean(word_durations) if word_durations else 0
    pause_durations = [word_intervals[i+1][1] - word_intervals[i][2] for i in range(len(word_intervals)-1)]
    avg_pause = np.mean(pause_durations) if pause_durations else 0

    return [speaking_rate, avg_word_duration, avg_pause]

# === DATA COLLECTION ===

mfcc_features = []
transcriptions = []
prosody_features = []
labels = []

for author in os.listdir(DATA_DIR):
    author_path = os.path.join(DATA_DIR, author)
    if os.path.isdir(author_path):
        for file in os.listdir(author_path):
            if file.endswith(".wav"):
                wav_path = os.path.join(author_path, file)
                txt_path = wav_path.replace(".wav", ".txt")
                tg_path = wav_path.replace(".wav", ".TextGrid")

                if not os.path.exists(txt_path) or not os.path.exists(tg_path):
                    continue

                try:
                    mfcc = extract_mfcc(wav_path)

                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()

                    word_intervals = parse_textgrid(tg_path)
                    prosody = extract_prosody(word_intervals)

                    mfcc_features.append(mfcc)
                    transcriptions.append(text)
                    prosody_features.append(prosody)
                    labels.append(author)
                except Exception as e:
                    print(f"Error processing {wav_path}: {e}")

# === PREPROCESSING ===

mfcc_features = np.array(mfcc_features)
prosody_features = np.array(prosody_features)
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
y_cat = to_categorical(y_encoded)

# Text to vector
vectorizer = TfidfVectorizer(max_features=100)
X_text = vectorizer.fit_transform(transcriptions).toarray()

# Train-test split
X_mfcc_train, X_mfcc_test, X_text_train, X_text_test, X_prosody_train, X_prosody_test, y_train, y_test = train_test_split(
    mfcc_features, X_text, prosody_features, y_cat, test_size=0.2, random_state=42)

# === BUILD MULTI-INPUT MODEL ===

# Input 1: MFCC
input_mfcc = Input(shape=(MAX_LEN, N_MFCC))
x1 = Flatten()(input_mfcc)
x1 = Dense(128, activation='relu')(x1)
x1 = Dropout(0.3)(x1)

# Input 2: Text
input_text = Input(shape=(X_text.shape[1],))
x2 = Dense(64, activation='relu')(input_text)

# Input 3: Prosodic features
input_prosody = Input(shape=(3,))
x3 = Dense(32, activation='relu')(input_prosody)

# Combine
combined = Concatenate()([x1, x2, x3])
z = Dense(128, activation='relu')(combined)
z = Dropout(0.3)(z)
output = Dense(len(np.unique(labels)), activation='softmax')(z)

model = Model(inputs=[input_mfcc, input_text, input_prosody], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === TRAIN ===
model.fit(
    [X_mfcc_train, X_text_train, X_prosody_train], y_train,
    validation_split=0.2, epochs=100, batch_size=32
)

# === EVALUATE ===
loss, acc = model.evaluate([X_mfcc_test, X_text_test, X_prosody_test], y_test)
print(f"Test accuracy: {acc:.4f}")
