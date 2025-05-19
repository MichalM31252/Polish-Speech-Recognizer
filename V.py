import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
DATA_DIR = "data"

# Hyperparameters
MAX_TEXT_LEN = 100
MAX_NUM_WORDS = 10000
SAMPLE_RATE = 16000
MFCC_DIM = 13
AUDIO_DURATION = 3  # seconds
MFCC_MAX_LEN = int(AUDIO_DURATION * (SAMPLE_RATE // 512))  # roughly frames per second

def load_data(data_dir):
    audio_data = []
    text_data = []
    labels_authors = []
    authors = os.listdir(data_dir)

    for author in authors:
        author_dir = os.path.join(data_dir, author)
        if not os.path.isdir(author_dir):
            continue

        for file in os.listdir(author_dir):
            if file.endswith(".wav"):
                base_name = file[:-4]
                wav_path = os.path.join(author_dir, base_name + ".wav")
                txt_path = os.path.join(author_dir, base_name + ".txt")

                # Load audio
                try:
                    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
                    y = librosa.util.fix_length(y, size=AUDIO_DURATION * SAMPLE_RATE)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_DIM)
                    mfcc = mfcc.T  # (time, features)
                    if mfcc.shape[0] > MFCC_MAX_LEN:
                        mfcc = mfcc[:MFCC_MAX_LEN]
                    else:
                        pad_width = MFCC_MAX_LEN - mfcc.shape[0]
                        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
                except Exception as e:
                    print(f"Error loading {wav_path}: {e}")
                    continue

                # Load text
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                except Exception as e:
                    print(f"Error reading {txt_path}: {e}")
                    continue

                audio_data.append(mfcc)
                text_data.append(text)
                labels_authors.append(author)

    return np.array(audio_data), text_data, labels_authors

# Load the dataset
audio_features, text_sentences, label_names = load_data(DATA_DIR)

# Encode labels
label_encoder = LabelEncoder()
labels_authors = label_encoder.fit_transform(label_names)
num_classes = len(label_encoder.classes_)

# Process text
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(text_sentences)
text_sequences = tokenizer.texts_to_sequences(text_sentences)
text_padded = pad_sequences(text_sequences, maxlen=MAX_TEXT_LEN, padding='post')

# Train/test split
X_audio_train, X_audio_test, X_text_train, X_text_test, y_train_authors, y_test_authors = train_test_split(
    audio_features, text_padded, labels_authors, test_size=0.2, random_state=42, stratify=labels
)

# Build the model
def build_model():
    # Audio branch
    audio_input = Input(shape=(MFCC_MAX_LEN, MFCC_DIM), name='audio_input')
    x_audio = LSTM(64)(audio_input)

    # Text branch
    text_input = Input(shape=(MAX_TEXT_LEN,), name='text_input')
    x_text = Embedding(MAX_NUM_WORDS, 64)(text_input)
    x_text = GlobalAveragePooling1D()(x_text)

    # Combine
    combined = Concatenate()([x_audio, x_text])
    x = Dense(64, activation='relu')(combined)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[audio_input, text_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# Train the model
history = model.fit(
    x = {'audio_input': X_audio_train, 'text_input': X_text_train},
    y = y_train_authors,
    validation_data = (
        {'audio_input': X_audio_test, 'text_input': X_text_test},
        y_test_authors
    ),
    epochs=20,
    batch_size=16
)

# Evaluate
test_loss, test_acc = model.evaluate(
    x = {'audio_input': X_audio_test, 'text_input': X_text_test},
    y = y_test_authors
)

import matplotlib.pyplot as plt

# Plotting training history
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_history(history)


print(f"\nTest Accuracy: {test_acc:.2f}")
