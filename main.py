import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# Krótki opis co sie dzieje:
# 1. Wyciągamy cechy MFCC z audio.
# 2. Wczytujemy dane audio z katalogu i zapisujemy ich cechy oraz etykiety.
# 3. Dzielimy dane na zestawy treningowe i testowe.
# 4. Tworzymy model sieci neuronowej do rozpoznawania autorów przy pomocy frameworku Tenserflow (Keras).
# 5. Trenujemy model na danych treningowych z wybranego folderu. (1 zestaw danych)
# 6. Oceniamy celność modelu na danych testowych. (porównanie tego co uważa model z tym jak jest naprawde)

# Tensorflow wykorzysuje Backpropagation do trenowania modelu, czyli algorytm który uczy się na podstawie błędów.

directory_name = "1-500"
SAMPLE_RATE = 16000
MAX_LEN = 130  # Max length of MFCC feature vectors
N_MFCC = 13    # Number of MFCCs

# 1. Wyciąganie cech MFCC z pliku audio
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T
    if len(mfcc) > MAX_LEN:
        mfcc = mfcc[:MAX_LEN]
    else:
        pad_width = MAX_LEN - len(mfcc)
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    return mfcc

# Dwa zbiory danych: X (cechy (to co wyciągneliśmy funkcją)) i y (etykiety - to co jest podpisane w folderach)
X = []
y = []

# 2. Wczytywanie danych audio z katalogu
# Audio jest zapisane w formie
# 1-500/nazwa_autora/
# i wewnątrz folderu są pliki:
# numer.TextGrid - opisuje czas w którym jest wypowiadane każde słowo i litera
# numer.wav - plik audio z nagraniem
# numer.txt - opisuje zdanie które jest wypowiadane w pliku audio
for author in os.listdir(directory_name):
    author_path = os.path.join(directory_name, author)
    if os.path.isdir(author_path):
        for file in os.listdir(author_path):
            if file.endswith(".wav"):
                file_path = os.path.join(author_path, file)
                try:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(author)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

X = np.array(X)
y = np.array(y)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

# 3. Dzielimy dane na zestawy treningowe i testowe
# 80% danych do treningu, 20% do testowania (sprawdzania czy model dobrze przewiduje)
# x_train - cechy audio, y_train - etykiety (poprawne odpowiedzi)
# x_test - cechy audio, y_test - etykiety (poprawne odpowiedzi)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 4. Tworzymy model sieci neuronowej
# Sequential - model sekwencyjny, który składa się z warstw ułożonych jedna po drugiej
# Flatten - spłaszcza dane wejściowe do jednego wymiaru
# Dense - warstwa gęsta, która łączy wszystkie neurony z poprzednią warstwą
# Dropout - warstwa, która losowo wyłącza neurony podczas treningu, aby zapobiec przeuczeniu (overfitting)
model = Sequential([
    Flatten(input_shape=(MAX_LEN, N_MFCC)),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')
])

# Adam optimizer do optymalizacji modelu
# Categorical_crossentropy - funkcja straty do porównania etykiet (y) z przewidywaniami modelu
# Metrics accuracy aby sprawdzić jak dobrze model przewiduje etykiety
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Trenujemy model na danych treningowych
# parametry wybrane na podstawie prób i błędów
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stop])

# 6. Oceniamy celność modelu na danych testowych
# test_loss - pokazuje jak pewny siebie jest model w przewidywaniu etykiet (im niższa wartość tym lepiej)
# test_accuracy - dokładność modelu na danych testowych (im wyższa wartość tym lepiej)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
