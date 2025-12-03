import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

# ------------------------------------------------------
# LOAD DATA FAST
# ------------------------------------------------------

FILE_PATH = "/Users/shanmuka/Desktop/project1/Test/general_corpus.txt"

with open(FILE_PATH, encoding="utf-8") as f:
    data = f.read()

corpus = [line.strip().lower() for line in data.split("\n") if line.strip()]

# Smaller dataset = faster training
corpus = corpus[:2000]

# ------------------------------------------------------
# TOKENIZER
# ------------------------------------------------------

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(corpus)

vocab_size = 5000

# ------------------------------------------------------
# TRAINING SEQUENCES
# ------------------------------------------------------

input_sequences = []
MAX_LEN = 15

for line in corpus:
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        seq = tokens[:i+1]
        input_sequences.append(seq)

input_sequences = pad_sequences(input_sequences, maxlen=MAX_LEN, padding='pre')

xs = input_sequences[:, :-1]
ys = input_sequences[:, -1]

print("Training Samples:", len(xs))

# ------------------------------------------------------
# SMALL & FAST MODEL
# ------------------------------------------------------

model = Sequential([
    Embedding(vocab_size, 32, input_length=MAX_LEN-1),
    LSTM(32),
    Dense(vocab_size, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(0.01),
    metrics=["accuracy"]
)

model.summary()

# ------------------------------------------------------
# MORE EPOCHS (still fast)
# ------------------------------------------------------

EPOCHS = 30   # Increase if you want (50, 100, etc.)

history = model.fit(xs, ys, epochs=EPOCHS, batch_size=128, verbose=1)

# ------------------------------------------------------
# TEXT GENERATION
# ------------------------------------------------------

seed_text = "i have a strange feeling"
next_words = 40  # You can increase this too

for _ in range(next_words):
    seq = tokenizer.texts_to_sequences([seed_text])[0]
    seq = pad_sequences([seq], maxlen=MAX_LEN-1, padding='pre')
    pred = np.argmax(model.predict(seq), axis=-1)[0]

    for word, idx in tokenizer.word_index.items():
        if idx == pred:
            seed_text += " " + word
            break

print("\nGenerated Text:")
print(seed_text)