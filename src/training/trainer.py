import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.architecture import build_lstm_model, build_transformer_model

MODEL_TYPE    = "transformer" 

DATASET_DIR = "datasets"
CHECKPOINT_DIR = "checkpoints"
SEQ_LENGTH = 200
BATCH_SIZE = 64
EPOCHS = 80

LSTM_UNITS = 256

EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 0.0005


def load_vocab():
    path = os.path.join(DATASET_DIR, "vocab.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stoi = data["stoi"]
    itos = {int(k): v for k, v in data["itos"].items()}
    return stoi, itos


def load_and_encode(filename, stoi):
    path = os.path.join(DATASET_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return np.array([stoi[ch] for ch in text if ch in stoi], dtype=np.int32)


def make_dataset(encoded, seq_length, batch_size):
    # Each sample: input = chars 0..N-1, target = chars 1..N (shifted by 1)
    data = tf.data.Dataset.from_tensor_slices(encoded)
    sequences = data.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(seq):
        return seq[:-1], seq[1:]

    return (
        sequences
        .map(split_input_target)
        .shuffle(1000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def train():
    stoi, itos = load_vocab()
    vocab_size = len(stoi)
    print(f"Vocab size: {vocab_size}")

    train_encoded = load_and_encode("train.txt", stoi)
    val_encoded   = load_and_encode("val.txt", stoi)
    print(f"Train tokens: {len(train_encoded):,} | Val tokens: {len(val_encoded):,}")

    train_ds = make_dataset(train_encoded, SEQ_LENGTH, BATCH_SIZE)
    val_ds   = make_dataset(val_encoded,   SEQ_LENGTH, BATCH_SIZE)

    if MODEL_TYPE == "transformer":
        model = build_transformer_model(
            vocab_size=vocab_size,
            seq_length=SEQ_LENGTH,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
    else:
        model = build_lstm_model(
            vocab_size=vocab_size,
            embed_dim=EMBED_DIM,
            lstm_units=LSTM_UNITS,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )

    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE, clipnorm=1.0, weight_decay=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    callbacks = [
        # Save best model by validation loss
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        # Stop early if val_loss stops improving for 3 epochs
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=1,
            restore_best_weights=True
        ),
        # Log to CSV for later analysis
        keras.callbacks.CSVLogger("outputs/training_log.csv"),

        # Reduce learning rate if val_loss plateaus for 2 epochs
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,        
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
    ]

    os.makedirs("outputs", exist_ok=True)

    print("\nStarting training...\n")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("\nTraining complete.")
    print(f"Best model saved to {CHECKPOINT_DIR}/best_model.keras")


if __name__ == "__main__":
    train()