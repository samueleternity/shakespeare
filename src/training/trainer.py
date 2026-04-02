import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.model.architecture import build_lstm_model, build_transformer_model
from src.data.bpe_tokenizer import BPETokenizer

MODEL_TYPE     = "transformer"
DATASET_DIR    = "datasets"
CHECKPOINT_DIR = "checkpoints"
SEQ_LENGTH     = 50
BATCH_SIZE     = 64
EPOCHS         = 80
LSTM_UNITS     = 256
EMBED_DIM      = 256
NUM_HEADS      = 8
FF_DIM         = 1024
NUM_LAYERS     = 4
DROPOUT        = 0.1
LEARNING_RATE  = 0.0001

def load_tokenizer():
    return BPETokenizer.load(os.path.join(DATASET_DIR, "bpe_vocab.json"))

def load_ids(filename):
    path = os.path.join(DATASET_DIR, filename)
    with open(path, "r") as f:
        return np.array([int(line) for line in f if line.strip()], dtype=np.int32)

def make_train_dataset(encoded, seq_length, batch_size):
    data      = tf.data.Dataset.from_tensor_slices(encoded)
    sequences = data.window(seq_length + 1, shift=1, drop_remainder=True)
    sequences = sequences.flat_map(lambda w: w.batch(seq_length + 1))

    def split_input_target(seq):
        return seq[:-1], seq[1:]

    return (
        sequences
        .map(split_input_target)
        .shuffle(buffer_size=10000)
        .batch(batch_size, drop_remainder=True)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

def make_val_dataset(encoded, seq_length, batch_size):
    data      = tf.data.Dataset.from_tensor_slices(encoded)
    sequences = data.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(seq):
        return seq[:-1], seq[1:]

    return (
        sequences
        .map(split_input_target)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

def get_train_steps(encoded, seq_length, batch_size):
    return (len(encoded) - seq_length) // batch_size

def train():
    tokenizer  = load_tokenizer()
    vocab_size = tokenizer.vocab_size()
    print(f"Vocab size: {vocab_size}")

    train_encoded = load_ids("train.ids")
    val_encoded   = load_ids("val.ids")
    print(f"Train tokens: {len(train_encoded):,} | Val tokens: {len(val_encoded):,}")

    train_ds    = make_train_dataset(train_encoded, SEQ_LENGTH, BATCH_SIZE)
    val_ds      = make_val_dataset(val_encoded,     SEQ_LENGTH, BATCH_SIZE)
    train_steps = get_train_steps(train_encoded,    SEQ_LENGTH, BATCH_SIZE)
    print(f"Steps per epoch: {train_steps:,} → capped at 500")

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
    os.makedirs("outputs", exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        keras.callbacks.CSVLogger("outputs/training_log.csv"),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        ),
    ]

    print("\nStarting training...\n")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=500,
        callbacks=callbacks,
    )

    print("\nTraining complete.")
    print(f"Best model saved to {CHECKPOINT_DIR}/best_model.keras")

if __name__ == "__main__":
    train()