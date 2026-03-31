import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.model.architecture import (
    build_lstm_model,
    build_transformer_model,
    TokenAndPositionEmbedding,
    TransformerBlock
)
from src.data.bpe_tokenizer import BPETokenizer

DATASET_DIR   = "datasets"
CHECKPOINT_DIR = "checkpoints"
SEQ_LENGTH    = 200
BATCH_SIZE    = 64

def load_tokenizer():
    return BPETokenizer.load(os.path.join(DATASET_DIR, "bpe_vocab.json"))

def load_ids(filename):
    path = os.path.join(DATASET_DIR, filename)
    with open(path, "r") as f:
        return np.array([int(line) for line in f if line.strip()], dtype=np.int32)


def make_dataset(encoded, seq_length, batch_size):
    data = tf.data.Dataset.from_tensor_slices(encoded)
    sequences = data.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(seq):
        return seq[:-1], seq[1:]

    return (
        sequences
        .map(split_input_target)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def evaluate_split(model, encoded, split_name):
    ds = make_dataset(encoded, SEQ_LENGTH, BATCH_SIZE)
    loss_metric = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for x_batch, y_batch in ds:
        logits = model(x_batch, training=False)

        # Loss
        loss = loss_metric(y_batch, logits)
        total_loss += loss.numpy()
        num_batches += 1

        # Accuracy
        predictions = tf.argmax(logits, axis=-1)
        correct = tf.cast(tf.equal(predictions, tf.cast(y_batch, tf.int64)), tf.float32)
        total_correct += tf.reduce_sum(correct).numpy()
        total_tokens  += tf.size(y_batch).numpy()

    avg_loss    = total_loss / num_batches
    perplexity  = np.exp(avg_loss)
    accuracy    = total_correct / total_tokens

    print(f"\n{'='*40}")
    print(f"  Split      : {split_name}")
    print(f"  Loss       : {avg_loss:.4f}")
    print(f"  Perplexity : {perplexity:.2f}")
    print(f"  Accuracy   : {accuracy*100:.2f}%")
    print(f"  Tokens     : {total_tokens:,}")
    print(f"{'='*40}")

    return avg_loss, perplexity, accuracy


def evaluate_all(config=None, notes=""):
    tokenizer  = load_tokenizer()
    vocab_size = tokenizer.vocab_size()

    print("Loading model from checkpoint...")
    model = keras.models.load_model(
        os.path.join(CHECKPOINT_DIR, "best_model.keras"),
        custom_objects={
            "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
            "TransformerBlock": TransformerBlock,
        }
    )

    results = {}
    
    for split in ["train", "val", "test"]:
        filename = f"{split}.ids"
        encoded  = load_ids(filename)
        loss, ppl, acc = evaluate_split(model, encoded, split.upper())
        results[split] = {
            "loss": float(loss),
            "perplexity": float(ppl),
            "accuracy": float(acc)
        }

    # Summary comparison
    print("\n\nSUMMARY")
    print(f"{'Split':<10} {'Loss':>8} {'Perplexity':>12} {'Accuracy':>10}")
    print("-" * 44)
    for split, m in results.items():
        print(f"{split:<10} {m['loss']:>8.4f} {m['perplexity']:>12.2f} {m['accuracy']*100:>9.2f}%")

    # Overfitting check
    gap = results["val"]["loss"] - results["train"]["loss"]
    print(f"\nTrain/Val loss gap : {gap:.4f}", end="  ")
    if gap < 0.1:
        print("(excellent — no overfitting)")
    elif gap < 0.3:
        print("(acceptable — slight overfitting)")
    else:
        print("(warning — consider more dropout or less capacity)")

    # Log experiment
    if config is None:
        config = {}
    from src.utils.experiment_tracker import log_experiment
    log_experiment(config, results, notes=notes)

    # Save results
    os.makedirs("outputs", exist_ok=True)
    results_path = os.path.join("outputs", "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    config = {
        "epochs_run":    80,
        "seq_length":    200,
        "batch_size":    64,
        "embed_dim":     128,
        "lstm_units":    256,
        "NUM_HEADS":     4,
        "FF_DIM":        512,
        "num_layers":    2,
        "dropout":       0.1,
        "learning_rate": 0.0005,
    }
    notes = "Iteration 11 - dataset change through BPE, 200 steps/epoch for first run"

    evaluate_all(config=config, notes=notes)