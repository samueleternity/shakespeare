import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os

DATASET_DIR = "datasets"
CHECKPOINT_DIR = "checkpoints"


def load_vocab():
    path = os.path.join(DATASET_DIR, "vocab.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stoi = data["stoi"]
    itos = {int(k): v for k, v in data["itos"].items()}
    return stoi, itos


def generate(seed_text, num_chars=500, temperature=0.5):
    stoi, itos = load_vocab()

    model = keras.models.load_model(
        os.path.join(CHECKPOINT_DIR, "best_model.keras")
    )

    # Encode the seed text into integers
    input_ids = [stoi[ch] for ch in seed_text if ch in stoi]

    generated = seed_text

    for _ in range(num_chars):
        x = np.array([input_ids])                      # shape [1, seq_len]
        logits = model.predict(x, verbose=0)            # shape [1, seq_len, 65]
        next_logits = logits[0, -1, :]                  # take last position

        # Temperature controls randomness:
        # low  (0.5) = conservative, repetitive
        # high (1.5) = creative, chaotic
        next_logits = next_logits / temperature
        probs = tf.nn.softmax(next_logits).numpy()

        next_id = np.random.choice(len(probs), p=probs) # sample from distribution
        input_ids.append(next_id)
        generated += itos[next_id]

    return generated


if __name__ == "__main__":
    seed = "First Citizen:\n"
    print(f"Seed: {repr(seed)}\n")
    print("=" * 50)
    print(generate(seed_text=seed, num_chars=500, temperature=0.8))
    print("=" * 50)