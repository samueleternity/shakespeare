import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os
from src.model.architecture import (
    build_lstm_model,
    build_transformer_model,
    TokenAndPositionEmbedding,
    TransformerBlock
)

DATASET_DIR    = "datasets"
CHECKPOINT_DIR = "checkpoints"

def load_vocab():
    path = os.path.join(DATASET_DIR, "vocab.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stoi = data["stoi"]
    itos = {int(k): v for k, v in data["itos"].items()}
    return stoi, itos

def sample_top_k(logits, temperature=0.8, top_k=10):
    logits = logits / temperature

    # Zero out everything outside top-k
    top_k_idx    = np.argsort(logits)[-top_k:]
    filtered     = np.full_like(logits, -np.inf)
    filtered[top_k_idx] = logits[top_k_idx]

    filtered -= filtered.max()
    probs     = np.exp(filtered)
    probs    /= probs.sum()

    return np.random.choice(len(probs), p=probs)

def generate(seed_text, num_chars=500, temperature=0.8, top_k=10, max_context=200):
    stoi, itos = load_vocab()

    model = keras.models.load_model(
        os.path.join(CHECKPOINT_DIR, "best_model.keras"),
        custom_objects={
            "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
            "TransformerBlock":          TransformerBlock,
        }
    )

    @tf.function(reduce_retracing=True)
    def predict_step(x):
        return model(x, training=False)

    input_ids = [stoi[ch] for ch in seed_text if ch in stoi]
    generated = seed_text

    print("Generating", end="", flush=True)
    for i in range(num_chars):
        context = input_ids[-max_context:]
        x       = tf.constant([context], dtype=tf.int32)
        logits  = predict_step(x)[0, -1, :].numpy()

        next_id = sample_top_k(logits, temperature=temperature, top_k=top_k)
        input_ids.append(next_id)
        generated += itos[next_id]

        if i % 50 == 0:
            print(".", end="", flush=True)

    print(" done\n")
    return generated

if __name__ == "__main__":
    seeds = [
        "First Citizen:\n",
        "HAMLET:\nTo be or not",
        "KING EDWARD IV:\nWhat say you, lords?",
    ]

    for seed in seeds:
        print(f"Seed: {repr(seed)}")
        print("=" * 50)
        print(generate(seed_text=seed, num_chars=500, temperature=0.8, top_k=10))
        print("=" * 50 + "\n")