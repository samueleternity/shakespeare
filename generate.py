import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os
from src.model.architecture import (
    build_transformer_model,
    TokenAndPositionEmbedding,
    TransformerBlock
)

DATASET_DIR = "datasets"
CHECKPOINT_DIR = "checkpoints"


def load_vocab():
    path = os.path.join(DATASET_DIR, "vocab.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stoi = data["stoi"]
    itos = {int(k): v for k, v in data["itos"].items()}
    return stoi, itos


def generate(seed_text, num_chars=100, temperature=0.3, max_model_len=256):
    stoi, itos = load_vocab()
    
    model = keras.models.load_model(
        os.path.join(CHECKPOINT_DIR, "best_model.keras"),
        custom_objects={
            "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
            "TransformerBlock": TransformerBlock,
        }
    )

    input_ids = [stoi[ch] for ch in seed_text if ch in stoi]
    generated = seed_text

    print("Generating...")
    for i in range(num_chars):
        x = np.array([input_ids[-max_model_len:]]) 
        
        logits = model.predict(x, verbose=0)
        next_logits = logits[0, -1, :]
        
        next_logits = next_logits / temperature
        probs = tf.nn.softmax(next_logits).numpy()

        next_id = np.random.choice(len(probs), p=probs)
        input_ids.append(next_id)
        
        char = itos[next_id]
        generated += char
        
        print(char, end="", flush=True) 

    return generated


if __name__ == "__main__":
    seed = "First Citizen:\n"
    print(f"Seed: {repr(seed)}\n")
    print("=" * 50)
    print(generate(seed_text=seed, num_chars=500, temperature=0.8))
    print("=" * 50)