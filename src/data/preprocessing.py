import os
import json

DATASET_DIR = "datasets"

def load_raw_text():
    path = os.path.join(DATASET_DIR, "raw.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def explore(text):
    print("=== Basic Stats ===")
    print(f"Total characters : {len(text):,}")
    print(f"Total lines      : {text.count(chr(10)):,}")
    print(f"Total words      : {len(text.split()):,}")

    print("\n=== Unique Characters ===")
    vocab = sorted(set(text))
    print(f"Vocabulary size  : {len(vocab)}")
    print(f"Characters       : {repr(''.join(vocab))}")

    print("\n=== First 500 Characters ===")
    print(text[:500])

def build_vocab(text):
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}  # char → int
    itos = {i: ch for i, ch in enumerate(vocab)}  # int → char
    return vocab, stoi, itos

def save_vocab(stoi, itos):
    vocab_data = {
        "stoi": stoi,
        "itos": {int(k): v for k, v in itos.items()}
    }
    path = os.path.join(DATASET_DIR, "vocab.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    print(f"Vocab saved to {path} ({len(stoi)} tokens)")

def encode(text, stoi):
    return [stoi[ch] for ch in text]

def decode(indices, itos):
    return ''.join([itos[i] for i in indices])

def split_and_save(text, stoi):
    data = encode(text, stoi)
    n = len(data)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)

    splits = {
        "train": data[:train_end],
        "val":   data[train_end:val_end],
        "test":  data[val_end:]
    }

    for name, tokens in splits.items():
        path = os.path.join(DATASET_DIR, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(decode(tokens, itos))
        print(f"{name}.txt — {len(tokens):,} tokens")

if __name__ == "__main__":
    text = load_raw_text()
    explore(text)

    print("\n=== Building Vocab & Splitting ===")
    vocab, stoi, itos = build_vocab(text)
    save_vocab(stoi, itos)
    split_and_save(text, stoi)