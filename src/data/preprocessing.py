import os
from bpe_tokenizer import BPETokenizer

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
    print("\n=== First 500 Characters ===")
    print(text[:500])

def build_and_save_tokenizer(text):
    tokenizer = BPETokenizer()
    tokenizer.train(text, num_merges=3000)
    tokenizer.save(os.path.join(DATASET_DIR, "bpe_vocab.json"))
    return tokenizer

def split_and_save(text, tokenizer):
    data = tokenizer.encode(text)
    n = len(data)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)

    splits = {
        "train": data[:train_end],
        "val":   data[train_end:val_end],
        "test":  data[val_end:]
    }

    for name, tokens in splits.items():
        # Save token ids as integers, one per line
        path = os.path.join(DATASET_DIR, f"{name}.ids")
        with open(path, "w") as f:
            f.write("\n".join(map(str, tokens)))
        print(f"{name}.ids — {len(tokens):,} tokens")

if __name__ == "__main__":
    text = load_raw_text()
    explore(text)

    print("\n=== Training BPE Tokenizer ===")
    tokenizer = build_and_save_tokenizer(text)

    print("\n=== Splitting Dataset ===")
    split_and_save(text, tokenizer)

    print("\n=== Verification ===")
    print(f"Vocab size : {tokenizer.vocab_size()}")
    sample_ids = tokenizer.encode(text[:200])
    print(f"First 200 chars → {len(sample_ids)} tokens")
    print(f"Decoded    : {repr(tokenizer.decode(sample_ids))}")