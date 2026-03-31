import json
import re
from collections import Counter
from pathlib import Path

# GPT-style whitespace marker
SPACE_MARKER = "Ġ"  # U+0120, visually distinct, never appears in Shakespeare

def _get_word_freqs(text: str) -> Counter:
    words = Counter()
    # Split on whitespace, preserve which words follow a space
    tokens = re.findall(r"\S+", text)
    for i, token in enumerate(tokens):
        prefix = SPACE_MARKER if i > 0 else ""
        word = tuple(prefix + token[0]) + tuple(token[1:]) if prefix else tuple(token)
        words[word] += 1
    return words


def _get_pairs(word_freqs: dict) -> Counter:
    """Count all adjacent symbol pairs across the entire word vocabulary."""
    pairs = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def _merge_pair(pair: tuple, word_freqs: dict) -> dict:
    """
    Merge every occurrence of `pair` in the word vocabulary.
    ("h","e") → "he" everywhere it appears.
    """
    new_word_freqs = {}
    a, b = pair
    merged = a + b

    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq

    return new_word_freqs

class BPETokenizer:
    def __init__(self):
        self.merges: list[tuple] = []       # ordered merge rules
        self.vocab: dict[str, int] = {}     # token → id
        self.itos: dict[int, str] = {}      # id → token
        self._trained = False

    def train(self, text: str, num_merges: int = 3000, verbose: bool = True):
        """
        Train BPE on raw text.

        Args:
            text:       raw corpus string
            num_merges: number of merge operations (controls vocab size)
            verbose:    print progress every 500 merges
        """
        if verbose:
            print(f"Building initial word frequencies...")

        word_freqs = _get_word_freqs(text)

        if verbose:
            print(f"  Unique words   : {len(word_freqs):,}")
            unique_chars = set(ch for word in word_freqs for ch in word)
            print(f"  Base characters: {len(unique_chars)}")
            print(f"  Target merges  : {num_merges}")
            print(f"  Final vocab    : ~{len(unique_chars) + num_merges} tokens\n")

        self.merges = []

        for i in range(num_merges):
            pairs = _get_pairs(word_freqs)
            if not pairs:
                print(f"No more pairs to merge at step {i}. Stopping.")
                break

            best_pair = max(pairs, key=pairs.get)
            word_freqs = _merge_pair(best_pair, word_freqs)
            self.merges.append(best_pair)

            if verbose and i % 500 == 0:
                print(f"  Merge {i:>4}/{num_merges} | "
                      f"{''.join(best_pair):<20} | "
                      f"freq={pairs[best_pair]:,}")

        if verbose:
            print(f"\nTraining complete. {len(self.merges)} merges learned.")

        self._build_vocab(word_freqs)
        self._trained = True

        if verbose:
            print(f"Vocab size: {len(self.vocab)}")

    def _build_vocab(self, word_freqs: dict):
        """
        Construct token→id mapping from all tokens that appear
        in the final word vocabulary after all merges.
        """
        tokens = set()
        for word in word_freqs:
            for symbol in word:
                tokens.add(symbol)

        # Sort for determinism
        sorted_tokens = sorted(tokens)

        self.vocab = {"<unk>": 0}
        for i, token in enumerate(sorted_tokens, start=1):
            self.vocab[token] = i

        self.itos = {i: token for token, i in self.vocab.items()}

    def _tokenize_word(self, word: tuple) -> list[str]:
        """
        Apply learned merge rules to a single word (as a char tuple).
        Replays the merge sequence in order.
        """
        symbols = list(word)

        for merge_a, merge_b in self.merges:
            merged = merge_a + merge_b
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == merge_a and symbols[i + 1] == merge_b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(self, text: str) -> list[int]:
        """
        Convert a string to a list of token ids.
        Unknown tokens map to id 0 (<unk>).
        """
        assert self._trained, "Tokenizer must be trained or loaded before encoding."

        word_freqs = _get_word_freqs(text)
        tokens = []

        raw_tokens = re.findall(r"\S+", text)
        for i, raw in enumerate(raw_tokens):
            prefix = SPACE_MARKER if i > 0 else ""
            word_tuple = tuple(prefix + raw[0]) + tuple(raw[1:]) if prefix else tuple(raw)
            symbols = self._tokenize_word(word_tuple)
            for symbol in symbols:
                tokens.append(self.vocab.get(symbol, 0))  # 0 = <unk>

        return tokens

    def decode(self, ids: list[int]) -> str:
        """
        Convert a list of token ids back to a string.
        Replaces SPACE_MARKER with a real space.
        """
        tokens = [self.itos.get(i, "<unk>") for i in ids]
        text = "".join(tokens)
        text = text.replace(SPACE_MARKER, " ")
        return text

    def vocab_size(self) -> int:
        return len(self.vocab)

    def show_sample_tokens(self, n: int = 30):
        """Print a sample of learned tokens (skipping single chars)."""
        multi_char = [t for t in self.vocab if len(t) > 1 and t != "<unk>"]
        print(f"Sample merged tokens ({min(n, len(multi_char))} of {len(multi_char)}):")
        for token in multi_char[:n]:
            print(f"  {repr(token):<20} → id {self.vocab[token]}")

    def show_tokenization(self, text: str):
        """Show how a piece of text is tokenized."""
        raw_tokens = re.findall(r"\S+", text)
        print(f"Input: {repr(text)}")
        print(f"Tokens:")
        for i, raw in enumerate(raw_tokens):
            prefix = SPACE_MARKER if i > 0 else ""
            word_tuple = tuple(prefix + raw[0]) + tuple(raw[1:]) if prefix else tuple(raw)
            symbols = self._tokenize_word(word_tuple)
            ids = [self.vocab.get(s, 0) for s in symbols]
            print(f"  {raw:<20} → {symbols}  {ids}")

    def save(self, path: str):
        """Save merges and vocab to a JSON file."""
        data = {
            "merges": [list(pair) for pair in self.merges],
            "vocab":  self.vocab,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path} ({len(self.vocab)} tokens, {len(self.merges)} merges)")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a previously trained tokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer.vocab  = data["vocab"]
        tokenizer.itos   = {int(i): token for token, i in data["vocab"].items()}
        tokenizer._trained = True
        print(f"Tokenizer loaded from {path} ({len(tokenizer.vocab)} tokens, {len(tokenizer.merges)} merges)")
        return tokenizer

if __name__ == "__main__":
    import os

    DATASET_DIR = "datasets"
    SAVE_PATH   = os.path.join(DATASET_DIR, "bpe_vocab.json")

    with open(os.path.join(DATASET_DIR, "raw.txt"), "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Corpus: {len(text):,} characters\n")

    # --- Experiment: show how vocab size changes with num_merges ---
    print("=" * 60)
    print("VOCAB SIZE EXPERIMENT")
    print("=" * 60)
    for num_merges in [500, 1000, 2000, 3000]:
        t = BPETokenizer()
        t.train(text, num_merges=num_merges, verbose=False)
        encoded = t.encode(text[:10000])
        compression = len(text[:10000]) / len(encoded)
        print(f"  merges={num_merges:<5} vocab={t.vocab_size():<6} "
              f"compression={compression:.2f}x")

    # --- Train final tokenizer ---
    print("\n" + "=" * 60)
    print("TRAINING FINAL TOKENIZER (3000 merges)")
    print("=" * 60)
    tokenizer = BPETokenizer()
    tokenizer.train(text, num_merges=3000, verbose=True)

    # --- Show sample tokens ---
    print("\n" + "=" * 60)
    print("SAMPLE MERGED TOKENS")
    print("=" * 60)
    tokenizer.show_sample_tokens(n=40)

    # --- Show tokenization of Shakespeare-specific phrases ---
    print("\n" + "=" * 60)
    print("TOKENIZATION EXAMPLES")
    print("=" * 60)
    examples = [
        "To be or not to be",
        "What's in a name?",
        "thou hast spoken no word all this while",
        "KING EDWARD IV:",
        "banish'd from the world",
    ]
    for example in examples:
        tokenizer.show_tokenization(example)
        ids = tokenizer.encode(example)
        decoded = tokenizer.decode(ids)
        print(f"  chars={len(example)} → tokens={len(ids)} "
              f"| decode_ok={decoded.strip() == example.strip()}")
        print()

    # --- Save ---
    print("=" * 60)
    tokenizer.save(SAVE_PATH)