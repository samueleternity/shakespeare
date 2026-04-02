import numpy as np

train = np.array([int(l) for l in open("datasets/train.ids") if l.strip()])
val   = np.array([int(l) for l in open("datasets/val.ids")   if l.strip()])
test  = np.array([int(l) for l in open("datasets/test.ids")  if l.strip()])

from bpe_tokenizer import BPETokenizer
tok = BPETokenizer.load("datasets/bpe_vocab.json")

print(f"Vocab size:    {tok.vocab_size():,}")
print(f"Train tokens:  {len(train):,}")
print(f"Val tokens:    {len(val):,}")
print(f"Test tokens:   {len(test):,}")
print(f"Train/Vocab ratio: {len(train)/tok.vocab_size():.1f}x")