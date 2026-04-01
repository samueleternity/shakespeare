import tensorflow as tf
from tensorflow import keras

def build_lstm_model(vocab_size, embed_dim=128, lstm_units=256, num_layers=2, dropout=0.3):
    inputs = keras.Input(shape=(None,), dtype="int32", name="token_ids")

    # 1. Embedding — maps each token integer to a dense vector
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name="embedding"
    )(inputs)

    # 2. Stacked LSTM layers
    for i in range(num_layers):
        return_seq = True  # all layers return sequences
        x = keras.layers.LSTM(
            units=lstm_units,
            return_sequences=return_seq,
            dropout=dropout,
            name=f"lstm_{i+1}"
        )(x)

    # 3. Project back to vocab size — one score per possible next token
    outputs = keras.layers.Dense(vocab_size, name="logits")(x)

    model = keras.Model(inputs, outputs, name="ShakespeareLM")
    return model

class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, seq_length, vocab_size, embed_dim, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.seq_length   = seq_length
        self.vocab_size   = vocab_size
        self.embed_dim    = embed_dim
        self.dropout_rate = dropout          
        self.token_emb    = keras.layers.Embedding(vocab_size, embed_dim, name="token_emb")
        self.pos_emb      = keras.layers.Embedding(seq_length, embed_dim, name="pos_emb")
        self.drop         = keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        seq_len   = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        out = self.token_emb(x) + self.pos_emb(positions)
        return self.drop(out, training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_length": self.seq_length,
            "vocab_size":  self.vocab_size,
            "embed_dim":   self.embed_dim,
            "dropout":     self.dropout_rate,
        })
        return config

class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim    = embed_dim        
        self.num_heads    = num_heads        
        self.ff_dim       = ff_dim           
        self.dropout_rate = dropout  
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = keras.layers.Dropout(dropout)
        self.drop2 = keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]

        causal_mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0
        )  

        attn_out = self.attention(
            x, x,
            attention_mask=causal_mask,
            training=training
        )
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim":    self.ff_dim,
            "dropout":   self.dropout_rate,
        })
        return config

def build_transformer_model(
    vocab_size,
    seq_length,
    embed_dim=128,
    num_heads=4,
    ff_dim=512,
    num_layers=4,
    dropout=0.3
):
    inputs = keras.Input(shape=(None,), dtype="int32", name="token_ids")

    x = TokenAndPositionEmbedding(
        seq_length=seq_length,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        dropout=dropout,
        name="token_pos_embedding"
    )(inputs)

    for i in range(num_layers):
        x = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f"transformer_block_{i+1}"
        )(x)

    outputs = keras.layers.Dense(vocab_size, name="logits")(x)

    return keras.Model(inputs, outputs, name="ShakespeareTransformer")


if __name__ == "__main__":
    from architecture import build_lstm_model, build_transformer_model
    print("=== LSTM ===")
    build_lstm_model(vocab_size=65).summary()
    print("\n=== Transformer ===")
    build_transformer_model(vocab_size=65, seq_length=200).summary()
