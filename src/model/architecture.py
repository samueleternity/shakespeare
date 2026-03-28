import tensorflow as tf
from tensorflow import keras

def build_lstm_model(vocab_size, embed_dim=64, lstm_units=128, num_layers=2, dropout=0.2):
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


if __name__ == "__main__":
    model = build_lstm_model(vocab_size=65)
    model.summary()