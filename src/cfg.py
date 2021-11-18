seed = 47

# seq2seq
EncoderLSTM = {
    "embedding_dim": 64,
    "hidden_size": 128,
    "nlayers": 1,
    "dropout_p": 0.3,
    "bidirectional": True,
}
DecoderLSTM = {
    "embedding_dim": 64,
    "hidden_size": 256,
    "nlayers": 1,
    "dropout_p": 0.3,
    "context_size": 256,  # 2 * encoder.hidden_size
}

# training
learning_rate = 1e-3
batch_size = 32
max_recipe_len = 100
