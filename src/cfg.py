import torch

# cuda
device = torch.device("cpu")

# seq2seq
EncoderLSTM = {
    "embedding_dim": 64,
    "hidden_size": 256,
    "device": device,
}
DecoderLSTM = {
    "embedding_dim": 128,
    "hidden_size": 256,
    "device": device,
}

# training
n_iters = 1000
learning_rate = 1e-3
max_action_length = 100
