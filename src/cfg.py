import torch

# cuda
device = torch.device("cuda")

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
n_epochs = 40
learning_rate = 1e-3
max_action_len = 200
batch_size = 256
