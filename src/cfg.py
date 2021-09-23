import torch

seed = 47

checkpoint_dir = "models/default"

# cuda
device = torch.device("cuda")

# seq2seq
EncoderLSTM = {
    "embedding_dim": 64,
    "hidden_size": 256,
    "device": device,
}
DecoderLSTM = {
    "embedding_dim": 64,
    "hidden_size": 256,
    "device": device,
}

# training
learning_rate = 1e-2
max_action_len = 100
batch_size = 16
